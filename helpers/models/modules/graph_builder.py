"""
In this module, we prepare the data for our graph models.
"""


import torch.nn as nn
import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch_geometric.data import Data, Batch
import pandas as pd


class BaseGraphBuilder(nn.Module):
    """
    Base class that provides a method for chunking a batched time‐series tensor
    into sliding windows of shape (B, W, N, window_size), where N is the number
    of nodes (e.g. 19 channels). Subclasses can call `chunk_time_windows_as_node_feats`
    whenever `use_windows=True`.
    """
    def __init__(
        self,
        use_windows: bool = False,
        window_size: int = 100,
        stride: int = 50,
    ):
        super().__init__()
        self.use_windows = use_windows
        self.window_size = window_size
        self.stride = stride

    @staticmethod
    def chunk_time_windows_as_node_feats(
        x: torch.Tensor,
        window_size: int,
        stride: int
    ) -> torch.Tensor:
        """
        Splits a batch of time‐series X into sliding windows, then arranges each window
        so that the N channels become "nodes" and each node’s feature vector is the
        windowed time‐slice.

        Args:
            x (torch.Tensor): shape = (B, T, N). Here:
                              - B = batch size
                              - T = total number of time‐steps
                              - N = number of channels (nodes in the GNN)
            window_size (int): length of each time‐window (i.e. the node‐feature dimension).
            stride (int):      hop between window starts.

        Returns:
            torch.Tensor of shape (B, W, N, window_size), where
              W = floor((T - window_size) / stride) + 1.

        Notes:
          1. If T < window_size, this returns W = 0 (no windows).
          2. Any trailing segment shorter than window_size is dropped. If you want to
             include it, pad T so that (T_padded - window_size) is divisible by stride
             before calling this.
        """
        B, T, N = x.shape
        # 1) Permute to (B, N, T) so we can call unfold on time‐axis (dim=2)
        x = x.permute(0, 2, 1)  # now shape = (B, N, T)

        # 2) Use unfold to carve sliding windows along time (dim=2):
        #    → (B, N, W, window_size), where
        #      W = floor((T - window_size) / stride) + 1
        x_unf = x.unfold(dimension=2, size=window_size, step=stride)
        #    x_unf.shape == (B, N, W, window_size)

        # 3) Permute so that each window appears as (N, window_size):
        #    → (B, W, N, window_size)
        result = x_unf.permute(0, 2, 1, 3).contiguous()
        return result

    def prepare_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts the input tensor into a flattened batch of node-feature matrices
        of shape (num_graphs, N, F), where:
          - If use_windows=False: x is assumed (B, F, N) → returns (B, N, F).
          - If use_windows=True:  x is assumed (B, T, N) → chunk into windows to get
                                  (B, W, N, window_size) → returns (B*W, N, window_size).
        """
        if self.use_windows:
            # x: (B, T, N)
            windows = self.chunk_time_windows_as_node_feats(
                x,
                window_size=self.window_size,
                stride=self.stride
            )
            # windows.shape == (B, W, N, window_size)
            B, W, N, F = windows.shape
            self.batch_size = B

            # Flatten to (B*W, N, F)
            x_prepared = windows.view(B * W, N, F)

        else:
            # x: (B, F, N) → transpose to (B, N, F)
            x_prepared = x.permute(0, 2, 1).contiguous()
            self.batch_size = x.shape[0]
            # x_prepared.shape == (B, N, F)

        return x_prepared

    def reshape_gnn_output(self, gnn_out: torch.Tensor) -> torch.Tensor:
        """
        Given the GNN’s output of shape (num_graphs, hidden_dim), reshape it to
        (batch_size, num_windows, hidden_dim) if use_windows=True, or to
        (batch_size, 1, hidden_dim) otherwise.

        Args:
            gnn_out (torch.Tensor): Tensor of shape (num_graphs, hidden_dim).
            batch_size (int): Original batch size B.

        Returns:
            torch.Tensor: 
                - If use_windows == True: shape (B, W, hidden_dim), where W = num_graphs // B.
                - If use_windows == False: shape (B, 1, hidden_dim).
        """
        num_graphs, hidden_dim = gnn_out.shape

        if self.use_windows:
            # number of windows per batch = (B*W) // B
            num_windows = num_graphs // self.batch_size
            return gnn_out.view(self.batch_size, num_windows, hidden_dim)
        else:
            # no windowing: treat each graph as the only “window”
            return gnn_out.view(self.batch_size, 1, hidden_dim)

class DistanceGraphBuilder(BaseGraphBuilder):
    """
    Builds a torch_geometric.data.Batch of graphs based on distance thresholds.
    Each graph has N nodes (e.g. 19 channels) with per-node features. If use_windows=True,
    the input is assumed to be a batched time-series of shape (B, T, N); otherwise,
    the input is assumed to be (B, F, N).

    Subclass of BaseGraphBuilder. Loads a distance CSV to construct a fixed adjacency.
    """
    def __init__(
        self,
        distance_file: str,
        distance_threshold: float,
        use_windows: bool = False,
        window_size: int = 100,
        stride: int = 50,
        device: torch.device | str = "cuda:0",
        **kwargs
    ):
        """
        Args:
            distance_file (str): Path to CSV with columns [from, to, distance].
            distance_threshold (float): Max distance to consider an edge.
            use_windows (bool): If True, forward() expects x of shape (B, T, N).
                                If False, forward() expects x of shape (B, F, N).
            window_size (int): Window length (used only if use_windows=True).
            stride (int): Step between window starts (used only if use_windows=True).
            device (torch.device or str): Where to place adjacency tensors.
        """
        super().__init__(
            use_windows=use_windows,
            window_size=window_size,
            stride=stride,
        )
        self.device = device

        # --- Load distance matrix from CSV ---
        df = pd.read_csv(distance_file)
        mat = df.pivot(index="from", columns="to", values="distance").values
        dist = torch.from_numpy(mat).to(device=device, dtype=torch.float32)

        # Build boolean adjacency mask (no self‐loops)
        mask = dist <= distance_threshold    # shape (N, N)
        mask.fill_diagonal_(False)

        # Convert to float adjacency and sparse format
        adj = mask.to(torch.float32)
        edge_index, edge_weight = dense_to_sparse(adj)

        # Register as buffers so they move with .to(device)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

    def forward(self, x: torch.Tensor) -> Batch:
        """
        Args:
            If self.use_windows == False:
                x: torch.Tensor of shape (B, F, N).  Each sample i is (F, N); we
                   transpose → (N, F) to form node-feature matrix for graph i.
            If self.use_windows == True:
                x: torch.Tensor of shape (B, T, N).  We chunk into windows of
                   length=self.window_size with step=self.stride, producing
                   (B, W, N, window_size). We then flatten to (B*W, N, window_size).

        Returns:
            A torch_geometric.data.Batch containing:
              • B graphs (if use_windows=False), each with x.shape = (N, F).
              • B * W graphs (if use_windows=True), each with x.shape = (N, window_size).
        """
        data_list = []

        # Prepare x so that it is (num_graphs, N, F)
        # where N is the number of nodes and F is the feature dimension
        x_prepared = self.prepare_node_features(x)

        # At this point, x is always of shape (num_graphs, N, F),
        # where num_graphs = B (if no windows) or B*W (if windows).
        for node_feats in x_prepared:
            # node_feats has shape (N, F)
            data_list.append(
                Data(
                    x=node_feats,               # (N, F)
                    edge_index=self.edge_index, # (2, E)
                    edge_attr=self.edge_weight  # (E,)
                )
            )

        return Batch.from_data_list(data_list)


class CorrelationGraphBuilder(BaseGraphBuilder):
    """
    Builds graphs based on Pearson correlation between node features.
    Inherits windowing functionality from BaseGraphBuilder.

    If use_windows=False:
        - Forward expects x of shape (B, F, N), where F is feature dim, N is number of nodes.
    If use_windows=True:
        - Forward expects x of shape (B, T, N), where T is time length; it will chunk into windows:
          (B, T, N) → (B, W, N, window_size) → flatten to (B*W, N, window_size).
    """
    def __init__(
        self,
        correlation_threshold: float,
        use_windows: bool = False,
        window_size: int = 100,
        stride: int = 50,
        **kwargs
    ):
        super().__init__(use_windows=use_windows, window_size=window_size, stride=stride)
        self.correlation_threshold = correlation_threshold

    def forward(self, x: torch.Tensor) -> Batch:
        """
        Args:
            If use_windows=False: x is (B, F, N)
            If use_windows=True:  x is (B, T, N)

        Returns:
            Batch of graphs, one per sample (if no windows) or one per window (if windows).
            Each graph has N nodes and edge weights based on thresholded correlation.
        """
        data_list = []

        # Prepare x as (num_graphs, N, F)
        x_prepared = self.prepare_node_features(x)

        for node_feats in x_prepared:
            # node_feats: shape = (N, F)
            # Compute Pearson correlation matrix between N rows
            # Ensure float32
            nf = node_feats.float()
            # torch.corrcoef expects shape (N, F) and returns (N, N)
            corr = torch.corrcoef(nf)
            # Zero out diagonal
            corr.fill_diagonal_(0.0)
            # Threshold to build adjacency
            mask = torch.abs(corr) >= self.correlation_threshold
            adj = corr * mask.to(corr.dtype)
            edge_index, edge_weight = dense_to_sparse(adj)
            data_list.append(
                Data(
                    x=node_feats,          # (N, F)
                    edge_index=edge_index, # (2, E)
                    edge_attr=edge_weight  # (E,)
                )
            )

        return Batch.from_data_list(data_list)


class LearnableGraphLearner(BaseGraphBuilder):
    """
    Learns a fully-connected adjacency matrix via a trainable parameter.
    Inherits windowing functionality from BaseGraphBuilder.

    If use_windows=False:
        - Forward expects x of shape (B, F, N).
    If use_windows=True:
        - Forward expects x of shape (B, T, N) and chunks into windows.
    """
    def __init__(
        self,
        num_nodes: int,
        use_windows: bool = False,
        window_size: int = 100,
        stride: int = 50,
        **kwargs
    ):
        super().__init__(use_windows=use_windows, window_size=window_size, stride=stride)
        # Learnable adjacency logits, shape (N, N)
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> Batch:
        """
        Args:
            If use_windows=False: x is (B, F, N)
            If use_windows=True:  x is (B, T, N)

        Returns:
            Batch of graphs, one per sample (or per window if windowing).
            Edges are learned via a sigmoid( adj_logits ) symmetric matrix.
        """
        data_list = []

        # Prepare x as (num_graphs, N, F)
        x_prepared = self.prepare_node_features(x)

        # Compute symmetric adjacency once
        adj = torch.sigmoid(self.adj_logits)
        adj = (adj + adj.T) / 2.0
        # Zero diagonal to avoid self-loops
        adj = adj.fill_diagonal_(0.0)

        edge_index, edge_weight = dense_to_sparse(adj)

        for node_feats in x_prepared:
            # node_feats: shape = (N, F)
            data_list.append(
                Data(
                    x=node_feats,          # (N, F)
                    edge_index=edge_index, # (2, E)
                    edge_attr=edge_weight  # (E,)
                )
            )

        return Batch.from_data_list(data_list)
