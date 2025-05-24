"""
In this module, we prepare the data for our graph models.
"""


import torch.nn as nn
import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch_geometric.data import Data, Batch
import pandas as pd

class DistanceGraphBuilder(nn.Module):
    def __init__(
        self,
        distance_file: str,
        distance_threshold: float,
        device: torch.device | str = "cuda:0",
        **kwargs
    ):
        super().__init__()
        # 1) read CSV → NumPy → Torch tensor
        df = pd.read_csv(distance_file)
        mat = df.pivot(index="from", columns="to", values="distance").values
        dist = torch.from_numpy(mat).to(device=device, dtype=torch.float32)

        # 2) build boolean mask & zero out diagonal in-place
        mask = dist <= distance_threshold        # (N, N) bool tensor
        mask.fill_diagonal_(False)               # zero self-loops

        # 3) convert to float adjacency & sparse format
        adj = mask.to(torch.float32)             # (N, N) float tensor
        edge_index, edge_weight = dense_to_sparse(adj)

        # 4) register as buffers so they move with .to(device)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

    def forward(self, x: torch.Tensor) -> Batch:
        """
        Args:
            x: (B, N, F)  batch of node-feature matrices
        Returns:
            A torch_geometric.data.Batch with the same edge_index/edge_weight
        """
        data_list = []
        for node_feats in x:
            # node_feats: (N, F)
            node_feats = node_feats.transpose(-2,-1)
            data_list.append(
                Data(
                    x=node_feats,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_weight
                )
            )
        return Batch.from_data_list(data_list)


class CorrelationGraphBuilder(nn.Module):
    def __init__(self, correlation_threshold, **kwargs):
        super().__init__()
        self.correlation_threshold = correlation_threshold

    def forward(self, x):
        data_list = []
        for node_features in x:
            corr = np.corrcoef(node_features.numpy())
            np.fill_diagonal(corr, 0)
            mask = np.abs(corr) >= self.correlation_threshold
            adj = corr * mask
            adj_tensor = torch.tensor(adj, dtype=torch.float)
            edge_index, edge_weight = dense_to_sparse(adj_tensor)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)
        return Batch.from_data_list(data_list)


class LearnableGraphLearner(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, x):
        data_list = []
        for node_features in x:
            adj = torch.sigmoid(self.adj)
            adj = (adj + adj.T) / 2  # Make symmetric
            edge_index, edge_weight = dense_to_sparse(adj)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)
        return Batch.from_data_list(data_list)