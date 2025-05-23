import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
import hydra


class ModularGraph(nn.Module):
    def __init__(
        self,
        conv_layers: list,
        activation: str = "gelu",
        use_batchnorm: bool = False,
        pool_type: str = "mean",
        num_classes: int | None = None,
        **kwargs
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.act_fn = nn.GELU() if activation.lower()=="gelu" else nn.ReLU()
        self.convs = nn.ModuleList()
        if use_batchnorm:
            self.bns = nn.ModuleList()

        # instantiate convolutional GNN layers from config
        for conv in conv_layers: 
            self.convs.append(conv)
            if use_batchnorm:
                # BatchNorm over node features
                self.bns.append(nn.BatchNorm1d(conv.out_channels))

        # output feature dimension = out_channels of last conv
        self.feature_dim = self.convs[-1].out_channels

        # select pooling function
        pool_type = pool_type.lower()
        if pool_type == "mean":
            self.pool_fn = global_mean_pool
        elif pool_type == "max":
            self.pool_fn = global_max_pool
        elif pool_type in ("sum", "add"):
            self.pool_fn = global_add_pool
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

        # optional graph-level classification head
        if num_classes is not None:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, data):
        # data: torch_geometric.data.Batch
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # apply each GNN layer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = self.act_fn(x)

        # pool node features to get graph-level embedding
        g = self.pool_fn(x, batch)  # shape: (B, feature_dim)

        # if classifier exists, return logits per graph, shape (B, num_classes)
        return self.classifier(g) if self.classifier is not None else g
