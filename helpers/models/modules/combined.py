import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class CombinedModel(nn.Module):
    def __init__(
        self,
        graph_module: nn.Module = None,
        temporal_module: nn.Module = None,
        graph_out_dim: int = None,
        temporal_out_dim: int = None,
    ):
        super().__init__()
        self.graph = graph_module
        self.temporal = temporal_module

        # Case A: both modules → need a fusion head
        if self.graph is not None and self.temporal is not None:
            assert graph_out_dim is not None and temporal_out_dim is not None
            combine_hidden = temporal_out_dim // 2
            self.classifier = nn.Sequential(
                nn.Linear(graph_out_dim + temporal_out_dim, combine_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(combine_hidden, 1)
            )

    def forward(self, x):
        # Only‐graph case
        if self.graph is not None and self.temporal is None:
            # we assume graph_module itself returns final logits
            edge_index = ...
            batch = ...
            return self.graph(x, edge_index, batch)

        # Only‐temporal case
        if self.temporal is not None and self.graph is None:
            # we assume temporal_module itself returns final logits
            return self.temporal(x)

        # Both modules case
        # 1) Graph → (N, D) → pool → (B, D)
        h = self.graph(x, edge_index)              # (N, Dg)
        h = global_mean_pool(h, batch)             # (B, Dg)

        # 2) Temporal → (B, Dt)
        t = self.temporal(x)               # (B, Dt)

        # 3) Concat & classify → (B, num_classes)
        z = torch.cat([h, t], dim=-1)
        return self.classifier(z)
