import torch.nn as nn
from torch_geometric.nn import (
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

        prev_channels = None
        for conv in conv_layers.values():
            self.convs.append(conv)
            # determine feature dim after this conv
            out_ch = conv.out_channels
            heads = getattr(conv, "heads", 1)
            # for attention convs, default concat=True if attribute exists
            concat = getattr(conv, "concat", False)
            feat_dim = out_ch * heads if concat else out_ch
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(feat_dim))
            prev_channels = feat_dim

        self.feature_dim = prev_channels

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

        # optional graph-level classifier
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # apply each GNN layer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = self.act_fn(x)

        # pool to graph-level embedding
        g = self.pool_fn(x, batch)

        # return logits if head exists, else embeddings
        return self.classifier(g) if self.classifier is not None else g
