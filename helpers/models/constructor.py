"""
This modules allows us to run both graph and temporal modules in a single forward pass.
We can run in the following modes:
- temp -> classifications
- graph -> classifications
- temp -> graph -> classifications
"""

import torch.nn as nn
from omegaconf import DictConfig
import hydra


class ModulardModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        if cfg.signal_transform.is_neural:
            self.neural_signal_transform = hydra.utils.instantiate(cfg.signal_transform)
        else:
            self.neural_signal_transform = None

        self.temporal = hydra.utils.instantiate(cfg.temporal_module) if cfg.temporal_module.is_enabled else None
        self.graph_builder = hydra.utils.instantiate(cfg.graph_builder) if cfg.graph_builder.is_enabled else None
        self.graph = hydra.utils.instantiate(cfg.graph_module) if cfg.graph_module.is_enabled else None


    def forward(self, x):
        if self.neural_signal_transform is not None:
            x = self.neural_signal_transform(x)

        # Only‐temporal case
        if self.temporal is not None and self.graph is None:
            return self.temporal(x)

        # Only‐graph case
        if self.graph is not None and self.temporal is None:
            graph_batch = self.graph_builder(x)
            return self.graph(graph_batch)

        # Both modules case: temp -> graph -> classifier
        raise NotImplementedError("Both modules case is not implemented yet")