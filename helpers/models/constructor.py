"""
Constructs a model based on the config file.
"""

import hydra
import torch.nn as nn
from omegaconf import DictConfig

from helpers.models.modules.combined import CombinedModel


def get_model(cfg: DictConfig) -> nn.Module:

    # Graph module
    if cfg.graph_module.is_enabled:
        graph_module = hydra.utils.instantiate(cfg.graph_module)
        graph_dim = cfg.graph_module.out_dim
    else:
        graph_module, graph_dim = None, None

    # Temporal module
    if cfg.temporal_module.is_enabled:
        temporal_module = hydra.utils.instantiate(cfg.temporal_module)
        temp_dim = cfg.temporal_module.out_dim
    else:
        temporal_module, temp_dim = None, None

    # Final model
    return CombinedModel(
        graph_module     = graph_module,
        temporal_module  = temporal_module,
        graph_out_dim    = graph_dim,
        temporal_out_dim = temp_dim,
    )
