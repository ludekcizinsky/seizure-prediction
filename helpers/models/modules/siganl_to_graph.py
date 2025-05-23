"""
In this module, we prepare the data for our graph models.
"""


import torch.nn as nn
import torch
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch_geometric.data import Data, Batch

class DistanceGraphBuilder(nn.Module):
    def __init__(self, distance_matrix, distance_threshold):
        super().__init__()
        self.distance_matrix = distance_matrix
        self.distance_threshold = distance_threshold

    def forward(self, x):
        data_list = []
        for eeg_sample in x:
            mask = self.distance_matrix <= self.threshold
            np.fill_diagonal(mask, 0)
            adj = mask.astype(float)
            adj_tensor = torch.tensor(adj, dtype=torch.float)
            edge_index, edge_weight = dense_to_sparse(adj_tensor)
            node_features = eeg_sample  # [19, 3000]
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)
        return Batch.from_data_list(data_list)

class CorrelationGraphBuilder(nn.Module):
    def __init__(self, correlation_threshold):
        super().__init__()
        self.correlation_threshold = correlation_threshold

    def forward(self, x):
        data_list = []
        for eeg_sample in x:
            corr = np.corrcoef(eeg_sample.numpy())
            np.fill_diagonal(corr, 0)
            mask = np.abs(corr) >= self.correlation_threshold
            adj = corr * mask
            adj_tensor = torch.tensor(adj, dtype=torch.float)
            edge_index, edge_weight = dense_to_sparse(adj_tensor)
            node_features = eeg_sample  # [19, 3000]
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)
        return Batch.from_data_list(data_list)

# Learnable Graph Learner
class LearnableGraphLearner(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, x):
        data_list = []
        for eeg_sample in x:
            adj = torch.sigmoid(self.adj)
            adj = (adj + adj.T) / 2  # Make symmetric
            edge_index, edge_weight = dense_to_sparse(adj)
            node_features = eeg_sample  # [19, 3000]
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)
        return Batch.from_data_list(data_list)