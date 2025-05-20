import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.gcn1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)

        self.gcn2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)

        self.activation = nn.GELU()
        
    def forward(self, x, edge_index):

        x = self.gcn1(x, edge_index)
        
        x = self.activation(x)

        x = self.gcn2(x, edge_index)
        
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, heads_1=8, heads_2=1, hidden_channels=32):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=heads_1, concat=True)

        self.gat2 = GATv2Conv(in_channels=hidden_channels*heads_1, out_channels=hidden_channels, heads=heads_2, concat=False)

        self.activation = nn.GELU()
        
    def forward(self, x, edge_index):

        x = self.gcn1(x, edge_index)
        
        x = self.activation(x)

        x = self.gcn2(x, edge_index)

        return x