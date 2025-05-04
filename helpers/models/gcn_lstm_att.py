"""
Based on https://www.sciencedirect.com/science/article/pii/S1746809422004098#fig4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SlidingWindowBatch(nn.Module):
    def __init__(self, window_size=125, step_size=62):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size

    def forward(self, data):
        batch_size, signal_len, channels = data.shape
        num_windows = (signal_len - self.window_size) // self.step_size + 1

        windows = []
        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_size
            window = data[:, start:end, :]  # shape: (batch_size, window_size, 19)
            windows.append(window)

        windows = torch.stack(windows, dim=1)  # shape: (batch_size, num_windows, window_size, 19)
        return windows.transpose(-2,-1) # shape: (batch_size, num_windows, 19, window_size)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.gcn1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)

        self.gcn2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)

        self.activation = nn.GELU()
        
    def forward(self, x, edge_index):
        # First GAT layer
        x = self.gcn1(x, edge_index)  # shape: (batch_size * num_windows, 19, hidden_channels * num_heads_1)
        
        x = self.activation(x)

        # Second GAT layer with 1 attention head
        x = self.gcn2(x, edge_index)  # shape: (batch_size * num_windows, 19, hidden_channels)
        
        return x
    
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        """
        Args:
            hidden_dim: dimension of BiLSTM output (hidden_size * 2 if BiLSTM)
            attn_dim: dimension of intermediate MLP hidden layer
        """
        super().__init__()
        self.attention_mlp = nn.Linear(hidden_dim, attn_dim)
        self.context_vector = nn.Parameter(torch.randn(attn_dim))

    def forward(self, h):
        """
        Args:
            h: Tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            s: Tensor of shape (batch_size, hidden_dim)
        """
        # Step 1: compute ut = tanh(Wh + b)
        u = torch.tanh(self.attention_mlp(h))  # (batch_size, seq_len, attn_dim)

        # Step 2: compute scores: dot(u_t, u_w)
        # context_vector: (attn_dim,)
        scores = torch.matmul(u, self.context_vector)  # (batch_size, seq_len)

        # Step 3: softmax over time
        alpha = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        # Step 4: weighted sum of h
        s = torch.bmm(alpha.unsqueeze(1), h)  # (batch_size, 1, hidden_dim)
        s = s.squeeze(1)  # (batch_size, hidden_dim)

        return s

class GCN_LSTM_ATT(nn.Module):
    def __init__(self, gcn_in_features=64, gcn_out_features=32, output_size=1, window_size=125, step_size=62, lstm_input_size=128, lstm_hidden_size=128, lstm_layers=3, bidirectional=False, attention_dim=64, dropout=0.5):
        super().__init__()

        # Sliding window batch layer
        self.sliding_window = SlidingWindowBatch(window_size=window_size, step_size=step_size)

        # Projection layer from 125 to 64 features
        self.projection = nn.Linear(window_size, gcn_in_features)

        # GAT layers
        self.gcn = GCNLayer(in_channels=gcn_in_features, hidden_channels=gcn_out_features)

        # Flattening the output and passing through a fully connected layer before BiLSTM
        self.fc = nn.Linear(19 * gcn_out_features, lstm_input_size)  # 19 nodes * 32 features per node

        # LSTM
        self.bilstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # Attention
        if bidirectional:
            self.attention = TemporalAttention(lstm_hidden_size*2, attention_dim)
        else:
            self.attention = TemporalAttention(lstm_hidden_size, attention_dim)

        # Final output layer
        if bidirectional:
            self.output_layer = nn.Linear(lstm_hidden_size*2, output_size)
        else:
            self.output_layer = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x, edge_index):
        # Apply sliding window on input
        windows = self.sliding_window(x)  # shape: (batch_size, num_windows, 19, window_size)

        # Projection layer
        windows_proj = self.projection(windows)  # shape: (batch_size, num_windows, 19, 64)

        # Reshape to (batch_size * num_windows, 19, 64) for GCN
        batch_size, num_windows, nodes, features = windows_proj.shape
        windows_proj = windows_proj.view(batch_size * num_windows, nodes, features)  # shape: (batch_size * num_windows, 19, 64)
        
        # Make the tensor contiguous before passing it to GCN
        windows_proj = windows_proj.contiguous()

        # Apply GCN
        gcn_out = self.gcn(windows_proj, edge_index)  # shape: (batch_size * num_windows, 19, 32)

        # Reshape back to (batch_size, num_windows, 19, 32)
        gcn_out = gcn_out.view(batch_size, num_windows, nodes, -1)  # shape: (batch_size, num_windows, 19, 32)

        # Flatten each window and pass through fully connected layer
        windows_flat = gcn_out.view(batch_size, num_windows, -1)  # shape: (batch_size, num_windows, 19 * 32)
        
        # Make the tensor contiguous before passing to FC layer
        windows_flat = windows_flat.contiguous()
        windows_flat = F.gelu(self.fc(windows_flat))  # shape: (batch_size, num_windows, 128)

        # Apply BiLSTM
        lstm_out, _ = self.bilstm(windows_flat)  # shape: (batch_size, num_windows, lstm_hidden_size * 2)

        # Apply Attention
        weighted_attention = self.attention(lstm_out) # shape: (batch_size, lstm_hidden_size * 2)
        
        # Final output layer
        out = self.output_layer(weighted_attention)  # shape: (batch_size, output_size)

        return out
