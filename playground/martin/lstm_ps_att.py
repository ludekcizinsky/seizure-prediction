"""
Based on https://ieeexplore.ieee.org/document/10381774
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory

class SlidingWindowBatch(nn.Module):
    def __init__(self, window_size=125, step_size=125):
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
        return windows

class SignatureEncoder(nn.Module):
    def __init__(self, input_channels, depth=2, add_time=True):
        """
        Args:
            input_channels (int): number of input channels (without time).
            depth (int): signature depth.
            add_time (bool): whether to augment with time channel.
        """
        super().__init__()
        self.input_channels = input_channels
        self.depth = depth
        self.add_time = add_time
        
        self.total_channels = input_channels + 1 if add_time else input_channels
        self.output_channels = signatory.signature_channels(self.total_channels, self.depth)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch_size, windows, channels, time)

        Returns:
            Tensor: shape (batch_size, windows, output_channels)
        """
        batch_size, n_windows, n_time, n_channels = x.shape

        # Step 2: flatten batch and window dimensions
        x = x.reshape(batch_size * n_windows, n_time, n_channels)

        # Step 3: add time if needed
        if self.add_time:
            time = torch.linspace(0, 1, n_time, device=x.device).unsqueeze(0).unsqueeze(-1)
            time = time.expand(x.size(0), -1, -1)  # (batch_size * windows, time, 1)
            x = torch.cat([time, x], dim=-1)  # (batch_size * windows, time, channels+1)

        # Step 4: compute signature
        sig = signatory.signature(x, depth=self.depth)  # (batch_size * windows, output_channels)

        # Step 5: reshape back to (batch_size, windows, output_channels)
        sig = sig.view(batch_size, n_windows, self.output_channels)

        return sig
    
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

class LSTM_PS_ATT(nn.Module):
    def __init__(self, output_size=1, window_size=125, lstm_input_size=128, lstm_hidden_size=128, lstm_layers=3, bidirectional=False, attention_dim=64, dropout=0.5):
        super().__init__()

        # Sliding window batch layer
        self.sliding_window = SlidingWindowBatch(window_size=window_size)

        # Path Signature feature extraction
        self.signature_extraction = SignatureEncoder(input_channels=19, depth=2, add_time=True)

        # Projection to the size of the BiLSTM input
        self.projection = nn.Linear(self.signature_extraction.output_channels, lstm_input_size)

        self.norm = nn.BatchNorm1d(num_features=lstm_input_size)
        self.activation = nn.GELU()

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

    def forward(self, x):
        # Apply sliding window on input
        windows = self.sliding_window(x)  # shape: (batch_size, num_windows, window_size, 19)

        # Apply signature extraction
        windows = self.signature_extraction(windows) # shape: (batch_size, num_windows, output_dim)

        # Projection layer
        windows = self.projection(windows)  # shape: (batch_size, num_windows, lstm_input_dim)

        windows = self.norm(windows.transpose(1,2)).transpose(1,2)
        windows = self.activation(windows)

        # Apply BiLSTM
        lstm_out, _ = self.bilstm(windows)  # shape: (batch_size, num_windows, lstm_hidden_size * 2)

        # Apply Attention
        weighted_attention = self.attention(lstm_out) # shape: (batch_size, lstm_hidden_size * 2)

        # Final output layer
        out = self.output_layer(weighted_attention)  # shape: (batch_size, output_size)

        return out