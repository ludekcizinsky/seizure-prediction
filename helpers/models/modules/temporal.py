import torch
import math
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

    def forward(self, x):

        return self.lstm(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class TransformerEncoder(nn.Module):

    def __init__(self, transformer_dim, num_heads, num_layers, batch_first, dropout):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.dropout = dropout
        self.num_layers = num_layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):

        return self.transformer_encoder(x)