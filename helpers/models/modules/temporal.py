import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, dropout):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

    def forward(self, x):

        return self.lstm(x)
    
class TransformerEncoder(nn.Module):

    def __init__(self, transformer_dim, num_heads, num_layers, batch_first, dropout):
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.dropout = dropout
        self.num_layers = num_layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):

        return self.transformer_encoder(x)