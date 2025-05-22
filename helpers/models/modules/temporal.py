import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

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


class Modular1DCNN(nn.Module):
    def __init__(self, in_channels, conv_layers, use_batchnorm, num_classes, **kwargs):
        super().__init__()
        layers = []
        in_ch = in_channels

        for layer in conv_layers:
            layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding
            ))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer.out_channels))
            layers.append(nn.ReLU())
            if layer.pool:
                layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = layer.out_channels

        self.features = nn.Sequential(*layers)
        if num_classes is not None:
            self.classifier = nn.Linear(in_ch, num_classes)
        else:
            self.classifier = None

    def forward(self, x):

        # channel first
        x = x.permute(0, 2, 1)

        # x: (B, C, T)
        x = self.features(x)

        if self.classifier is not None:
            # global average pool to (B, C, 1) → (B, C)
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            return self.classifier(x)
        else:
            return x



if __name__ == "__main__":

    from omegaconf import OmegaConf
    from hydra import initialize, compose

    B, T, C = 2, 3000, 19
    x = torch.randn(B, C, T)

    with initialize(config_path="../../../configs", version_base="1.1"):
        cfg = compose(config_name="train")

    print(OmegaConf.to_yaml(cfg))
    model = Modular1DCNN(cfg)
    print(model(x).shape)

