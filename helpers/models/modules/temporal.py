import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import PatchTSTConfig, PatchTSTForClassification

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout, num_classes, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional, 
            dropout=dropout
        )
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = None


    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        out, (h_n, c_n) = self.lstm(x)  # out shape: [batch_size, seq_len, hidden_dim]
        last_timestep = out[:, -1, :]  # [batch_size, hidden_dim]
        if self.classifier is not None:
            logits = self.classifier(last_timestep)  # [batch_size, 1]
            return logits
        else:
            return last_timestep
    
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
    
class ModularTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_classes: int = None,
        **kwargs
    ):
        super().__init__()
        # project 19 channels → transformer_dim
        self.input_proj = nn.Linear(in_channels, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # optional classification head
        if num_classes is not None:
            self.classifier = nn.Linear(transformer_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x):
        # x: (B, T, C) where C == in_channels
        x = self.input_proj(x)          # → (B, T, transformer_dim)
        x = self.transformer(x)         # → (B, T, transformer_dim)

        if self.classifier is not None:
            # pool over time → (B, transformer_dim)
            x = x.mean(dim=1)
            return self.classifier(x)   # → (B, num_classes)
        else:
            # return full sequence of features
            return x                   # → (B, T, transformer_dim)


class PatchTSTWrapper(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 19,
        num_targets: int = None,
        context_length: int = 354,
        patch_length: int = 12,
        patch_stride: int = 12,
        num_hidden_layers: int = 3,
        d_model: int = 128,
        num_attention_heads: int = 4,
        share_embedding: bool = True,
        share_projection: bool = False,
        channel_attention: bool = False,
        norm_type: str = "layernorm",
        use_cls_token: bool = True,
        attention_dropout: float = 0.1,
        positional_dropout: float = 0.1,
        patch_dropout: float = 0.1,
        **_
    ):
        super().__init__()
        cfg = PatchTSTConfig(
            num_input_channels=num_input_channels,
            num_targets=num_targets or 0,    # dummy if None
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            use_cls_token=use_cls_token,
            num_hidden_layers=num_hidden_layers,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            share_embedding=share_embedding,
            channel_attention=channel_attention,
            ffn_dim=4*d_model,
            share_projection=share_projection,
            norm_type=norm_type,
            attention_dropout=attention_dropout,
            positional_dropout=positional_dropout,
            patch_dropout=patch_dropout,
        )

        self.has_classifier = num_targets is not None

        if self.has_classifier:
            # standard classification model
            self.model = PatchTSTForClassification(config=cfg)
        else:
            # base model without the classification head
            self.model = PatchTSTModel(config=cfg)

    def forward(self, x):
        """
        x: (B, T=context_length, C=num_input_channels)
        """
        if self.has_classifier:
            out = self.model(past_values=x, return_dict=True)
            return out.prediction_logits       # (B, num_targets)
        else:
            out = self.model(past_values=x, return_dict=True)
            # last_hidden_state: (B, num_patches, d_model)
            return out.last_hidden_state


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

