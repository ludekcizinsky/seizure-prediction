import torch

import torch.nn as nn

class CNN1DTokenEmbedder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_layers=2, kernel_size=7):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels if i == 0 else embed_dim,
                    embed_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(embed_dim))
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool to 1 token per node

    def forward(self, x):
        # x: [batch, series_length]
        x = x.unsqueeze(1)  # [batch, 1, series_length]
        x = self.cnn(x)     # [batch, embed_dim, series_length]
        x = self.pool(x)    # [batch, embed_dim, 1]
        x = x.squeeze(-1)   # [batch, embed_dim]
        return x

class CNN1DAttentionClassifier(nn.Module):
    def __init__(
        self,
        series_length=3000,
        num_nodes=19,
        embed_dim=64,
        num_cnn_layers=2,
        num_transformer_layers=2,
        num_heads=4,
        mlp_dim=128,
        dropout=0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # One embedder per node (shared weights)
        self.token_embedder = CNN1DTokenEmbedder(
            in_channels=1,
            embed_dim=embed_dim,
            num_layers=num_cnn_layers
        )

        # Positional encoding for tokens (nodes)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_nodes, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        # x: [batch, series_length, num_nodes]
        x = x.permute(0, 2, 1)  # [batch, num_nodes, series_length]
        tokens = []
        for i in range(self.num_nodes):
            node_series = x[:, i, :]  # [batch, series_length]
            token = self.token_embedder(node_series)  # [batch, embed_dim]
            tokens.append(token)
        tokens = torch.stack(tokens, dim=1)  # [batch, num_nodes, embed_dim]
        tokens = tokens + self.pos_embed  # Add positional encoding

        attn_out = self.transformer(tokens)  # [batch, num_nodes, embed_dim]
        out = attn_out.mean(dim=1)  # Global average pooling over tokens

        # logits = self.classifier(out).squeeze(-1)  # [batch]
        logits = self.classifier(out)  # [batch, output_size=1]
        return logits

# # Example usage:
# model = CNN1DAttentionClassifier()
# x = torch.randn(8, 3000, 19)  # batch of 8
# logits = model(x)
# print(logits.shape)  # Should be [8]
# print(logits)