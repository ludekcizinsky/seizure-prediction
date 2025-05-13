import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification

class PatchTSTWrapper(nn.Module):
    def __init__(self,
        num_input_channels=19,
        num_targets=1,
        context_length=354,
        patch_length=12,
        patch_stride=12,
        num_hidden_layers=3,
        d_model=128,
        num_attention_heads=4,
        share_embedding=True, # share embedding across channels
        share_projection=False, # share projection across channels
        channel_attention=False, # let channels attend to each other
        norm_type="layernorm", # batchnorm or layernorm
        use_cls_token=True,
        attention_dropout=0.1,
        positional_dropout=0.1,
        patch_dropout=0.1,
    ):
        super().__init__()
        cfg   = PatchTSTConfig(
            num_input_channels=num_input_channels,
            num_targets=num_targets,
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
        model = PatchTSTForClassification(config=cfg)

        self.model = model

    def forward(self, x):
        logits = self.model(
            past_values=x,
        ).prediction_logits
        return logits
