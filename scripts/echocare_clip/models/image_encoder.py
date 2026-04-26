"""
EchoCare Image Encoder: SwinTransformer backbone + MLP projection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer


class EchoCareImageEncoder(nn.Module):
    """
    EchoCare SwinTransformer backbone + MLP projection head.

    Input  : (B, 3, 256, 256)
    Output : (B, projection_dim)  L2-normalized
    """

    def __init__(
        self,
        feature_size: int = 128,
        in_channels: int = 3,
        projection_dim: int = 256,
        use_checkpoint: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=[8, 8],
            patch_size=[2, 2],
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
            use_v2=True,
        )

        encoder_out_dim = feature_size * (2 ** 4)  # 128 * 16 = 2048

        self.projection = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Linear(encoder_out_dim, projection_dim),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def load_pretrained(self, checkpoint_path: str, device: str = "cpu"):
        """Load EchoCare MAE pretrained weights (removes mask_token artifact)."""
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict.pop("mask_token", None)
        self.encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded EchoCare pretrained weights from '{checkpoint_path}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)           # list of feature maps
        deep = feats[-1]                  # (B, 2048, 8, 8)
        emb = deep.mean(dim=(2, 3))       # GAP → (B, 2048)
        emb = self.projection(emb)        # (B, projection_dim)
        return F.normalize(emb, dim=-1)
