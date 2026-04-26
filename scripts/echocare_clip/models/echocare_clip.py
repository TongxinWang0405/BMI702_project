"""
Main EchoCare-CLIP dual-encoder model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoCare_CLIP(nn.Module):
    """
    CLIP-style contrastive model combining EchoCare image encoder and a text encoder.

    Training objective: symmetric InfoNCE loss over matched (image, text) pairs.
    Temperature: learnable log_temperature, initialized at log(init_temperature).
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, images, input_ids, attention_mask=None):
        image_emb = self.encode_image(images)                    # (B, D)
        text_emb = self.encode_text(input_ids, attention_mask)   # (B, D)

        temperature = self.log_temperature.exp()
        logits_per_image = (image_emb @ text_emb.T) / temperature  # (B, B)
        logits_per_text = logits_per_image.T

        labels = torch.arange(image_emb.size(0), device=image_emb.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

        return loss, image_emb, text_emb
