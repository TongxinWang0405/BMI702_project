"""
Text encoders for EchoCare-CLIP: CLIP text tower and BERT text tower.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, BertModel


class CLIPTextEncoder(nn.Module):
    """
    CLIP text tower + MLP projection head.

    Input  : (B, max_seq_len) token ids
    Output : (B, projection_dim)  L2-normalized
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 256,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()

        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        text_out_dim = self.text_encoder.config.hidden_size  # 512 for base

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(text_out_dim, text_out_dim),
            nn.ReLU(),
            nn.Linear(text_out_dim, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.pooler_output          # (B, 512) — [EOS] token embedding
        text_emb = self.projection(text_emb)      # (B, projection_dim)
        return F.normalize(text_emb, dim=-1)


class BERTTextEncoder(nn.Module):
    """
    BERT text tower + MLP projection head.

    Input  : (B, max_seq_len) token ids
    Output : (B, projection_dim)  L2-normalized
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        projection_dim: int = 256,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()

        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        text_out_dim = self.text_encoder.config.hidden_size  # 768 for base

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(text_out_dim, text_out_dim),
            nn.ReLU(),
            nn.Linear(text_out_dim, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # BERT pooler_output is the [CLS] token representation passed through a Linear+Tanh layer
        text_emb = outputs.pooler_output          # (B, 768)
        text_emb = self.projection(text_emb)      # (B, projection_dim)
        return F.normalize(text_emb, dim=-1)
