"""
Utilities for model building, loading, and tokenization.
"""

import os
import torch
from transformers import CLIPTokenizer, AutoTokenizer

from .image_encoder import EchoCareImageEncoder
from .text_encoder import CLIPTextEncoder, BERTTextEncoder
from .echocare_clip import EchoCare_CLIP


def get_tokenizer(text_encoder_type="clip", model_name="openai/clip-vit-base-patch32"):
    """Get the appropriate tokenizer based on the text encoder type."""
    if text_encoder_type == "clip":
        return CLIPTokenizer.from_pretrained(model_name)
    elif text_encoder_type == "bert":
        return AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown text encoder type: {text_encoder_type}")


def get_max_seq_len(cfg: dict, text_encoder_type: str) -> int:
    """Return the correct max sequence length for the given encoder type."""
    if text_encoder_type == "bert":
        return cfg.get("bert_max_seq_len", 128)
    return cfg.get("max_seq_len", 77)


def tokenize_texts(texts, tokenizer, max_length=77, device="cpu"):
    """
    Tokenize a list of strings.
    Returns (input_ids, attention_mask) both shape (B, max_length).
    """
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


def build_model(
    config,
    text_encoder_type="clip",
    freeze_image=True,
    freeze_text=True,
    device="cpu"
) -> EchoCare_CLIP:
    """
    Instantiate EchoCare_CLIP, load pretrained EchoCare image encoder weights,
    and apply freeze flags for each encoder.
    """
    img_enc = EchoCareImageEncoder(
        feature_size=config["feature_size"],
        in_channels=config["in_channels"],
        projection_dim=config["projection_dim"],
        freeze_encoder=freeze_image,
    )

    echocare_checkpoint = config.get("echocare_checkpoint")
    if echocare_checkpoint and os.path.exists(echocare_checkpoint):
        img_enc.load_pretrained(echocare_checkpoint, device=device)
    else:
        print("[WARNING] EchoCare MAE checkpoint not found (or null) — using random SwinTransformer weights.")

    if text_encoder_type == "clip":
        txt_enc = CLIPTextEncoder(
            clip_model_name=config["clip_model_name"],
            projection_dim=config["projection_dim"],
            freeze_text_encoder=freeze_text,
        )
    elif text_encoder_type == "bert":
        txt_enc = BERTTextEncoder(
            bert_model_name=config["bert_model_name"],
            projection_dim=config["projection_dim"],
            freeze_text_encoder=freeze_text,
        )
    else:
        raise ValueError(f"Unknown text encoder type: {text_encoder_type}")

    model = EchoCare_CLIP(
        image_encoder=img_enc,
        text_encoder=txt_enc,
        init_temperature=config["init_temperature"],
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.1f}%)")

    return model


def load_model(config, checkpoint_path, text_encoder_type="clip", device="cpu") -> EchoCare_CLIP:
    """
    Build the model and load trained weights. For evaluation, encoders are fully frozen.
    """
    # For evaluation, we freeze both to save memory (gradients aren't needed anyway)
    model = build_model(
        config,
        text_encoder_type=text_encoder_type,
        freeze_image=True,
        freeze_text=True,
        device=device
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # For .pth files that are saved as dict with 'model_state_dict', or raw state_dicts
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"Loaded trained weights from: {checkpoint_path}")

    return model
