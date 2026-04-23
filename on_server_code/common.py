"""
Shared code for EchoCare-CLIP training experiments.

Imported by train_exp1.py ... train_exp4.py and prepare_data.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from monai.networks.nets.swin_unetr import SwinTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, BertModel, CLIPTextModel, CLIPTokenizer


# ─── Config ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.json"


def load_config(path: str | os.PathLike = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    # Resolve paths relative to the config file's directory for robustness.
    base = Path(path).resolve().parent
    for k, v in cfg["paths"].items():
        p = Path(v)
        if not p.is_absolute():
            cfg["paths"][k] = str((base / p).resolve())
    return cfg


def save_config(cfg: dict, path: str | os.PathLike = DEFAULT_CONFIG_PATH) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ─────────────────────────────────────────────────────────────────
class MultiDatasetUltrasound(Dataset):
    """Multi-source ultrasound dataset with free-text captions."""

    def __init__(self, dataframe: pd.DataFrame, split: str, cfg: dict):
        self.df = dataframe.reset_index(drop=True)
        self.split = split

        sz = cfg["image_encoder"]["image_size"]
        mean = cfg["normalization"]["img_mean"]
        std = cfg["normalization"]["img_std"]

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(int(sz * 1.125)),
                transforms.RandomCrop(sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(sz),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, row["caption"], row["dataset"]


def build_dataloaders(cfg: dict, splits_dir: str | None = None):
    """Load train/val/test parquets and return DataLoaders."""
    splits_dir = splits_dir or cfg["paths"]["data_split_dir"]
    train_df = pd.read_parquet(os.path.join(splits_dir, "train_df.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "val_df.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test_df.parquet"))

    train_ds = MultiDatasetUltrasound(train_df, "train", cfg)
    val_ds = MultiDatasetUltrasound(val_df, "val", cfg)
    test_ds = MultiDatasetUltrasound(test_df, "test", cfg)

    kw = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kw)
    return train_loader, val_loader, test_loader


# ─── Models ──────────────────────────────────────────────────────────────────
class EchoCareImageEncoder(nn.Module):
    def __init__(self, feature_size=128, in_channels=3, projection_dim=256,
                 use_checkpoint=True, freeze_encoder=False):
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
        encoder_out_dim = feature_size * (2 ** 4)  # 2048
        self.projection = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.ReLU(),
            nn.Linear(encoder_out_dim, projection_dim),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def load_pretrained(self, checkpoint_path: str, device: str = "cpu"):
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict.pop("mask_token", None)
        self.encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded EchoCare pretrained weights from '{checkpoint_path}'")

    def forward(self, x):
        feats = self.encoder(x)
        deep = feats[-1]
        emb = deep.mean(dim=(2, 3))
        emb = self.projection(emb)
        return F.normalize(emb, dim=-1)


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32",
                 projection_dim=256, freeze_text_encoder=True):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        text_out_dim = self.text_encoder.config.hidden_size
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(text_out_dim, text_out_dim),
            nn.ReLU(),
            nn.Linear(text_out_dim, projection_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.projection(out.pooler_output), dim=-1)


class BertTextEncoder(nn.Module):
    """BERT-family text tower (Bio_ClinicalBERT, PubMedBERT, etc.) + MLP head.

    Uses the [CLS] pooler output. Shape-compatible with CLIPTextEncoder.
    """
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT",
                 projection_dim=256, freeze_text_encoder=True):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(model_name)
        text_out_dim = self.text_encoder.config.hidden_size
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(text_out_dim, text_out_dim),
            nn.ReLU(),
            nn.Linear(text_out_dim, projection_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.projection(out.pooler_output), dim=-1)


def build_text_encoder(text_cfg: dict, projection_dim: int, freeze: bool) -> nn.Module:
    t = text_cfg["type"].lower()
    if t == "clip":
        return CLIPTextEncoder(
            model_name=text_cfg["model_name"],
            projection_dim=projection_dim,
            freeze_text_encoder=freeze,
        )
    if t == "bert":
        return BertTextEncoder(
            model_name=text_cfg["model_name"],
            projection_dim=projection_dim,
            freeze_text_encoder=freeze,
        )
    raise ValueError(f"Unknown text encoder type: {text_cfg['type']}")


class EchoCare_CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, init_temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, images, input_ids, attention_mask=None):
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        temperature = self.log_temperature.exp()
        logits_per_image = (image_emb @ text_emb.T) / temperature
        logits_per_text = logits_per_image.T
        labels = torch.arange(image_emb.size(0), device=image_emb.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2, image_emb, text_emb


def build_model(cfg: dict, freeze_image: bool, freeze_text: bool,
                device: torch.device, text_encoder_key: str = "clip"):
    img_enc = EchoCareImageEncoder(
        feature_size=cfg["image_encoder"]["feature_size"],
        in_channels=cfg["image_encoder"]["in_channels"],
        projection_dim=cfg["projection_dim"],
        freeze_encoder=freeze_image,
    )
    ckpt = cfg["paths"]["echocare_checkpoint"]
    if os.path.exists(ckpt):
        img_enc.load_pretrained(ckpt, device=str(device))
    else:
        print(f"[WARNING] EchoCare checkpoint not found at {ckpt} — using random init.")

    txt_enc = build_text_encoder(
        text_cfg=cfg["text_encoders"][text_encoder_key],
        projection_dim=cfg["projection_dim"],
        freeze=freeze_text,
    )

    model = EchoCare_CLIP(
        image_encoder=img_enc,
        text_encoder=txt_enc,
        init_temperature=cfg["training"]["init_temperature"],
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.1f}%)")
    return model


# ─── Tokenization ────────────────────────────────────────────────────────────
def get_tokenizer(cfg: dict, text_encoder_key: str = "clip"):
    """Return a HF tokenizer with `._max_length` attached for downstream use."""
    tcfg = cfg["text_encoders"][text_encoder_key]
    t = tcfg["type"].lower()
    if t == "clip":
        tok = CLIPTokenizer.from_pretrained(tcfg["model_name"])
    elif t == "bert":
        tok = AutoTokenizer.from_pretrained(tcfg["model_name"])
    else:
        raise ValueError(f"Unknown text encoder type: {tcfg['type']}")
    tok._max_length = tcfg["max_seq_len"]
    return tok


def tokenize_texts(texts, tokenizer, device="cpu", max_length=None):
    if max_length is None:
        max_length = getattr(tokenizer, "_max_length", 77)
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


# ─── Training loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, tokenizer, device, epoch):
    model.train()
    total = 0.0
    for batch_idx, (images, texts, _) in enumerate(dataloader):
        images = images.to(device)
        ids, mask = tokenize_texts(texts, tokenizer, device=device)
        optimizer.zero_grad()
        loss, _, _ = model(images, ids, mask)
        loss.backward()
        optimizer.step()
        total += loss.item()
        if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
            print(f"  [E{epoch+1}] Batch {batch_idx+1}/{len(dataloader)}"
                  f" | loss={loss.item():.4f}"
                  f" | temp={model.log_temperature.exp().item():.4f}",
                  flush=True)
    return total / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, tokenizer, device):
    model.eval()
    total = 0.0
    for images, texts, _ in dataloader:
        images = images.to(device)
        ids, mask = tokenize_texts(texts, tokenizer, device=device)
        loss, _, _ = model(images, ids, mask)
        total += loss.item()
    return total / len(dataloader)


@torch.no_grad()
def compute_alignment_score(model, dataloader, tokenizer, device):
    model.eval()
    all_img, all_txt = [], []
    for images, texts, _ in dataloader:
        images = images.to(device)
        ids, mask = tokenize_texts(texts, tokenizer, device=device)
        all_img.append(model.encode_image(images).cpu())
        all_txt.append(model.encode_text(ids, mask).cpu())
    img_embs = torch.cat(all_img)
    txt_embs = torch.cat(all_txt)
    paired_sim = (img_embs * txt_embs).sum(dim=1).mean().item()
    all_sim = img_embs @ txt_embs.T
    off_mask = ~torch.eye(img_embs.size(0), dtype=torch.bool)
    cross_sim = all_sim[off_mask].mean().item()
    print(f"  Alignment score (paired):   {paired_sim:.4f}")
    print(f"  Cross-pair mean similarity: {cross_sim:.4f}")
    return paired_sim


def train_experiment(cfg, model, train_loader, val_loader, tokenizer, device,
                     best_ckpt_path: str, label: str = ""):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"]
    )
    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f'\n{"="*62}\n  Training: {label}\n{"="*62}', flush=True)

    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
    for epoch in range(cfg["training"]["num_epochs"]):
        t_loss = train_one_epoch(model, train_loader, optimizer, tokenizer, device, epoch)
        v_loss = validate_one_epoch(model, val_loader, tokenizer, device)
        scheduler.step()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        lr_now = scheduler.get_last_lr()[0]
        print(f">>> Epoch {epoch+1:2d}/{cfg['training']['num_epochs']}"
              f"  Train={t_loss:.4f}  Val={v_loss:.4f}  LR={lr_now:.2e}",
              flush=True)
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  [Checkpoint] best val={best_val:.4f} → {best_ckpt_path}",
                  flush=True)

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val:.4f}", flush=True)
    return train_losses, val_losses


# ─── Experiment runner shared by train_expN.py ──────────────────────────────
def run_experiment(exp_key: str, config_path: str | os.PathLike = DEFAULT_CONFIG_PATH):
    cfg = load_config(config_path)
    exp = cfg["experiments"][exp_key]
    text_key = exp.get("text_encoder_key", "clip")
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(f"Text encoder: {cfg['text_encoders'][text_key]['model_name']} "
          f"(key={text_key}, max_seq_len={cfg['text_encoders'][text_key]['max_seq_len']})",
          flush=True)

    tokenizer = get_tokenizer(cfg, text_encoder_key=text_key)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"DataLoaders ready:"
          f"  Train={len(train_loader)} batches,"
          f"  Val={len(val_loader)} batches,"
          f"  Test={len(test_loader)} batches", flush=True)

    model = build_model(cfg, freeze_image=exp["freeze_image"],
                        freeze_text=exp["freeze_text"], device=device,
                        text_encoder_key=text_key)

    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, exp["best_ckpt_name"])
    final_ckpt = os.path.join(ckpt_dir, exp["checkpoint_name"])

    train_losses, val_losses = train_experiment(
        cfg, model, train_loader, val_loader, tokenizer, device,
        best_ckpt_path=best_ckpt, label=exp["label"],
    )

    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model → {final_ckpt}", flush=True)

    print(f"\nTest-set alignment score for {exp['label']}:", flush=True)
    score = compute_alignment_score(model, test_loader, tokenizer, device)

    # Dump a per-experiment run summary next to the checkpoint.
    summary = {
        "experiment": exp_key,
        "label": exp["label"],
        "text_encoder_key": text_key,
        "text_encoder_model": cfg["text_encoders"][text_key]["model_name"],
        "freeze_image": exp["freeze_image"],
        "freeze_text": exp["freeze_text"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_alignment_score": score,
        "best_ckpt": best_ckpt,
        "final_ckpt": final_ckpt,
    }
    summary_path = os.path.join(ckpt_dir, f"{exp_key}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary → {summary_path}", flush=True)
