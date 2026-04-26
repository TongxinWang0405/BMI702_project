"""
Evaluate zero-shot classification using text prompts.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from ..models.utils import tokenize_texts


_ORGAN_DISPLAY = {
    "breast":  "breast",
    "lung":    "lung",
    "liver":   "liver",
    "thyroid": "thyroid",
    "carotid": "carotid artery",
}


def build_label_prompt_for_organ(label: str, organ: str) -> str:
    """Uniform organ-specific zero-shot prompt (matches result notebook)."""
    organ_text = _ORGAN_DISPLAY.get(str(organ).strip().lower(), str(organ).strip().lower())
    label_str  = str(label).strip().lower()
    return f"An ultrasound image of {organ_text} with {label_str} findings."


@torch.no_grad()
def evaluate_zero_shot_classification(model, dataloader, tokenizer, max_seq_len=77, device="cpu"):
    """
    Perform zero-shot classification by comparing image embeddings to
    prompted label embeddings for each sub-dataset/organ.
    """
    model.eval()
    
    # First, collect all features and metadata
    all_img_embs = []
    all_labels = []
    all_organs = []
    
    for batch in dataloader:
        images = batch[0].to(device)
        labels = batch[2]
        organs = [str(ds).lower() for ds in batch[3]]  # using dataset name as proxy for organ
        
        img_embs = model.encode_image(images).cpu()
        all_img_embs.append(img_embs)
        all_labels.extend(list(labels))
        all_organs.extend(organs)
        
    all_img_embs = torch.cat(all_img_embs)
    df = pd.DataFrame({
        "label": all_labels,
        "organ": [ds if "breast" in ds else "lung" if "lung" in ds else "liver" if "liver" in ds or "auli" in ds else "thyroid" if "thyroid" in ds or "auitd" in ds else "carotid" if "carotid" in ds else ds for ds in all_organs],
        "emb_idx": range(len(all_labels))
    })
    
    # We want to skip 'unknown' labels
    df = df[df["label"] != "unknown"]
    
    y_true_all, y_pred_all = [], []
    
    # Process organ by organ
    for organ, group_df in df.groupby("organ"):
        unique_labels = sorted(group_df["label"].unique())
        if len(unique_labels) < 2:
            continue
            
        prompts = [build_label_prompt_for_organ(lbl, organ) for lbl in unique_labels]
        ids, mask = tokenize_texts(prompts, tokenizer, max_length=max_seq_len, device=device)
        text_embs = model.encode_text(ids, mask).cpu()  # (NumClasses, D)
        
        img_embs = all_img_embs[group_df["emb_idx"].values]  # (N, D)
        
        # Logit scale
        scale = 1.0 / model.log_temperature.exp().item()
        logits = scale * (img_embs @ text_embs.T)  # (N, NumClasses)
        preds_idx = logits.argmax(dim=1).numpy()
        
        y_true_all.extend(group_df["label"].tolist())
        y_pred_all.extend([unique_labels[idx] for idx in preds_idx])
        
    if len(y_true_all) == 0:
        return 0.0
        
    return accuracy_score(y_true_all, y_pred_all)
