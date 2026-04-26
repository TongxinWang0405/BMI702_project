"""
End-to-end few-shot adaptation using organ-specific label prompts (Notebook Cells 56-62).

Instead of a separate linear head, we fine-tune the entire EchoCare-CLIP model using
cross-entropy loss over CLIP similarity scores between image embeddings and class text
prompt embeddings — the same approach the result notebook uses for trained models.

Public API
----------
evaluate_fewshot_e2e(model, train_df, val_df, test_df, cfg, device)
    → (per_organ_raw_df, aggregate_df)
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import LabeledUltrasoundDataset
from ..data.loader import infer_organ_from_dataset
from ..models.utils import tokenize_texts
from .classification import build_label_prompt_for_organ
from .linear_probe import sample_fewshot_per_class, multiclass_metrics, summarize_across_organs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_loader(df: pd.DataFrame, cfg: dict, shuffle: bool = False) -> DataLoader:
    ds = LabeledUltrasoundDataset(df.reset_index(drop=True), split="train" if shuffle else "test", cfg=cfg)
    return DataLoader(
        ds,
        batch_size=cfg.get("eval", {}).get("fewshot_e2e_batch_size", 16),
        shuffle=shuffle,
        drop_last=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )


@torch.no_grad()
def _evaluate_prompt_classifier(model, df: pd.DataFrame, class_names: list,
                                  organ: str, tokenizer, cfg: dict,
                                  device: str) -> dict:
    """Zero-shot classification using organ-specific label prompts."""
    max_seq = cfg["max_seq_len"]
    batch_size = cfg.get("eval", {}).get("fewshot_e2e_batch_size", 16)

    prompts = [build_label_prompt_for_organ(c, organ) for c in class_names]
    ids, mask = tokenize_texts(prompts, tokenizer, max_length=max_seq, device=device)
    text_embs = model.encode_text(ids, mask)                 # (C, D)
    scale = torch.clamp(1.0 / model.log_temperature.exp(), max=100.0)

    y_true, y_pred_list, all_probs = [], [], []

    loader = _make_loader(df, cfg, shuffle=False)
    for batch in loader:
        images = batch[0].to(device)
        raw_labels = list(batch[2])

        img_embs = model.encode_image(images)                # (B, D)
        logits = scale * (img_embs @ text_embs.T)            # (B, C)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        y_true.extend(raw_labels)
        y_pred_list.extend([class_names[i] for i in preds])
        all_probs.append(probs)

    y_prob = np.concatenate(all_probs, axis=0)
    metrics = multiclass_metrics(y_true, y_pred_list, y_prob, class_names)
    metrics["n_eval"] = len(y_true)
    return metrics


def _train_one_organ(model, support_df: pd.DataFrame, val_df: pd.DataFrame,
                      class_names: list, organ: str, tokenizer,
                      cfg: dict, device: str, seed: int):
    """
    Fine-tune model on support_df using prompt-based cross-entropy.
    Returns (best_model_state, training_history_df, best_val_f1).
    """
    eval_cfg   = cfg.get("eval", {})
    n_epochs   = eval_cfg.get("fewshot_e2e_epochs", 8)
    lr         = float(eval_cfg.get("fewshot_e2e_lr", {}).get("trained", 1e-5))
    wd         = float(eval_cfg.get("fewshot_e2e_weight_decay", 1e-4))
    grad_clip  = float(eval_cfg.get("fewshot_e2e_grad_clip", 1.0))
    batch_size = eval_cfg.get("fewshot_e2e_batch_size", 16)
    max_seq    = cfg["max_seq_len"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    best_state  = copy.deepcopy(model.state_dict())
    best_val_f1 = -np.inf
    history     = []

    # Minibatch iterator (reshuffled each epoch)
    def iter_batches(df, bs, epoch_seed):
        idx = np.arange(len(df))
        np.random.default_rng(epoch_seed).shuffle(idx)
        for s in range(0, len(idx), bs):
            yield df.iloc[idx[s:s + bs]].reset_index(drop=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch_df in iter_batches(support_df, batch_size, seed + epoch):
            # Build image batch via LabeledUltrasoundDataset transform
            tmp_ds = LabeledUltrasoundDataset(batch_df, split="train", cfg=cfg)
            images = torch.stack([tmp_ds[i][0] for i in range(len(tmp_ds))]).to(device)
            raw_labels = batch_df["label"].astype(str).str.lower().tolist()
            targets = torch.tensor(
                [label_to_idx[y] for y in raw_labels], dtype=torch.long, device=device
            )

            prompts = [build_label_prompt_for_organ(c, organ) for c in class_names]
            ids, attn = tokenize_texts(prompts, tokenizer, max_length=max_seq, device=device)

            img_embs  = model.encode_image(images)
            text_embs = model.encode_text(ids, attn)
            scale     = torch.clamp(1.0 / model.log_temperature.exp(), max=100.0)
            logits    = scale * (img_embs @ text_embs.T)
            loss      = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validate
        model.eval()
        val_metrics = _evaluate_prompt_classifier(
            model, val_df, class_names, organ, tokenizer, cfg, device
        )
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({
            "epoch":         epoch + 1,
            "train_loss":    mean_loss,
            "val_auroc":     val_metrics["auroc"],
            "val_accuracy":  val_metrics["accuracy"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        })

        if val_metrics["f1_weighted"] > best_val_f1:
            best_val_f1 = val_metrics["f1_weighted"]
            best_state  = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history), best_val_f1


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate_fewshot_e2e(model, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          test_df: pd.DataFrame, cfg: dict,
                          device: str = "cpu", tokenizer=None):
    """
    End-to-end few-shot fine-tuning per organ × frac × seed.

    Parameters
    ----------
    model     : trained EchoCare_CLIP (deep-copied for each run)
    train_df  : labeled train DataFrame (output of prepare_labeled_eval_df)
    val_df    : labeled val DataFrame
    test_df   : labeled test DataFrame
    cfg       : config dict
    device    : torch device string
    tokenizer : CLIP/BERT tokenizer (if None, skipped with warning)

    Returns
    -------
    per_organ_raw_df  : one row per (model, organ, frac, seed)
    aggregate_df      : macro + weighted averages per (frac, seed), then summarised
    """
    if tokenizer is None:
        print("[fewshot_e2e] No tokenizer provided — skipping.")
        return pd.DataFrame(), pd.DataFrame()

    eval_cfg  = cfg.get("eval", {})
    fracs     = eval_cfg.get("fewshot_e2e_fracs", [0.01, 0.05, 0.10])
    num_seeds = eval_cfg.get("fewshot_e2e_num_seeds", 3)
    base_seed = cfg.get("random_seed", 42)

    organs = sorted(
        set(train_df["organ"].unique())
        & set(val_df["organ"].unique())
        & set(test_df["organ"].unique())
    )

    raw_rows  = []
    agg_rows  = []

    for frac in fracs:
        for rep in range(num_seeds):
            seed = base_seed + rep

            per_organ_this_run = []

            for organ in organs:
                tr = train_df[train_df["organ"] == organ].copy()
                va = val_df[val_df["organ"] == organ].copy()
                te = test_df[test_df["organ"] == organ].copy()

                # keep classes with ≥2 train examples
                keep = tr["label"].value_counts()
                keep = keep[keep >= 2].index.tolist()
                tr = tr[tr["label"].isin(keep)]
                va = va[va["label"].isin(keep)]
                te = te[te["label"].isin(keep)]

                if tr["label"].nunique() < 2 or len(te) == 0 or len(va) == 0:
                    continue

                support_df = sample_fewshot_per_class(tr, frac=frac, seed=seed)
                class_names = sorted(support_df["label"].astype(str).str.lower().unique())

                va = va[va["label"].isin(class_names)]
                te = te[te["label"].isin(class_names)]
                if len(va) == 0 or len(te) == 0 or len(class_names) < 2:
                    continue

                print(
                    f"  [e2e] frac={frac:.3f} seed={seed} organ={organ}"
                    f"  n_support={len(support_df)}  n_test={len(te)}"
                )

                ft_model = copy.deepcopy(model).to(device)
                ft_model, _, _ = _train_one_organ(
                    ft_model, support_df, va, class_names, organ,
                    tokenizer, cfg, device, seed,
                )

                test_metrics = _evaluate_prompt_classifier(
                    ft_model, te, class_names, organ, tokenizer, cfg, device
                )
                test_metrics.update({
                    "organ":       organ,
                    "fewshot_frac": frac,
                    "seed":        seed,
                    "n_train":     len(support_df),
                })
                raw_rows.append(test_metrics)
                per_organ_this_run.append(test_metrics)

                del ft_model

            if per_organ_this_run:
                agg = summarize_across_organs(pd.DataFrame(per_organ_this_run))
                agg["fewshot_frac"] = frac
                agg["seed"] = seed
                agg_rows.append(agg)

    if not raw_rows:
        print("[fewshot_e2e] No results produced.")
        return pd.DataFrame(), pd.DataFrame()

    per_organ_raw = pd.DataFrame(raw_rows)

    # Aggregate over seeds
    agg_raw = pd.concat(agg_rows, ignore_index=True)
    aggregate_summary = (
        agg_raw
        .groupby(["average_type", "fewshot_frac"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
        )
    )

    return per_organ_raw, aggregate_summary
