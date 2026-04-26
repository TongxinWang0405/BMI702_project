"""
Evaluate representation quality using linear probing and few-shot linear probing.

Public API
----------
evaluate_linear_probe(model, train_df, val_df, test_df, cfg, device)
    → (per_organ_df, aggregate_df)

evaluate_fewshot_probe(model, train_df, val_df, test_df, cfg, device)
    → (per_organ_summary_df, aggregate_summary_df)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..data.dataset import LabeledUltrasoundDataset


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings_from_df(model, df: pd.DataFrame, cfg: dict,
                                device: str = "cpu") -> np.ndarray:
    """
    Build a LabeledUltrasoundDataset from df and extract image embeddings.
    Returns a numpy array of shape (len(df), projection_dim) in df row order.
    """
    dataset = LabeledUltrasoundDataset(df.reset_index(drop=True), split="test", cfg=cfg)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False,
        num_workers=cfg.get("num_workers", 2), pin_memory=True,
    )
    model.eval()
    chunks = []
    for batch in loader:
        images = batch[0].to(device)
        embs = model.encode_image(images).cpu().numpy()
        chunks.append(embs)
    return np.concatenate(chunks, axis=0)


# ── Sampling ───────────────────────────────────────────────────────────────────

def sample_fewshot_per_class(df: pd.DataFrame, frac: float, seed: int,
                              label_col: str = "label") -> pd.DataFrame:
    """
    Per-class stratified sampling: at least 2 samples per class when possible.
    """
    pieces = []
    for label, group in df.groupby(label_col):
        n = int(np.ceil(len(group) * frac))
        n = max(2, n) if len(group) >= 2 else 1
        n = min(n, len(group))
        pieces.append(group.sample(n=n, random_state=seed))
    return (
        pd.concat(pieces, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def multiclass_metrics(y_true, y_pred, y_prob, class_names) -> dict:
    """Accuracy, F1-weighted, and macro-OvR AUROC (handles binary and multiclass)."""
    out = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        if len(class_names) == 2:
            pos_idx = 1
            y_bin = (np.asarray(y_true) == class_names[pos_idx]).astype(int)
            out["auroc"] = roc_auc_score(y_bin, y_prob[:, pos_idx])
        else:
            y_bin = label_binarize(y_true, classes=class_names)
            out["auroc"] = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auroc"] = float("nan")
    return out


def summarize_across_organs(per_organ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-organ metrics into macro and weighted-by-n_eval averages.
    Returns DataFrame with columns: average_type, auroc, accuracy, f1_weighted, n_organs.
    """
    if per_organ_df.empty:
        return pd.DataFrame(columns=["average_type", "auroc", "accuracy", "f1_weighted", "n_organs"])

    n_col = "n_eval" if "n_eval" in per_organ_df.columns else "n_test"
    weights = per_organ_df[n_col].to_numpy(dtype=float)
    rows = []

    for avg_name in ("macro", "weighted_by_test_n"):
        row = {"average_type": avg_name, "n_organs": int(len(per_organ_df))}
        for metric in ("auroc", "accuracy", "f1_weighted"):
            vals = per_organ_df[metric].to_numpy(dtype=float)
            mask = ~np.isnan(vals)
            if mask.sum() == 0:
                row[metric] = float("nan")
            elif avg_name == "macro":
                row[metric] = float(vals[mask].mean())
            else:
                row[metric] = float(np.average(vals[mask], weights=weights[mask]))
        rows.append(row)

    return pd.DataFrame(rows)


# ── Probe fitting ──────────────────────────────────────────────────────────────

def fit_best_linear_probe(X_tr, y_tr, X_va=None, y_va=None,
                           c_grid=None, max_iter=3000, seed=42):
    """
    Fit logistic regression, selecting C by validation F1-weighted.
    Returns (pipeline, label_encoder, best_c, best_val_f1).
    """
    c_grid = c_grid or [0.01, 0.1, 1.0, 10.0]
    le = LabelEncoder()
    le.fit(y_tr)
    y_tr_enc = le.transform(y_tr)

    best_model, best_score, best_c = None, -np.inf, None

    for c in c_grid:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=c, max_iter=max_iter, class_weight="balanced",
                solver="lbfgs", random_state=seed,
            )),
        ])
        clf.fit(X_tr, y_tr_enc)

        if X_va is not None and y_va is not None and len(X_va) > 0:
            va_mask = np.isin(y_va, le.classes_)
            if va_mask.sum() >= 2:
                y_va_enc = le.transform(y_va[va_mask])
                score = f1_score(y_va_enc, clf.predict(X_va[va_mask]),
                                 average="weighted", zero_division=0)
            else:
                score = f1_score(y_tr_enc, clf.predict(X_tr),
                                 average="weighted", zero_division=0)
        else:
            score = f1_score(y_tr_enc, clf.predict(X_tr),
                             average="weighted", zero_division=0)

        if score > best_score:
            best_score, best_model, best_c = score, clf, c

    return best_model, le, best_c, best_score


# ── Per-organ probe runner ─────────────────────────────────────────────────────

def run_per_organ_probe(train_meta, train_X, val_meta, val_X,
                         test_meta, test_X,
                         fewshot_frac=None, fewshot_seed=42,
                         c_grid=None, max_iter=3000) -> pd.DataFrame:
    """
    For each organ: optionally subsample train, fit probe, evaluate on test.
    Returns DataFrame with per-organ metrics.
    """
    rows = []

    for organ in test_meta["organ"].unique():
        tr = train_meta[train_meta["organ"] == organ].copy()
        va = val_meta[val_meta["organ"] == organ].copy()
        te = test_meta[test_meta["organ"] == organ].copy()

        # keep classes with ≥2 train examples
        keep = tr["label"].value_counts()
        keep = keep[keep >= 2].index.tolist()
        tr = tr[tr["label"].isin(keep)]
        va = va[va["label"].isin(keep)]
        te = te[te["label"].isin(keep)]

        if tr["label"].nunique() < 2 or len(te) == 0:
            continue

        if fewshot_frac is not None:
            tr = sample_fewshot_per_class(tr, fewshot_frac, fewshot_seed)

        X_tr = train_X[tr["row_id"].to_numpy()]
        y_tr = tr["label"].to_numpy()
        X_va = val_X[va["row_id"].to_numpy()]
        y_va = va["label"].to_numpy()
        X_te = test_X[te["row_id"].to_numpy()]
        y_te = te["label"].to_numpy()

        clf, le, best_c, _ = fit_best_linear_probe(
            X_tr, y_tr, X_va, y_va, c_grid=c_grid, max_iter=max_iter, seed=fewshot_seed,
        )

        keep_te = np.isin(y_te, le.classes_)
        X_te, y_te = X_te[keep_te], y_te[keep_te]
        if len(y_te) == 0:
            continue

        y_pred_enc = clf.predict(X_te)
        y_pred = le.inverse_transform(y_pred_enc)
        y_prob = clf.predict_proba(X_te)
        class_names = list(le.classes_)

        metrics = multiclass_metrics(y_te, y_pred, y_prob, class_names)
        metrics.update({
            "organ":    organ,
            "n_train":  int(len(tr)),
            "n_eval":   int(len(y_te)),
            "n_classes": int(len(class_names)),
            "best_c":   best_c,
        })
        rows.append(metrics)

    return pd.DataFrame(rows)


# ── Public entry points ────────────────────────────────────────────────────────

def evaluate_linear_probe(model, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame, cfg: dict,
                           device: str = "cpu"):
    """
    Full-data linear probe (per organ).
    Returns (per_organ_df, aggregate_df).
    """
    c_grid   = cfg.get("eval", {}).get("linear_probe_c_grid", [0.01, 0.1, 1.0, 10.0])
    max_iter = cfg.get("eval", {}).get("probe_max_iter", 3000)

    print("  Extracting embeddings (train)…")
    X_train = extract_embeddings_from_df(model, train_df, cfg, device)
    print("  Extracting embeddings (val)…")
    X_val   = extract_embeddings_from_df(model, val_df, cfg, device)
    print("  Extracting embeddings (test)…")
    X_test  = extract_embeddings_from_df(model, test_df, cfg, device)

    per_organ = run_per_organ_probe(
        train_df, X_train, val_df, X_val, test_df, X_test,
        fewshot_frac=None, c_grid=c_grid, max_iter=max_iter,
    )
    aggregate = summarize_across_organs(per_organ)
    return per_organ, aggregate


def evaluate_fewshot_probe(model, train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame, cfg: dict,
                            device: str = "cpu"):
    """
    Few-shot linear probe: per organ, multiple fractions × seeds.
    Returns (per_organ_summary_df, aggregate_summary_df).
    """
    eval_cfg  = cfg.get("eval", {})
    fracs     = eval_cfg.get("fewshot_fracs", [0.01, 0.05, 0.10])
    num_seeds = eval_cfg.get("fewshot_num_seeds", 5)
    c_grid    = eval_cfg.get("linear_probe_c_grid", [0.01, 0.1, 1.0, 10.0])
    max_iter  = eval_cfg.get("probe_max_iter", 3000)

    print("  Extracting embeddings (train)…")
    X_train = extract_embeddings_from_df(model, train_df, cfg, device)
    print("  Extracting embeddings (val)…")
    X_val   = extract_embeddings_from_df(model, val_df, cfg, device)
    print("  Extracting embeddings (test)…")
    X_test  = extract_embeddings_from_df(model, test_df, cfg, device)

    raw_rows = []
    agg_rows = []

    for frac in fracs:
        for seed in range(num_seeds):
            actual_seed = cfg.get("random_seed", 42) + seed
            per_organ = run_per_organ_probe(
                train_df, X_train, val_df, X_val, test_df, X_test,
                fewshot_frac=frac, fewshot_seed=actual_seed,
                c_grid=c_grid, max_iter=max_iter,
            )
            per_organ["fewshot_frac"] = frac
            per_organ["seed"] = actual_seed
            raw_rows.append(per_organ)

            agg = summarize_across_organs(per_organ)
            agg["fewshot_frac"] = frac
            agg["seed"] = actual_seed
            agg_rows.append(agg)

    if not raw_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_organ_raw = pd.concat(raw_rows, ignore_index=True)
    agg_raw = pd.concat(agg_rows, ignore_index=True)

    # Summarise over seeds
    per_organ_summary = (
        per_organ_raw
        .groupby(["organ", "fewshot_frac"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
            n_eval_mean=("n_eval", "mean"),
        )
    )

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

    return per_organ_summary, aggregate_summary
