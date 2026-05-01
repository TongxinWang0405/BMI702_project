"""
Discover datasets under prepared_data_root, load & merge their CSVs, resolve
image paths, hold out user-specified datasets for OOD evaluation, split the
rest into train/val/test stratified by dataset, compute train-set normalization
statistics, and persist everything to the data_split folder.

Usage:
    python prepare_data.py [--config config.json]

The normalization mean/std computed from the training split are written back
into config.json (section `normalization`) so that every downstream training
script reads the same values.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import DEFAULT_CONFIG_PATH, load_config, save_config


# ── Column-name detection heuristics ─────────────────────────────────────────
PATH_COL_CANDIDATES = [
    "file_name", "image_name", "path", "filename", "image_path", "filepath",
    "file", "image", "img_path", "img", "image_file",
]
CAPTION_COL_CANDIDATES = [
    "caption", "text_caption", "text_captions", "description", "text", "label",
    "report", "finding", "findings", "annotation", "sentence",
]
TISSUE_COL_CANDIDATES = [
    "tissue", "organ", "tissue_type", "body_part", "region", "anatomy",
    "tissue_name", "zone",
]
CONDITION_COL_CANDIDATES = [
    "condition", "diagnosis", "classification", "birads", "pathology",
    "label_condition", "malignancy", "interpretation", "class", "category",
]


def detect_column(df_columns, candidates):
    cols_lower = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def discover_datasets(root):
    """Return sorted (name, csv_path) pairs. Skips the data_splits folder."""
    SKIP_DIRS = {"data_splits", "data_split"}
    datasets = []
    for dirpath, dirnames, filenames in os.walk(root):
        dataset_name = os.path.basename(dirpath)
        if dataset_name in SKIP_DIRS:
            dirnames.clear()
            continue
        csv_files = [f for f in filenames if f.lower().endswith(".csv")]
        if csv_files:
            csv_full = [os.path.join(dirpath, f) for f in csv_files]
            csv_full.sort(key=os.path.getsize, reverse=True)
            datasets.append((dataset_name, csv_full[0]))
    return sorted(datasets)


def load_dataset_csv(csv_path, dataset_name):
    df = pd.read_csv(csv_path)
    path_col = detect_column(df.columns, PATH_COL_CANDIDATES)
    cap_col = detect_column(df.columns, CAPTION_COL_CANDIDATES)
    if path_col is None:
        raise ValueError(
            f"[{dataset_name}] Cannot detect image path column.\n"
            f"  Available: {list(df.columns)}\n"
            f"  Expected one of: {PATH_COL_CANDIDATES}"
        )
    if cap_col is None:
        raise ValueError(
            f"[{dataset_name}] Cannot detect caption column.\n"
            f"  Available: {list(df.columns)}\n"
            f"  Expected one of: {CAPTION_COL_CANDIDATES}"
        )
    out = pd.DataFrame({
        "image_path": df[path_col].astype(str),
        "caption":    df[cap_col].astype(str),
        "dataset":    dataset_name,
    })
    tissue_col = detect_column(df.columns, TISSUE_COL_CANDIDATES)
    cond_col = detect_column(df.columns, CONDITION_COL_CANDIDATES)
    if tissue_col is not None:
        out["tissue"] = df[tissue_col].astype(str)
        print(f"    → detected tissue column: {tissue_col!r}")
    if cond_col is not None:
        out["condition"] = df[cond_col].astype(str)
        print(f"    → detected condition column: {cond_col!r}")
    print(f"  [{dataset_name}] {len(out)} rows | "
          f"path_col={path_col!r} | caption_col={cap_col!r}")
    return out


def resolve_image_path(raw_path, csv_dir):
    basename = os.path.basename(raw_path)
    candidates = [
        raw_path,
        os.path.join(csv_dir, raw_path),
        os.path.join(csv_dir, basename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    try:
        for entry in os.scandir(csv_dir):
            if entry.is_dir():
                p = os.path.join(entry.path, basename)
                if os.path.isfile(p):
                    return p
                p = os.path.join(entry.path, raw_path)
                if os.path.isfile(p):
                    return p
    except OSError:
        pass
    return None


def safe_split(df, test_size, stratify_col, seed):
    try:
        return train_test_split(df, test_size=test_size,
                                stratify=df[stratify_col], random_state=seed)
    except ValueError as e:
        print(f"  [WARNING] Stratified split failed ({e}). Using random split.")
        return train_test_split(df, test_size=test_size, random_state=seed)


# ── Normalization stats ──────────────────────────────────────────────────────
class _RawTrainDataset(Dataset):
    def __init__(self, df, image_size):
        self.paths = df["image_path"].tolist()
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.tf(Image.open(self.paths[idx]).convert("RGB"))


def compute_normalization(train_df, image_size, num_workers):
    ds = _RawTrainDataset(train_df, image_size)
    loader = DataLoader(ds, batch_size=64, num_workers=num_workers,
                        pin_memory=False, shuffle=False)
    mean = torch.zeros(3)
    var = torch.zeros(3)
    n = 0
    print(f"Computing mean and std over {len(ds):,} train images...")
    for imgs in loader:
        B, C, H, W = imgs.shape
        pixels = imgs.permute(1, 0, 2, 3).reshape(C, -1)
        mean += pixels.mean(dim=1)
        var += pixels.var(dim=1)
        n += 1
    mean /= n
    var /= n
    std = var.sqrt()
    return mean.tolist(), std.tolist()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--skip-norm", action="store_true",
                        help="Skip computing train-set normalization statistics.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg["paths"]["prepared_data_root"]
    splits_dir = cfg["paths"]["data_split_dir"]
    os.makedirs(splits_dir, exist_ok=True)

    # ── Discover datasets ────────────────────────────────────────────────
    datasets_found = discover_datasets(root)
    print(f'Found {len(datasets_found)} dataset(s) under: {root}\n')
    for name, csv_path in datasets_found:
        print(f"  {name:40s}  ->  {csv_path}")
    if not datasets_found:
        print(f"[ERROR] No datasets found under {root}", file=sys.stderr)
        sys.exit(1)

    # ── Load & resolve ───────────────────────────────────────────────────
    all_dfs = []
    load_report = []
    for dataset_name, csv_path in datasets_found:
        csv_dir = os.path.dirname(csv_path)
        try:
            df = load_dataset_csv(csv_path, dataset_name)
        except Exception as e:
            print(f"  [CSV ERROR] {dataset_name}: {e}")
            load_report.append((dataset_name, 0, 0, "CSV_ERROR"))
            continue

        resolved, n_ok, n_fail = [], 0, 0
        for raw_path in df["image_path"]:
            p = resolve_image_path(raw_path, csv_dir)
            resolved.append(p)
            if p is not None: n_ok += 1
            else:             n_fail += 1
        df["resolved_path"] = resolved
        df_ok = df[df["resolved_path"].notna()].copy()
        df_ok["image_path"] = df_ok["resolved_path"]
        df_ok = df_ok.drop(columns=["resolved_path"])
        if len(df_ok) > 0:
            all_dfs.append(df_ok)
        status = "OK" if n_fail == 0 else f"{n_fail} missing"
        load_report.append((dataset_name, n_ok, n_fail, status))

    # ── Load summary table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f'{"Dataset":<38} {"Loaded":>8} {"Failed":>8}  Status')
    print("=" * 72)
    for name, n_ok, n_fail, status in load_report:
        ok_str = f"{n_ok:,}" if n_ok else "-"
        fail_str = f"{n_fail:,}" if n_fail else "-"
        print(f"{name:<38} {ok_str:>8} {fail_str:>8}  {status}")
    print("=" * 72)
    total_ok = sum(r[1] for r in load_report)
    total_fail = sum(r[2] for r in load_report)
    print(f'{"TOTAL":<38} {total_ok:>8,} {total_fail:>8,}')

    if not all_dfs:
        print("[ERROR] No data loaded successfully.", file=sys.stderr)
        sys.exit(1)

    full_df = pd.concat(all_dfs, ignore_index=True)[["image_path", "caption", "dataset"]]

    print(f"\nCombined dataset: {len(full_df):,} samples from "
          f"{full_df['dataset'].nunique()} source(s)")
    print("\nPer-dataset breakdown:")
    counts = full_df["dataset"].value_counts()
    for ds, n in counts.items():
        print(f"  {ds:<40} {n:>6,}  ({100*n/len(full_df):.1f}%)")

    # ── Hold-out datasets (for zero-shot / OOD eval) ─────────────────────
    held_out = list(cfg["split"].get("held_out_datasets", []))
    if held_out:
        print(f"\nHolding out {len(held_out)} dataset(s) from train/val/test:")
        for ds in held_out:
            n = (full_df["dataset"] == ds).sum()
            print(f"  - {ds}: {n:,} samples")
        heldout_df = full_df[full_df["dataset"].isin(held_out)].reset_index(drop=True)
        core_df = full_df[~full_df["dataset"].isin(held_out)].reset_index(drop=True)
    else:
        print("\nNo datasets held out (split.held_out_datasets is empty).")
        heldout_df = full_df.iloc[0:0].copy()
        core_df = full_df

    if len(core_df) == 0:
        print("[ERROR] All datasets held out — nothing to train on.", file=sys.stderr)
        sys.exit(1)

    # ── Train/val/test split stratified by dataset ───────────────────────
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    seed = cfg["split"]["random_seed"]
    val_frac = val_size / (1.0 - test_size)

    train_val_df, test_df = safe_split(core_df, test_size, "dataset", seed)
    train_df, val_df = safe_split(train_val_df, val_frac, "dataset", seed)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\nSplit sizes  (core = {len(core_df):,}, held-out = {len(heldout_df):,})")
    print(f"  Train   : {len(train_df):>6,}  ({100*len(train_df)/len(core_df):.1f}% of core)")
    print(f"  Val     : {len(val_df):>6,}  ({100*len(val_df)/len(core_df):.1f}% of core)")
    print(f"  Test    : {len(test_df):>6,}  ({100*len(test_df)/len(core_df):.1f}% of core)")
    if len(heldout_df):
        print(f"  HeldOut : {len(heldout_df):>6,}")

    # ── Compute normalization stats from training images ────────────────
    if not args.skip_norm:
        mean, std = compute_normalization(
            train_df,
            image_size=cfg["image_encoder"]["image_size"],
            num_workers=cfg["training"]["num_workers"],
        )
        cfg["normalization"]["img_mean"] = [round(v, 6) for v in mean]
        cfg["normalization"]["img_std"]  = [round(v, 6) for v in std]
        print(f"  mean : {[round(v, 4) for v in cfg['normalization']['img_mean']]}")
        print(f"  std  : {[round(v, 4) for v in cfg['normalization']['img_std']]}")
        save_config(cfg, args.config)
        print(f"Updated normalization written to {args.config}")
    else:
        print("Skipped normalization-statistics computation (--skip-norm).")

    # ── Persist splits ───────────────────────────────────────────────────
    for name, df in [("train", train_df), ("val", val_df),
                     ("test", test_df), ("full", core_df),
                     ("held_out", heldout_df)]:
        if len(df) == 0 and name == "held_out":
            continue
        csv_path = os.path.join(splits_dir, f"{name}_df.csv")
        parquet_path = os.path.join(splits_dir, f"{name}_df.parquet")
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        print(f"  {name:8s}: {len(df):>6,} rows → {csv_path}")

    split_metadata = {
        "config": cfg,
        "datasets_found": datasets_found,
        "held_out_datasets": held_out,
        "full_df_shape": list(full_df.shape),
        "per_dataset_counts": full_df["dataset"].value_counts().to_dict(),
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "held_out": len(heldout_df),
        },
        "library_versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }
    meta_path = os.path.join(splits_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(split_metadata, f, indent=2, default=str)
    print(f"\nSaved split metadata → {meta_path}")
    print(f"All splits saved to: {splits_dir}")


if __name__ == "__main__":
    main()
