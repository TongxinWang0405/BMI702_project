"""
Data loading utilities: discover datasets, load CSVs, build DataLoaders.
"""

import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import MultiDatasetUltrasound, LabeledUltrasoundDataset


# ── Known column name variants (case-insensitive) ──────────────────────────
PATH_COL_CANDIDATES = [
    "file_name", "image_name", "path", "filename", "image_path",
    "filepath", "file", "image", "img_path", "img", "image_file",
]
CAPTION_COL_CANDIDATES = [
    "caption", "text_caption", "text_captions", "description", "text",
    "report", "finding", "findings", "annotation", "sentence",
]
LABEL_COL_CANDIDATES = [
    "label", "class", "condition", "diagnosis", "classification",
]


def detect_column(df_columns, candidates):
    """Return the first matching column name (case-insensitive), or None."""
    cols_lower = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def discover_datasets(root):
    """
    Walk the prepared_data root and return all (dataset_name, csv_path) pairs.
    A dataset folder is any directory that directly contains at least one .csv.
    If a folder has multiple CSVs, the largest one is used.
    """
    SKIP_DIRS = {"data_splits"}
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
    """
    Load a dataset CSV, auto-detecting path/caption/label columns.
    Returns a DataFrame with unified columns: image_path, caption, dataset, label.
    """
    df = pd.read_csv(csv_path)

    path_col = detect_column(df.columns, PATH_COL_CANDIDATES)
    cap_col = detect_column(df.columns, CAPTION_COL_CANDIDATES)
    label_col = detect_column(df.columns, LABEL_COL_CANDIDATES)

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
        "caption": df[cap_col].astype(str),
        "dataset": dataset_name,
    })

    if label_col is not None:
        out["label"] = df[label_col].astype(str).str.strip().str.lower()
    else:
        out["label"] = pd.NA

    print(f"  [{dataset_name}] {len(out)} rows | "
          f"path_col={path_col!r} | caption_col={cap_col!r} | label_col={label_col!r}")
    return out


def resolve_image_path(raw_path, csv_dir):
    """
    Try multiple strategies to locate an image file.
    Returns the first existing resolved path, or None.
    """
    basename = os.path.basename(str(raw_path))
    candidates = [
        str(raw_path),
        os.path.join(csv_dir, str(raw_path)),
        os.path.join(csv_dir, basename),
    ]
    # Also try immediate subfolders of csv_dir
    try:
        for entry in os.scandir(csv_dir):
            if entry.is_dir():
                candidates.append(os.path.join(entry.path, basename))
                candidates.append(os.path.join(entry.path, str(raw_path)))
    except OSError:
        pass

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def load_all_datasets(data_root):
    """
    Discover and load all datasets under data_root.
    Returns a combined DataFrame with columns: image_path, caption, dataset, label.
    """
    datasets_found = discover_datasets(data_root)
    print(f"Found {len(datasets_found)} dataset(s) under: {data_root}\n")
    for name, csv_path in datasets_found:
        print(f"  {name:40s}  ->  {csv_path}")

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
            if p is not None:
                n_ok += 1
            else:
                n_fail += 1

        df["resolved_path"] = resolved
        df_ok = df[df["resolved_path"].notna()].copy()
        df_ok["image_path"] = df_ok["resolved_path"]
        df_ok = df_ok.drop(columns=["resolved_path"])

        if len(df_ok) > 0:
            all_dfs.append(df_ok)

        status = "OK" if n_fail == 0 else f"{n_fail} missing"
        load_report.append((dataset_name, n_ok, n_fail, status))

    # Print summary
    print("\n" + "=" * 80)
    print(f"{'Dataset':<42} {'Loaded':>8} {'Failed':>8}  Status")
    print("=" * 80)
    for name, n_ok, n_fail, status in load_report:
        ok_str = f"{n_ok:,}" if n_ok else "-"
        fail_str = f"{n_fail:,}" if n_fail else "-"
        print(f"{name:<42} {ok_str:>8} {fail_str:>8}  {status}")
    print("=" * 80)
    print(f"{'TOTAL':<42} {sum(r[1] for r in load_report):>8,} {sum(r[2] for r in load_report):>8,}")

    if not all_dfs:
        raise RuntimeError(
            f"No data loaded successfully! Check that data_root exists: {data_root}"
        )

    full_df = pd.concat(all_dfs, ignore_index=True)[["image_path", "caption", "dataset", "label"]]
    print(f"\nCombined dataset: {len(full_df):,} samples from {full_df['dataset'].nunique()} source(s)")
    return full_df


def split_data(full_df, cfg):
    """
    Split full_df into train/val/test DataFrames, stratified by dataset source.
    Returns (train_df, val_df, test_df).
    """
    val_frac = cfg["val_size"] / (1.0 - cfg["test_size"])

    def safe_split(df, test_size, stratify_col, seed):
        try:
            return train_test_split(
                df, test_size=test_size,
                stratify=df[stratify_col], random_state=seed,
            )
        except ValueError as e:
            print(f"  [WARNING] Stratified split failed ({e}). Using random split.")
            return train_test_split(df, test_size=test_size, random_state=seed)

    train_val_df, test_df = safe_split(full_df, cfg["test_size"], "dataset", cfg["random_seed"])
    train_df, val_df = safe_split(train_val_df, val_frac, "dataset", cfg["random_seed"])

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Split sizes (total = {len(full_df):,})")
    print(f"  Train : {len(train_df):>6,}  ({100*len(train_df)/len(full_df):.1f}%)")
    print(f"  Val   : {len(val_df):>6,}  ({100*len(val_df)/len(full_df):.1f}%)")
    print(f"  Test  : {len(test_df):>6,}  ({100*len(test_df)/len(full_df):.1f}%)")

    return train_df, val_df, test_df


def build_dataloaders(train_df, val_df, test_df, cfg, mode="train"):
    """
    Build DataLoaders for training or evaluation.

    mode="train": returns train_loader, val_loader, test_loader (MultiDatasetUltrasound)
    mode="eval":  returns test_loader (LabeledUltrasoundDataset)
    """
    loader_kw = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )

    if mode == "train":
        train_dataset = MultiDatasetUltrasound(train_df, split="train", cfg=cfg)
        val_dataset = MultiDatasetUltrasound(val_df, split="val", cfg=cfg)
        test_dataset = MultiDatasetUltrasound(test_df, split="test", cfg=cfg)

        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kw)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kw)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kw)

        print("DataLoaders ready:")
        print(f"  Train : {len(train_loader):>4} batches  ({len(train_dataset):,} samples)")
        print(f"  Val   : {len(val_loader):>4} batches  ({len(val_dataset):,} samples)")
        print(f"  Test  : {len(test_loader):>4} batches  ({len(test_dataset):,} samples)")

        return train_loader, val_loader, test_loader

    elif mode == "eval":
        test_dataset = LabeledUltrasoundDataset(test_df, split="test", cfg=cfg)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kw)
        return test_loader

    elif mode == "labeled":
        # Returns (train_loader, val_loader, test_loader) all as LabeledUltrasoundDataset.
        train_dataset = LabeledUltrasoundDataset(train_df, split="train", cfg=cfg)
        val_dataset   = LabeledUltrasoundDataset(val_df,   split="test",  cfg=cfg)
        test_dataset  = LabeledUltrasoundDataset(test_df,  split="test",  cfg=cfg)
        train_loader = DataLoader(train_dataset, shuffle=False, drop_last=False, **loader_kw)
        val_loader   = DataLoader(val_dataset,   shuffle=False, drop_last=False, **loader_kw)
        test_loader  = DataLoader(test_dataset,  shuffle=False, drop_last=False, **loader_kw)
        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Labeled-split helpers (used by probe evaluations) ─────────────────────────

def infer_organ_from_dataset(dataset_name: str) -> str:
    """Map a dataset name to a canonical organ string."""
    s = str(dataset_name).strip().lower()
    if "busi" in s or "breast" in s:
        return "breast"
    if "lung" in s:
        return "lung"
    if "auli" in s or "liver" in s:
        return "liver"
    if "auitd" in s or "thyroid" in s:
        return "thyroid"
    if "artery" in s or "carotid" in s:
        return "carotid"
    return "unknown"


def prepare_labeled_eval_df(df: pd.DataFrame,
                             keep_unknown_organs: bool = False) -> pd.DataFrame:
    """
    Filter a split DataFrame to rows that have valid labels, add an 'organ'
    column (inferred from dataset name), and a stable 'row_id' index.

    Used before passing DataFrames to probe evaluation functions.
    """
    out = df.copy().reset_index(drop=True)
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out = out[~out["label"].isin(["", "nan", "none", "unknown"])].copy()
    out["organ"] = out["dataset"].apply(infer_organ_from_dataset)
    if not keep_unknown_organs:
        out = out[out["organ"] != "unknown"].copy()
    out = out.reset_index(drop=True)
    out["row_id"] = list(range(len(out)))
    return out
