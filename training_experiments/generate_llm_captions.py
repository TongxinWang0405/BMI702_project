"""
Generate LLM captions for template-captioned ultrasound datasets.

We re-caption every row whose original caption was produced by filling
prompt templates on structured metadata (diagnosis, BI-RADS, shape,
margin, echogenicity, patient age/sex, region, findings text, etc.).
Datasets whose captions are already natural language (BUSCoT expert
captions, Carotid, USData Chinese translated findings) are left
untouched — rows from those datasets keep their original caption as
`llm_caption` so downstream training can read a single column.

Required raw-metadata layout (point `--raw-metadata-root` here):

    <raw_metadata_root>/
        BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx

No raw files are needed for BUSI, Thyroid, Liver, or Lung: their metadata
is already in the preprocessed captions.csv (BUSI, Thyroid, Lung) or in
the image filename prefix (Liver). Lung is rewritten from its existing
template caption because the source metadata.xlsx cannot be mapped onto
the renamed `lung_{cls}_{i}.png` filenames in the curated dataset.

Every loader below raises loudly on any missing / inconsistent input —
there are no silent fallbacks. If you get an error, fix the raw data.

Operating on the splits (not the source CSVs) means:
  - you do NOT need to re-run prepare_data.py
  - train/val/test/held-out membership is preserved EXACTLY
  - normalization statistics in config.json stay valid

Usage (on the server, after `pip install vllm openpyxl`):

    python generate_llm_captions.py \\
        --config config.json \\
        --raw-metadata-root /path/to/raw_metadata \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import DEFAULT_CONFIG_PATH, load_config
from prepare_data import (
    CAPTION_COL_CANDIDATES,
    PATH_COL_CANDIDATES,
    detect_column,
    discover_datasets,
)


SPLIT_NAMES = ["train", "val", "test", "full", "held_out"]


# ──────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────
def clean_value(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "unknown", "n/a", "na", "-"}:
        return None
    return s


def humanize_key(col_name: str) -> str:
    s = col_name.replace("_", " ").replace(".", " ").strip()
    if s.lower() == "birads":
        return "BI-RADS"
    return s


def _require_path_col(df: pd.DataFrame, csv_path: Path, tag: str) -> str:
    col = detect_column(df.columns, PATH_COL_CANDIDATES)
    if col is None:
        raise KeyError(
            f"[{tag}] {csv_path}: no image-path column in {list(df.columns)}; "
            f"expected one of {PATH_COL_CANDIDATES}"
        )
    return col


def _require_label_col(df: pd.DataFrame, csv_path: Path, tag: str) -> str:
    col = next((c for c in df.columns if c.lower() == "label"), None)
    if col is None:
        raise KeyError(
            f"[{tag}] {csv_path}: missing `label` column (have {list(df.columns)})"
        )
    return col


def _basenames_in_csv(captions_csv: Path, tag: str) -> list[str]:
    df = pd.read_csv(captions_csv)
    path_col = _require_path_col(df, captions_csv, tag)
    names = []
    for raw in df[path_col].astype(str):
        bn = os.path.basename(raw.strip())
        if bn:
            names.append(bn)
    return names


# ──────────────────────────────────────────────────────────────────────
# Per-dataset metadata loaders
# ──────────────────────────────────────────────────────────────────────
# Each loader returns {image_basename: {field_name: value}} for every
# image in that dataset's captions.csv. Loaders raise on any missing
# raw file or schema violation.
# ──────────────────────────────────────────────────────────────────────

_BUSI_CONDITIONS = {"benign", "malignant", "normal"}
_THYROID_CONDITIONS = {"benign", "malignant", "normal"}
_LIVER_CONDITIONS = {
    "liver_benign":    "benign",
    "liver_malignant": "malignant",
    "liver_normal":    "normal",
}

_BREAST_EXCEL_NAME = "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
_BREAST_SHEET = "BrEaST-Lesions-USG clinical dat"  # Excel truncates to 31 chars
_BREAST_FIELDS = [
    "Age", "Tissue_composition",
    "Shape", "Margin", "Echogenicity", "Posterior_features",
    "Calcifications", "Skin_thickening", "Halo", "Signs", "Symptoms",
    "Diagnosis", "Classification", "Interpretation", "BIRADS",
    "Verification",
]

# Lung uses its existing template caption directly — the source
# metadata.xlsx cannot be remapped onto the renamed filenames in the
# curated dataset. The LLM rewrites the existing caption into 3 variants.


def load_busi(captions_csv: Path, raw_root: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(captions_csv)
    path_col = _require_path_col(df, captions_csv, "BUSI")
    label_col = _require_label_col(df, captions_csv, "BUSI")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        basename = os.path.basename(str(row[path_col]).strip())
        if not basename:
            continue
        v = clean_value(row[label_col])
        if v is None:
            raise ValueError(f"[BUSI] empty label for {basename}")
        label = v.lower()
        if label not in _BUSI_CONDITIONS:
            raise ValueError(
                f"[BUSI] unexpected label {label!r} for {basename}; "
                f"expected one of {sorted(_BUSI_CONDITIONS)}"
            )
        out[basename] = {"Tissue": "breast", "Condition": label}
    return out


def load_thyroid(captions_csv: Path, raw_root: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(captions_csv)
    path_col = _require_path_col(df, captions_csv, "Thyroid")
    label_col = _require_label_col(df, captions_csv, "Thyroid")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        basename = os.path.basename(str(row[path_col]).strip())
        if not basename:
            continue
        v = clean_value(row[label_col])
        if v is None:
            raise ValueError(f"[Thyroid] empty label for {basename}")
        label = v.lower()
        if label not in _THYROID_CONDITIONS:
            raise ValueError(
                f"[Thyroid] unexpected label {label!r} for {basename}; "
                f"expected one of {sorted(_THYROID_CONDITIONS)}"
            )
        out[basename] = {"Tissue": "thyroid", "Condition": label}
    return out


def load_liver(captions_csv: Path, raw_root: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(captions_csv)
    path_col = _require_path_col(df, captions_csv, "Liver")
    out: dict[str, dict[str, str]] = {}
    for raw in df[path_col].astype(str):
        basename = os.path.basename(raw.strip())
        if not basename:
            continue
        prefix = "_".join(basename.split("_")[:2]).lower()
        condition = _LIVER_CONDITIONS.get(prefix)
        if condition is None:
            raise ValueError(
                f"[Liver] cannot parse condition from filename {basename!r}; "
                f"expected prefix one of {sorted(_LIVER_CONDITIONS)}"
            )
        out[basename] = {"Tissue": "liver", "Condition": condition}
    return out


def load_breast(captions_csv: Path, raw_root: Path) -> dict[str, dict[str, str]]:
    excel = raw_root / _BREAST_EXCEL_NAME
    if not excel.is_file():
        raise FileNotFoundError(
            f"[BrEaST] clinical-data Excel not found: {excel}. "
            f"Download it from the BrEaST dataset distribution and place "
            f"it at that exact path."
        )
    df_xl = pd.read_excel(excel, sheet_name=_BREAST_SHEET)
    df_xl["Image_filename"] = df_xl["Image_filename"].astype(str).str.strip()

    out: dict[str, dict[str, str]] = {}
    for _, row in df_xl.iterrows():
        fname = row["Image_filename"]
        if not fname or pd.isna(fname):
            continue
        fields: dict[str, str] = {}
        for c in _BREAST_FIELDS:
            v = clean_value(row.get(c))
            if v is not None:
                fields[humanize_key(c)] = v
        if fields:
            out[str(fname)] = fields

    # Every basename in captions.csv MUST be in the Excel.
    missing = [bn for bn in _basenames_in_csv(captions_csv, "BrEaST")
               if bn not in out]
    if missing:
        raise KeyError(
            f"[BrEaST] {len(missing)} basenames in {captions_csv} are not in "
            f"{excel} (e.g. {missing[:3]}). The Excel must be the same one "
            f"used by BrEaST_preprocess.ipynb."
        )
    return out


def load_lung(captions_csv: Path, raw_root: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(captions_csv)
    path_col = _require_path_col(df, captions_csv, "Lung")
    cap_col = detect_column(df.columns, CAPTION_COL_CANDIDATES)
    if cap_col is None:
        raise KeyError(
            f"[Lung] {captions_csv}: no caption column in {list(df.columns)}; "
            f"expected one of {CAPTION_COL_CANDIDATES}"
        )
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        basename = os.path.basename(str(row[path_col]).strip())
        if not basename:
            continue
        cap = clean_value(row[cap_col])
        if cap is None:
            raise ValueError(f"[Lung] empty caption for {basename}")
        out[basename] = {"Tissue": "lung", "OriginalCaption": cap}
    return out


# ──────────────────────────────────────────────────────────────────────
# Dataset dispatch
# ──────────────────────────────────────────────────────────────────────
# Case-insensitive substring match against the dataset folder name as
# surfaced by prepare_data.discover_datasets. Datasets not listed here
# are NOT re-captioned — their rows keep their original caption.
DATASET_LOADERS: list[tuple[str, Callable[[Path, Path], dict], str]] = [
    ("BUSI",             load_busi,     "BUSI (breast + label)"),
    ("BrEaST",           load_breast,   "BrEaST (clinical-data Excel)"),
    ("Liver_ultrasound", load_liver,    "Liver AULI (filename prefix)"),
    ("Lung_ultrasound",  load_lung,     "Lung (rewrite existing template caption)"),
    ("Thyroid",          load_thyroid,  "Thyroid (tissue + label)"),
]

# Folder-name substrings that mark a dataset as already having natural
# captions — must be skipped even if they otherwise match a template
# loader's token (e.g. "Thyroid USData..." contains "Thyroid").
_NATURAL_CAPTION_TOKENS = ("buscot", "carotid", "usdata")


def match_loader(dataset_name: str):
    low = dataset_name.lower()
    if any(tok in low for tok in _NATURAL_CAPTION_TOKENS):
        return None, None, None
    for tok, fn, desc in DATASET_LOADERS:
        if tok.lower() in low:
            return tok, fn, desc
    return None, None, None


# ──────────────────────────────────────────────────────────────────────
# Prompt + LLM output parsing
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a radiology assistant writing English captions for ultrasound images.
You are given structured metadata fields (key: value pairs) from a clinical dataset.
Produce THREE DIFFERENT captions for the SAME image.

Content rules (apply to every caption):
- Ground the caption ONLY in the provided fields. Never invent values, measurements, or findings.
- Incorporate EVERY provided field (diagnosis, BI-RADS, pathology, anatomical region, lesion descriptors, patient demographics, findings, etc.). Do not omit any.
- Read like natural clinical language written by a radiologist, not a key-value dump.
- Length: 15 to 60 words, one to three sentences. Length should scale with the number of provided fields: shorter captions (~15-25 words) when few fields are given, and longer captions (up to ~40-60 words, two or more sentences) when many fields are given, so that every field is naturally incorporated.

Diversity rules (across the three captions):
- Captions must differ meaningfully in wording, sentence structure, tone, or emphasis.
- Suggested variants: (1) concise and clinical, (2) descriptive and narrative, (3) reorder the emphasis.
- No near-duplicates or trivial paraphrases.

Output format (MUST follow exactly):
- Return ONLY the three captions, nothing else.
- Each caption is on its own single line.
- Prefix each line with "1. ", "2. ", "3. " (number, period, space) — no other numbering style.
- No preamble, no labels like "Caption 1:", no bullets, no quotation marks, no blank lines between captions, no trailing commentary.

Example of the required output format (content is illustrative only):
1. First caption text here.
2. Second caption text here.
3. Third caption text here."""


USER_PROMPT_TEMPLATE = (
    "Dataset: {dataset}\n"
    "Structured fields:\n{fields_block}\n\n"
    "Write the three captions now, following the output format exactly."
)


def build_user_prompt(dataset: str, fields: dict[str, str]) -> str:
    if not fields:
        raise ValueError(
            f"build_user_prompt called with no fields for dataset {dataset!r} — "
            f"the loader for this dataset returned an empty dict, which "
            f"should be impossible."
        )
    fields_block = "\n".join(f"- {k}: {v}" for k, v in fields.items())
    return USER_PROMPT_TEMPLATE.format(dataset=dataset, fields_block=fields_block)


_WHITESPACE_RE = re.compile(r"\s+")
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\(?\d+[\.\)\:]|\-|\*)\s*(.+?)\s*$")


def parse_candidates(text: str) -> list[str]:
    raw = text.strip().strip("`").strip()
    candidates: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED_LINE_RE.match(line)
        body = m.group(1) if m else line
        body = body.strip().strip("\"'").strip()
        if body:
            candidates.append(_WHITESPACE_RE.sub(" ", body))
    if not candidates:
        cleaned = _WHITESPACE_RE.sub(" ", raw).strip()
        return [cleaned] if cleaned else []
    return candidates


def pick_caption(text: str, rng: random.Random) -> str:
    cands = parse_candidates(text)
    return rng.choice(cands) if cands else ""


def compute_diversity_stats(captions: list[str]) -> dict[str, Any]:
    def tok(s: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", s.lower())

    per_cap_lens = [len(tok(c)) for c in captions]
    all_toks: list[str] = []
    bigrams: list[tuple[str, str]] = []
    for c in captions:
        ts = tok(c)
        all_toks.extend(ts)
        bigrams.extend(zip(ts, ts[1:]))
    total = len(all_toks) or 1
    total_bi = len(bigrams) or 1
    return {
        "n_captions": len(captions),
        "tokens_total": len(all_toks),
        "unique_unigrams": len(set(all_toks)),
        "unique_bigrams": len(set(bigrams)),
        "distinct_1": round(len(set(all_toks)) / total, 4),
        "distinct_2": round(len(set(bigrams)) / total_bi, 4),
        "length_mean_words": round(float(np.mean(per_cap_lens)) if per_cap_lens else 0.0, 2),
        "length_p50_words": int(np.percentile(per_cap_lens, 50)) if per_cap_lens else 0,
        "length_p90_words": int(np.percentile(per_cap_lens, 90)) if per_cap_lens else 0,
    }


def compute_token_stats(captions: list[str], tokenizer) -> dict[str, Any]:
    if not captions:
        return {}
    lens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in captions]
    return {
        "subtoken_mean": round(float(np.mean(lens)), 2),
        "subtoken_p50": int(np.percentile(lens, 50)),
        "subtoken_p90": int(np.percentile(lens, 90)),
        "subtoken_max": int(max(lens)),
    }


def chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
def setup_logger(log_dir: Path, run_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llm_captions_{run_id}.log"
    logger = logging.getLogger("llm_captions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    logger.info(f"Logging to {log_path}")
    return logger


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--raw-metadata-root", required=True,
                    help="Directory containing BrEaST-Lesions-USG-clinical-"
                         "data-Dec-15-2023.xlsx.")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="*", default=SPLIT_NAMES)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: cap total unique images processed.")
    ap.add_argument("--replace-caption", action="store_true",
                    help="Also write `{split}_df_llm.{parquet,csv}` where "
                         "the `caption` column is replaced by the LLM caption.")
    ap.add_argument("--promote-llm", action="store_true",
                    help="After writing `{split}_df_llm.*`, atomically "
                         "rename originals to `{split}_df_template.*` and "
                         "the llm files to `{split}_df.*`. Implies --replace-caption.")
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args()
    if args.promote_llm:
        args.replace_caption = True

    raw_root = Path(args.raw_metadata_root).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"--raw-metadata-root does not exist or is not a directory: {raw_root}"
        )

    cfg = load_config(args.config)
    root = cfg["paths"]["prepared_data_root"]
    splits_dir = Path(cfg["paths"]["data_split_dir"])
    logs_dir = Path(cfg["paths"].get("logs_dir", "../logs"))
    report_dir = Path(args.report_dir) if args.report_dir else logs_dir / "llm_captions"

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(report_dir, run_id)
    logger.info("=" * 72)
    logger.info(f"LLM caption generation run {run_id}")
    logger.info(f"  model               : {args.model}")
    logger.info(f"  sampling            : temp={args.temperature} "
                f"top_p={args.top_p} rep_pen={args.repetition_penalty} "
                f"seed={args.seed}")
    logger.info(f"  prepared_data_root  : {root}")
    logger.info(f"  raw_metadata_root   : {raw_root}")
    logger.info(f"  data_split_dir      : {splits_dir}")
    logger.info(f"  splits              : {args.splits}")
    logger.info("=" * 72)

    # ── Load split parquets ───────────────────────────────────────────
    split_dfs: dict[str, pd.DataFrame] = {}
    for sp in args.splits:
        p = splits_dir / f"{sp}_df.parquet"
        if not p.exists():
            logger.warning(f"  split file missing, skipping: {p}")
            continue
        df = pd.read_parquet(p)
        split_dfs[sp] = df
        logger.info(f"  loaded split {sp:<8s}: {len(df):>6,} rows from {p}")
    if not split_dfs:
        raise FileNotFoundError("No split files loaded. Run prepare_data.py first.")

    # ── Load metadata for every template-captioned dataset ───────────
    # Natural-caption datasets (no dispatch entry) are simply skipped —
    # their rows will carry their original caption as llm_caption.
    logger.info("Loading per-dataset metadata:")
    lookup: dict[str, dict[str, dict[str, str]]] = {}
    for ds_name, csv_path in discover_datasets(root):
        _, fn, desc = match_loader(ds_name)
        if fn is None:
            logger.info(f"  [{ds_name}] skipped (natural captions — keep as-is)")
            continue
        logger.info(f"  [{ds_name}] loading — {desc}")
        lookup[ds_name] = fn(Path(csv_path), raw_root)
        logger.info(f"  [{ds_name}] {len(lookup[ds_name]):,} basenames, "
                    f"fields sample = {sorted(next(iter(lookup[ds_name].values())).keys())}")
    if not lookup:
        raise RuntimeError(
            f"No template-captioned datasets found under {root}. "
            f"Nothing to re-caption."
        )

    template_ds = set(lookup.keys())

    # ── Collect unique image_paths across splits (template-ds only) ──
    # Also carry the original caption so we can compute pre-LLM stats
    # for comparison against the generated captions.
    combined_parts = []
    for sp, df in split_dfs.items():
        orig_cap_col = detect_column(df.columns, CAPTION_COL_CANDIDATES)
        if orig_cap_col is None:
            raise KeyError(
                f"[{sp}] no caption column in {list(df.columns)} "
                f"(expected one of {CAPTION_COL_CANDIDATES})"
            )
        part = df[["image_path", "dataset", orig_cap_col]].copy()
        part = part.rename(columns={orig_cap_col: "__orig_caption"})
        part["__split"] = sp
        combined_parts.append(part)
    combined = pd.concat(combined_parts, ignore_index=True)
    unique_all = combined.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    unique = unique_all[unique_all["dataset"].isin(template_ds)].reset_index(drop=True)
    logger.info(f"Unique images to re-caption: {len(unique):,} "
                f"(of {len(unique_all):,} total unique across splits)")
    if args.limit is not None:
        unique = unique.head(args.limit).copy()
        logger.info(f"DEBUG: limited to first {args.limit}")

    # ── Resolve fields for every row. Any miss is a hard error. ──────
    fields_per_row: list[dict[str, str]] = []
    for _, r in unique.iterrows():
        ds = r["dataset"]
        bn = os.path.basename(str(r["image_path"]))
        fields = lookup[ds].get(bn)
        if fields is None:
            raise KeyError(
                f"[{ds}] basename {bn!r} present in split but missing from "
                f"loader output. The split file and the per-dataset "
                f"captions.csv have drifted — rerun prepare_data.py."
            )
        fields_per_row.append(fields)

    # ── Load tokenizer + vLLM ────────────────────────────────────────
    try:
        import vllm
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"vLLM/transformers import failed: {e}. Install with "
            f"`pip install vllm transformers`."
        )

    logger.info(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── Pre-LLM (original/template) caption stats for comparison ──────
    orig_captions = unique["__orig_caption"].astype(str).tolist()
    orig_datasets = unique["dataset"].tolist()
    orig_diversity_global = compute_diversity_stats(orig_captions)
    orig_token_stats_global = compute_token_stats(orig_captions, tokenizer)
    logger.info("Pre-LLM (template) caption stats:")
    logger.info(f"  global diversity  : {json.dumps(orig_diversity_global)}")
    logger.info(f"  global token stat : {json.dumps(orig_token_stats_global)}")
    orig_diversity_by_ds: dict[str, dict] = {}
    orig_token_stats_by_ds: dict[str, dict] = {}
    for ds in sorted(set(orig_datasets)):
        caps_ds = [c for c, d in zip(orig_captions, orig_datasets) if d == ds]
        orig_diversity_by_ds[ds] = compute_diversity_stats(caps_ds)
        orig_token_stats_by_ds[ds] = compute_token_stats(caps_ds, tokenizer)
        logger.info(f"  [{ds}] n={len(caps_ds):,} "
                    f"diversity={json.dumps(orig_diversity_by_ds[ds])} "
                    f"tokens={json.dumps(orig_token_stats_by_ds[ds])}")

    prompts = [
        chat_prompt(tokenizer, SYSTEM_PROMPT, build_user_prompt(r["dataset"], f))
        for (_, r), f in zip(unique.iterrows(), fields_per_row)
    ]

    logger.info(f"Loading vLLM engine for {args.model} ...")
    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        seed=args.seed,
        trust_remote_code=True,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
    )

    logger.info(f"Generating {len(prompts):,} captions ...")
    gen_t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_elapsed = time.time() - gen_t0
    pick_rng = random.Random(args.seed)
    captions = [pick_caption(o.outputs[0].text, pick_rng) for o in outputs]
    n_empty = sum(1 for c in captions if not c)
    if n_empty:
        raise RuntimeError(
            f"{n_empty} empty completions returned by the model. Inspect "
            f"the vLLM output and either raise --max-new-tokens or fix the "
            f"prompt."
        )
    logger.info(f"Generation done in {gen_elapsed:.1f}s "
                f"({len(prompts)/max(gen_elapsed,1e-6):.1f} prompts/s)")

    # ── Stats ─────────────────────────────────────────────────────────
    diversity_global = compute_diversity_stats(captions)
    token_stats_global = compute_token_stats(captions, tokenizer)
    logger.info("Post-LLM caption stats:")
    logger.info(f"  global diversity  : {json.dumps(diversity_global)}")
    logger.info(f"  global token stat : {json.dumps(token_stats_global)}")
    diversity_by_ds: dict[str, dict] = {}
    token_stats_by_ds: dict[str, dict] = {}
    for ds in sorted(set(orig_datasets)):
        caps_ds = [c for c, d in zip(captions, orig_datasets) if d == ds]
        diversity_by_ds[ds] = compute_diversity_stats(caps_ds)
        token_stats_by_ds[ds] = compute_token_stats(caps_ds, tokenizer)
        logger.info(f"  [{ds}] n={len(caps_ds):,} "
                    f"diversity={json.dumps(diversity_by_ds[ds])} "
                    f"tokens={json.dumps(token_stats_by_ds[ds])}")

    # ── Side-by-side comparison (pre-LLM vs post-LLM) ────────────────
    logger.info("Comparison (pre-LLM → post-LLM):")
    logger.info(f"  GLOBAL distinct_1 {orig_diversity_global['distinct_1']} → "
                f"{diversity_global['distinct_1']} | "
                f"distinct_2 {orig_diversity_global['distinct_2']} → "
                f"{diversity_global['distinct_2']} | "
                f"len_mean {orig_diversity_global['length_mean_words']} → "
                f"{diversity_global['length_mean_words']} | "
                f"subtok_mean {orig_token_stats_global.get('subtoken_mean')} → "
                f"{token_stats_global.get('subtoken_mean')}")
    for ds in sorted(set(orig_datasets)):
        o, n = orig_diversity_by_ds[ds], diversity_by_ds[ds]
        ot, nt = orig_token_stats_by_ds[ds], token_stats_by_ds[ds]
        logger.info(f"  [{ds}] distinct_1 {o['distinct_1']} → {n['distinct_1']} | "
                    f"distinct_2 {o['distinct_2']} → {n['distinct_2']} | "
                    f"len_mean {o['length_mean_words']} → {n['length_mean_words']} | "
                    f"subtok_mean {ot.get('subtoken_mean')} → {nt.get('subtoken_mean')}")

    # ── Write back to each split ──────────────────────────────────────
    cap_by_path = dict(zip(unique["image_path"].tolist(), captions))
    for sp, df in split_dfs.items():
        df_aug = df.copy()
        orig_cap_col = detect_column(df_aug.columns, CAPTION_COL_CANDIDATES)
        if orig_cap_col is None:
            raise KeyError(
                f"[{sp}] no caption column in {list(df_aug.columns)} "
                f"(expected one of {CAPTION_COL_CANDIDATES})"
            )

        df_aug["llm_caption"] = df_aug["image_path"].map(cap_by_path)
        is_template = df_aug["dataset"].isin(template_ds)
        # Natural-caption rows reuse the original caption so the column
        # is uniformly populated downstream.
        df_aug.loc[~is_template, "llm_caption"] = df_aug.loc[~is_template, orig_cap_col]

        n_missing_tmpl = int(df_aug.loc[is_template, "llm_caption"].isna().sum())
        if n_missing_tmpl and args.limit is None:
            raise RuntimeError(
                f"[{sp}] {n_missing_tmpl} template-dataset rows have no "
                f"LLM caption even though --limit was not set — this should "
                f"be unreachable."
            )
        if n_missing_tmpl:
            # --limit was set; fall back to the original caption just for
            # the unprocessed rows so the file is still writable.
            df_aug["llm_caption"] = df_aug["llm_caption"].fillna(df_aug[orig_cap_col])

        out_parquet = splits_dir / f"{sp}_df.parquet"
        out_csv = splits_dir / f"{sp}_df.csv"
        df_aug.to_parquet(out_parquet, index=False)
        df_aug.to_csv(out_csv, index=False)
        logger.info(f"  [{sp}] wrote augmented split (+llm_caption) "
                    f"→ {out_parquet.name} & {out_csv.name}")

        if args.replace_caption:
            df_llm = df_aug.copy()
            df_llm[orig_cap_col] = df_llm["llm_caption"]
            llm_parquet = splits_dir / f"{sp}_df_llm.parquet"
            llm_csv = splits_dir / f"{sp}_df_llm.csv"
            df_llm.to_parquet(llm_parquet, index=False)
            df_llm.to_csv(llm_csv, index=False)
            logger.info(f"  [{sp}] wrote LLM-caption split "
                        f"→ {llm_parquet.name} & {llm_csv.name}")

    # ── Optional promotion ────────────────────────────────────────────
    if args.promote_llm:
        logger.info("Promoting LLM splits to be the default for training ...")
        for sp in split_dfs.keys():
            for ext in (".parquet", ".csv"):
                orig = splits_dir / f"{sp}_df{ext}"
                parked = splits_dir / f"{sp}_df_template{ext}"
                llm = splits_dir / f"{sp}_df_llm{ext}"
                if not llm.exists():
                    continue
                if parked.exists():
                    raise FileExistsError(
                        f"{parked} already exists — refuse to overwrite. "
                        f"Move it aside and re-run, or skip --promote-llm."
                    )
                shutil.move(str(orig), str(parked))
                shutil.move(str(llm), str(orig))
                logger.info(f"  promoted: {orig.name} (now LLM), "
                            f"{parked.name} (template backup)")

    # ── Aggregate report ──────────────────────────────────────────────
    aggregate = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model": args.model,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "prepared_data_root": root,
        "raw_metadata_root": str(raw_root),
        "data_split_dir": str(splits_dir),
        "splits_processed": list(split_dfs.keys()),
        "unique_images_captioned": len(unique),
        "generation_seconds": round(gen_elapsed, 2),
        "global_diversity": diversity_global,
        "global_token_stats": token_stats_global,
        "per_dataset_diversity": diversity_by_ds,
        "per_dataset_token_stats": token_stats_by_ds,
        "pre_llm_global_diversity": orig_diversity_global,
        "pre_llm_global_token_stats": orig_token_stats_global,
        "pre_llm_per_dataset_diversity": orig_diversity_by_ds,
        "pre_llm_per_dataset_token_stats": orig_token_stats_by_ds,
        "replace_caption": args.replace_caption,
        "promote_llm": args.promote_llm,
        "datasets_recaptioned": sorted(template_ds),
    }
    report_path = report_dir / f"llm_captions_{run_id}_report.json"
    with open(report_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    logger.info(f"Aggregate report → {report_path}")

    # ── Sample captions for sanity check ──────────────────────────────
    n_examples = min(5, len(unique))
    if n_examples > 0:
        sample_rng = random.Random(args.seed)
        sample_idx = sample_rng.sample(range(len(unique)), n_examples)
        logger.info("=" * 72)
        logger.info(f"Sample captions ({n_examples} of {len(unique)}):")
        for i in sample_idx:
            r = unique.iloc[i]
            fields = fields_per_row[i]
            fields_str = "; ".join(f"{k}={v}" for k, v in fields.items())
            logger.info(f"  — [{r['dataset']}] {os.path.basename(str(r['image_path']))}")
            logger.info(f"      metadata : {fields_str}")
            logger.info(f"      caption  : {captions[i]}")
        logger.info("=" * 72)

    logger.info("Done.")


if __name__ == "__main__":
    main()
