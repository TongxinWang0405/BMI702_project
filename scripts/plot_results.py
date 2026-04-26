"""
Generate all result plots from pre-computed CSVs in results/.
Run from project root:  python plot_results.py
Figures are saved to results/figures/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EXP_NAMES = {
    1: "MLP only\n(CLIP)",
    2: "MLP+Img\n(CLIP)",
    3: "MLP+Txt\n(CLIP)",
    4: "MLP+Img+Txt\n(CLIP)",
    5: "MLP only\n(BERT)",
    6: "MLP+Img\n(BERT)",
    7: "MLP+Txt\n(BERT)",
    8: "MLP+Img+Txt\n(BERT)",
}

# Color scheme: CLIP experiments blue family, BERT experiments orange family
EXP_COLORS = {
    1: "#4878CF", 2: "#2A5094", 3: "#6EA6CD", 4: "#1A3A6A",
    5: "#E8602C", 6: "#B83A1A", 7: "#F4A460", 8: "#8B2500",
}

EXPS = list(range(1, 9))


# ── Helpers ──────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def annotate_bars(ax, bars, fmt=".3f", offset_frac=0.01):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + span * offset_frac,
                f"{h:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )


def bar_chart(names, values, title, ylabel, colors=None, ylim_bottom=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors or "#4878CF", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    bottom = ylim_bottom if ylim_bottom is not None else min(0, np.nanmin(values) - 0.05)
    ax.set_ylim(bottom, np.nanmax(values) + 0.12)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars)
    plt.tight_layout()
    return fig


# ── Load data ────────────────────────────────────────────────────────────────
summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"), index_col=0)
summary.index = [f"Exp {i}" for i in EXPS]

names   = [EXP_NAMES[i] for i in EXPS]
colors  = [EXP_COLORS[i] for i in EXPS]


# ── 1. Alignment ─────────────────────────────────────────────────────────────
print("Plotting alignment...")
fig = bar_chart(
    names, summary["alignment"].values,
    title="Paired alignment score (test set)",
    ylabel="Mean paired cosine similarity",
    colors=colors,
)
save(fig, "01_alignment_paired.png")

fig = bar_chart(
    names, summary["cross_alignment"].values,
    title="Cross-pair alignment score (test set)",
    ylabel="Mean cross-pair cosine similarity",
    colors=colors,
)
save(fig, "02_alignment_cross.png")


# ── 2. Zero-shot classification ───────────────────────────────────────────────
print("Plotting zero-shot classification...")
fig = bar_chart(
    names, summary["zero_shot_acc"].values,
    title="Zero-shot label-prompt classification (test set)",
    ylabel="Accuracy",
    colors=colors,
    ylim_bottom=0,
)
save(fig, "03_zero_shot_acc.png")


# ── 3. Retrieval ──────────────────────────────────────────────────────────────
print("Plotting retrieval...")
metrics = ["i2t_r@1", "t2i_r@1", "i2t_r@5", "t2i_r@5", "i2t_r@10", "t2i_r@10"]
n_metrics = len(metrics)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
axes = axes.flatten()

for idx, col in enumerate(metrics):
    ax = axes[idx]
    vals = summary[col].values
    bars = ax.bar(np.arange(len(names)), vals, color=colors, width=0.6)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=35, ha="right")
    ax.set_title(col.upper(), fontsize=10)
    ax.set_ylim(0, max(vals.max() * 1.25, 0.05))
    ax.grid(axis="y", alpha=0.25)
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)

fig.suptitle("Retrieval Recall@K (Image↔Text, test set)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "04_retrieval.png")


# ── 4. Linear probe aggregate AUROC ──────────────────────────────────────────
print("Plotting linear probe...")

lp_rows = []
for i in EXPS:
    path = os.path.join(RESULTS_DIR, f"exp{i}", "linear_probe_aggregate.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    row = df[df["average_type"] == "weighted_by_test_n"].iloc[0]
    lp_rows.append({"exp": i, "auroc": row["auroc"], "accuracy": row["accuracy"]})

lp_df = pd.DataFrame(lp_rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, label, title in [
    (axes[0], "auroc",    "Weighted AUROC",    "Linear Probe — Weighted AUROC"),
    (axes[1], "accuracy", "Weighted Accuracy",  "Linear Probe — Weighted Accuracy"),
]:
    vals = [lp_df[lp_df["exp"] == i][metric].values[0] if i in lp_df["exp"].values else float("nan") for i in EXPS]
    bars = ax.bar(np.arange(len(names)), vals, color=colors, width=0.6)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=0, fontsize=9)
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars)

plt.tight_layout()
save(fig, "05_linear_probe_aggregate.png")


# ── 5. Linear probe per-organ breakdown ──────────────────────────────────────
print("Plotting linear probe per organ...")

per_organ_rows = []
for i in EXPS:
    path = os.path.join(RESULTS_DIR, f"exp{i}", "linear_probe_per_organ.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df["exp"] = i
    df["exp_name"] = EXP_NAMES[i].replace("\n", " ")
    per_organ_rows.append(df)

if per_organ_rows:
    po_df = pd.concat(per_organ_rows, ignore_index=True)
    organs = sorted(po_df["organ"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(organs) * 2), 5))
    n_models = len(EXPS)
    width = 0.8 / n_models
    x = np.arange(len(organs))

    for j, i in enumerate(EXPS):
        sub = po_df[po_df["exp"] == i].set_index("organ").reindex(organs)
        ax.bar(x + (j - (n_models - 1) / 2) * width, sub["auroc"].values,
               width=width, label=EXP_NAMES[i].replace("\n", " "),
               color=EXP_COLORS[i], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in organs], fontsize=10)
    ax.set_ylabel("AUROC")
    ax.set_title("Linear Probe AUROC — Per Organ")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    plt.tight_layout()
    save(fig, "06_linear_probe_per_organ.png")


# ── 6. Few-shot probe AUROC curves ───────────────────────────────────────────
print("Plotting few-shot probe curves...")

fs_rows = []
for i in EXPS:
    path = os.path.join(RESULTS_DIR, f"exp{i}", "fewshot_probe_aggregate.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df = df[df["average_type"] == "weighted_by_test_n"].copy()
    df["exp"] = i
    fs_rows.append(df)

if fs_rows:
    fs_df = pd.concat(fs_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in EXPS:
        sub = fs_df[fs_df["exp"] == i].sort_values("fewshot_frac")
        if sub.empty:
            continue
        x = sub["fewshot_frac"].values * 100
        y = sub["auroc_mean"].values
        yerr = sub["auroc_std"].fillna(0).values
        label = EXP_NAMES[i].replace("\n", " ")
        ax.plot(x, y, marker="o", label=label, color=EXP_COLORS[i])
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.12, color=EXP_COLORS[i])

    ax.set_xlabel("Labeled training fraction (%)")
    ax.set_ylabel("Weighted aggregate AUROC")
    ax.set_title("Few-Shot Probe Adaptation (test set)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save(fig, "07_fewshot_probe_curves.png")


# ── 7. End-to-end few-shot AUROC curves ──────────────────────────────────────
print("Plotting e2e few-shot curves...")

e2e_rows = []
for i in EXPS:
    path = os.path.join(RESULTS_DIR, f"exp{i}", "e2e_fewshot_aggregate.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df = df[df["average_type"] == "weighted_by_test_n"].copy()
    df["exp"] = i
    e2e_rows.append(df)

if e2e_rows:
    e2e_df = pd.concat(e2e_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in EXPS:
        sub = e2e_df[e2e_df["exp"] == i].sort_values("fewshot_frac")
        if sub.empty:
            continue
        x = sub["fewshot_frac"].values * 100
        y = sub["auroc_mean"].values
        yerr = sub["auroc_std"].fillna(0).values
        label = EXP_NAMES[i].replace("\n", " ")
        ax.plot(x, y, marker="o", label=label, color=EXP_COLORS[i])
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.12, color=EXP_COLORS[i])

    ax.set_xlabel("Labeled support fraction (%)")
    ax.set_ylabel("Weighted aggregate AUROC")
    ax.set_title("End-to-End Few-Shot Fine-Tuning (test set)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    save(fig, "08_e2e_fewshot_curves.png")


# ── 8. Summary heatmap ────────────────────────────────────────────────────────
print("Plotting summary heatmap...")

heat_cols = {
    "alignment":        "Alignment",
    "zero_shot_acc":    "ZS Acc",
    "i2t_r@1":          "I2T R@1",
    "t2i_r@1":          "T2I R@1",
    "linear_probe_auroc": "LP AUROC",
    "linear_probe_acc": "LP Acc",
}

heat_df = summary[list(heat_cols.keys())].rename(columns=heat_cols)
heat_arr = heat_df.values.astype(float)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heat_arr, aspect="auto", cmap="RdYlGn",
               vmin=np.nanmin(heat_arr), vmax=np.nanmax(heat_arr))
plt.colorbar(im, ax=ax, fraction=0.03)

ax.set_xticks(range(len(heat_cols)))
ax.set_xticklabels(list(heat_cols.values()), fontsize=10)
ax.set_yticks(range(len(EXPS)))
ax.set_yticklabels([EXP_NAMES[i].replace("\n", " ") for i in EXPS], fontsize=9)
ax.set_title("Summary Heatmap — All Metrics × All Experiments", fontsize=12)

for r in range(heat_arr.shape[0]):
    for c in range(heat_arr.shape[1]):
        v = heat_arr[r, c]
        if np.isfinite(v):
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < (v - np.nanmin(heat_arr)) / (np.nanmax(heat_arr) - np.nanmin(heat_arr) + 1e-9) < 0.7 else "white")

plt.tight_layout()
save(fig, "09_summary_heatmap.png")

print(f"\nAll figures saved to {FIG_DIR}/")
