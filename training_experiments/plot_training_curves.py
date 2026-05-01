"""
Plot training and validation loss curves, one figure per experiment.

Loads the per-experiment run summaries written by `run_experiment()` and draws
one plot per experiment (train solid, val dashed, black, y-axis fixed 0-5).
Also aggregates the raw loss traces into a single JSON + CSV so the data can
be re-used for custom plots later.

Run:
    python plot_training_curves.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from common import load_config


def plot_all_curves(config_path: str | os.PathLike | None = None,
                    out_dir: str | os.PathLike | None = None) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    out_dir = Path(out_dir) if out_dir else ckpt_dir / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: dict[str, dict] = {}
    for exp_key, exp in cfg["experiments"].items():
        summary_path = ckpt_dir / f"{exp_key}_summary.json"
        if not summary_path.exists():
            print(f"[skip] {summary_path} not found")
            continue
        with open(summary_path) as f:
            s = json.load(f)
        t_losses = s["train_losses"]
        v_losses = s["val_losses"]
        epochs = list(range(1, len(t_losses) + 1))
        label = s.get("label", exp_key)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(epochs, t_losses, color="black", linewidth=1.8, label="Train")
        ax.plot(epochs, v_losses, color="black", linewidth=1.8,
                linestyle="--", alpha=0.8, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("InfoNCE Loss")
        ax.set_ylim(0, 5)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()

        fig_path = out_dir / f"{exp_key}_training_curve.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot -> {fig_path}")

        traces[exp_key] = {
            "label": label,
            "epochs": epochs,
            "train_losses": t_losses,
            "val_losses": v_losses,
        }

    # Save aggregated traces for later re-plotting.
    traces_json = out_dir / "training_curves_traces.json"
    with open(traces_json, "w") as f:
        json.dump(traces, f, indent=2)
    print(f"Saved traces (JSON) -> {traces_json}")

    traces_csv = out_dir / "training_curves_traces.csv"
    with open(traces_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "label", "epoch", "train_loss", "val_loss"])
        for exp_key, tr in traces.items():
            for e, tl, vl in zip(tr["epochs"], tr["train_losses"], tr["val_losses"]):
                w.writerow([exp_key, tr["label"], e, tl, vl])
    print(f"Saved traces (CSV)  -> {traces_csv}")


if __name__ == "__main__":
    plot_all_curves()
