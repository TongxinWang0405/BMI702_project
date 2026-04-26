import os
import argparse
import yaml
import torch
import pandas as pd

from echocare_clip.data.loader import (
    load_all_datasets, split_data, build_dataloaders, prepare_labeled_eval_df,
)
from echocare_clip.models.utils import load_model, get_tokenizer, get_max_seq_len
from echocare_clip.evaluation import (
    compute_alignment_score,
    evaluate_zero_shot_classification,
    evaluate_linear_probe,
    evaluate_fewshot_probe,
    evaluate_fewshot_e2e,
    evaluate_retrieval_metrics,
    summarize_across_organs,
)


def run_evaluation(model, tokenizer,
                   test_loader,
                   labeled_train_df, labeled_val_df, labeled_test_df,
                   cfg, device, output_dir=None, max_seq_len=None):
    """Run all evaluation metrics for one model."""
    results = {}
    max_seq = max_seq_len if max_seq_len is not None else cfg["max_seq_len"]

    # 1. Alignment Score
    print("\n--- Alignment Score ---")
    align = compute_alignment_score(model, test_loader, tokenizer, max_seq, device)
    results["alignment"]       = align["paired_similarity"]
    results["cross_alignment"] = align["cross_pair_similarity"]
    print(f"  Paired:     {align['paired_similarity']:.4f}")
    print(f"  Cross-pair: {align['cross_pair_similarity']:.4f}")

    # 2. Zero-shot Classification
    print("\n--- Zero-Shot Classification ---")
    zs_acc = evaluate_zero_shot_classification(model, test_loader, tokenizer, max_seq, device)
    results["zero_shot_acc"] = zs_acc
    print(f"  Accuracy: {zs_acc:.4f}")

    # 3. Retrieval
    print("\n--- Retrieval Metrics ---")
    ret = evaluate_retrieval_metrics(model, test_loader, tokenizer, max_seq, device)
    results.update(ret)
    print(f"  I2T R@1: {ret['i2t_r@1']:.4f}  T2I R@1: {ret['t2i_r@1']:.4f}")

    # 4. Linear Probe
    print("\n--- Linear Probe ---")
    lp_per_organ, lp_agg = evaluate_linear_probe(
        model, labeled_train_df, labeled_val_df, labeled_test_df, cfg, device
    )
    if not lp_per_organ.empty:
        weighted = lp_agg[lp_agg["average_type"] == "weighted_by_test_n"]
        results["linear_probe_auroc"] = weighted["auroc"].values[0] if len(weighted) else float("nan")
        results["linear_probe_acc"]   = weighted["accuracy"].values[0] if len(weighted) else float("nan")
        print(lp_per_organ[["organ", "accuracy", "auroc", "f1_weighted", "n_train", "n_eval"]].to_string(index=False))
        print(lp_agg.to_string(index=False))
        if output_dir:
            lp_per_organ.to_csv(os.path.join(output_dir, "linear_probe_per_organ.csv"), index=False)
            lp_agg.to_csv(os.path.join(output_dir, "linear_probe_aggregate.csv"), index=False)
    else:
        results["linear_probe_auroc"] = float("nan")
        results["linear_probe_acc"]   = float("nan")
        print("  No labeled data.")

    # 5. Few-Shot Probe
    print("\n--- Few-Shot Probe ---")
    fs_per_organ, fs_agg = evaluate_fewshot_probe(
        model, labeled_train_df, labeled_val_df, labeled_test_df, cfg, device
    )
    if not fs_agg.empty:
        weighted = fs_agg[fs_agg["average_type"] == "weighted_by_test_n"]
        print(weighted[["fewshot_frac", "auroc_mean", "auroc_std",
                          "accuracy_mean", "f1_weighted_mean"]].to_string(index=False))
        if output_dir:
            fs_per_organ.to_csv(os.path.join(output_dir, "fewshot_probe_per_organ.csv"), index=False)
            fs_agg.to_csv(os.path.join(output_dir, "fewshot_probe_aggregate.csv"), index=False)
    else:
        print("  No labeled data.")

    # 6. End-to-End Few-Shot Fine-Tuning
    print("\n--- Few-Shot E2E Fine-Tuning ---")
    e2e_raw, e2e_agg = evaluate_fewshot_e2e(
        model, labeled_train_df, labeled_val_df, labeled_test_df,
        cfg, device, tokenizer=tokenizer,
    )
    if not e2e_agg.empty:
        weighted = e2e_agg[e2e_agg["average_type"] == "weighted_by_test_n"]
        print(weighted[["fewshot_frac", "auroc_mean", "auroc_std",
                          "accuracy_mean", "f1_weighted_mean"]].to_string(index=False))
        if output_dir:
            e2e_raw.to_csv(os.path.join(output_dir, "e2e_fewshot_per_organ.csv"), index=False)
            e2e_agg.to_csv(os.path.join(output_dir, "e2e_fewshot_aggregate.csv"), index=False)
    else:
        print("  No labeled data.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EchoCare-CLIP")
    parser.add_argument("--config", type=str, default="echocare_clip/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to a specific checkpoint")
    parser.add_argument("--text-encoder", type=str, choices=["clip", "bert"], default="clip")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all experiments defined in config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data setup ──────────────────────────────────────────────────────────────
    full_df = load_all_datasets(cfg["data_root"])
    train_df, val_df, test_df = split_data(full_df, cfg)

    # Full test set for alignment / zero-shot / retrieval
    test_loader = build_dataloaders(train_df, val_df, test_df, cfg, mode="eval")

    # Labeled + organ-annotated splits for probe evaluations
    labeled_train_df = prepare_labeled_eval_df(train_df)
    labeled_val_df   = prepare_labeled_eval_df(val_df)
    labeled_test_df  = prepare_labeled_eval_df(test_df)
    print(f"Labeled split sizes — train: {len(labeled_train_df):,}  "
          f"val: {len(labeled_val_df):,}  test: {len(labeled_test_df):,}")

    os.makedirs(cfg.get("output_dir", "./results"), exist_ok=True)

    # ── Evaluate ────────────────────────────────────────────────────────────────
    if args.all:
        summary = {}
        for exp_id, exp_cfg in cfg["experiments"].items():
            ckpt = os.path.join(cfg["model_dir"], cfg["checkpoint_files"][exp_id])
            if not os.path.exists(ckpt):
                print(f"Skipping Exp {exp_id} — checkpoint not found: {ckpt}")
                continue

            print(f"\n{'='*60}\nExp {exp_id}: {exp_cfg['name']}\n{'='*60}")
            enc_type  = exp_cfg["text_encoder"]
            model     = load_model(cfg, ckpt, text_encoder_type=enc_type, device=device)
            tokenizer = get_tokenizer(enc_type, cfg[f"{enc_type}_model_name"])

            out_dir = os.path.join(cfg.get("output_dir", "./results"), f"exp{exp_id}")
            os.makedirs(out_dir, exist_ok=True)

            res = run_evaluation(
                model, tokenizer,
                test_loader,
                labeled_train_df, labeled_val_df, labeled_test_df,
                cfg, device, output_dir=out_dir,
                max_seq_len=get_max_seq_len(cfg, enc_type),
            )
            summary[f"Exp {exp_id}"] = res

        # Summary table
        print("\n" + "=" * 75)
        print("SUMMARY")
        print("=" * 75)
        hdr = (f"{'Exp':10s} | {'Align':>6} | {'ZS Acc':>6} | {'I2T R@1':>7}"
               f" | {'LP AUC':>6} | {'LP Acc':>6}")
        print(hdr)
        print("-" * len(hdr))
        for name, res in summary.items():
            print(
                f"{name:10s}"
                f" | {res.get('alignment', float('nan')):6.3f}"
                f" | {res.get('zero_shot_acc', float('nan')):6.3f}"
                f" | {res.get('i2t_r@1', float('nan')):7.3f}"
                f" | {res.get('linear_probe_auroc', float('nan')):6.3f}"
                f" | {res.get('linear_probe_acc', float('nan')):6.3f}"
            )

        # Save summary
        pd.DataFrame(summary).T.to_csv(
            os.path.join(cfg.get("output_dir", "./results"), "summary.csv")
        )

    elif args.checkpoint:
        print(f"Evaluating: {args.checkpoint}")
        model     = load_model(cfg, args.checkpoint,
                               text_encoder_type=args.text_encoder, device=device)
        tokenizer = get_tokenizer(args.text_encoder,
                                  cfg[f"{args.text_encoder}_model_name"])
        run_evaluation(
            model, tokenizer,
            test_loader,
            labeled_train_df, labeled_val_df, labeled_test_df,
            cfg, device,
            output_dir=cfg.get("output_dir", "./results"),
            max_seq_len=get_max_seq_len(cfg, args.text_encoder),
        )

    else:
        parser.error("Provide --checkpoint <path> or --all")


if __name__ == "__main__":
    main()
