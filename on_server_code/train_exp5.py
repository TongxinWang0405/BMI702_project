"""
Experiment 5 — Bio_ClinicalBERT text encoder, MLP heads only
(both image and text encoders frozen).

Run (example):
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_exp5.py > ../logs/exp5.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp5")
