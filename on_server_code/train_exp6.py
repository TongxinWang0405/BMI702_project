"""
Experiment 6 — Bio_ClinicalBERT text encoder, MLP heads + Image Encoder
(text encoder frozen).

Run (example):
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_exp6.py > ../logs/exp6.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp6")
