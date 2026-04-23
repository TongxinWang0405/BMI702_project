"""
Experiment 8 — Bio_ClinicalBERT text encoder, MLP heads + Image + Text Encoders
(full fine-tune).

Run (example):
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_exp8.py > ../logs/exp8.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp8")
