"""
Experiment 3 — Train MLP heads + Text Encoder (image encoder frozen).

Run (example):
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_exp3.py > ../logs/exp3.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp3")
