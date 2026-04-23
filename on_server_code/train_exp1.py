"""
Experiment 1 — Train MLP projection heads only (both encoders frozen).

Run (example):
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_exp1.py > ../logs/exp1.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp1")
