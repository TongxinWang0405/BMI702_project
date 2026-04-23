"""
Experiment 4 — Train MLP heads + Image Encoder + Text Encoder (full fine-tune).

Run (example):
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_exp4.py > ../logs/exp4.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp4")
