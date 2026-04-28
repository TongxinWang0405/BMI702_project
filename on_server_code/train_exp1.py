"""
Experiment 1 — Train MLP projection heads only (both encoders frozen).

Run (example):
    CUDA_VISIBLE_DEVICES=1 nohup python -u train_exp1.py > ../logs_llm/exp1_llm.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp1")
