"""
Experiment 5 — Bio_ClinicalBERT text encoder, MLP heads only
(both image and text encoders frozen).

Run (example):
    CUDA_VISIBLE_DEVICES=2 nohup python -u train_exp5.py > ../logs_llm/exp5_llm.log 2>&1 &
"""
from common import run_experiment

if __name__ == "__main__":
    run_experiment("exp5")
