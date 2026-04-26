"""
Evaluate retrieval metrics (Recall@K).
"""

import torch
import numpy as np
from ..models.utils import tokenize_texts


def compute_recall_at_k(sim_matrix, ks=(1, 5, 10)):
    """
    Assumes row i matches column i.
    Returns recall@k for row->col and col->row retrieval.
    """
    n = sim_matrix.shape[0]
    ranks_r2c = []
    ranks_c2r = []
    
    # Row -> Col (e.g., Image -> Text)
    # Sort columns descending
    sorted_idx_r2c = np.argsort(-sim_matrix, axis=1)
    for i in range(n):
        rank = np.where(sorted_idx_r2c[i] == i)[0][0]
        ranks_r2c.append(rank)
        
    # Col -> Row (e.g., Text -> Image)
    # Sort rows descending
    sorted_idx_c2r = np.argsort(-sim_matrix, axis=0)
    for j in range(n):
        rank = np.where(sorted_idx_c2r[:, j] == j)[0][0]
        ranks_c2r.append(rank)
        
    ranks_r2c = np.array(ranks_r2c)
    ranks_c2r = np.array(ranks_c2r)
    
    res = {}
    for k in ks:
        res[f"i2t_r@{k}"] = (ranks_r2c < k).mean()
        res[f"t2i_r@{k}"] = (ranks_c2r < k).mean()
        
    res["i2t_mean_rank"] = ranks_r2c.mean() + 1
    res["t2i_mean_rank"] = ranks_c2r.mean() + 1
    
    return res


@torch.no_grad()
def evaluate_retrieval_metrics(model, dataloader, tokenizer, max_seq_len=77, device="cpu", ks=(1, 5, 10)):
    """Evaluate image-to-text and text-to-image retrieval."""
    model.eval()
    all_img, all_txt = [], []

    for batch in dataloader:
        images = batch[0].to(device)
        texts = batch[1]
        ids, mask = tokenize_texts(texts, tokenizer, max_length=max_seq_len, device=device)
        
        all_img.append(model.encode_image(images).cpu())
        all_txt.append(model.encode_text(ids, mask).cpu())

    img_embs = torch.cat(all_img).numpy()
    txt_embs = torch.cat(all_txt).numpy()

    sim_matrix = img_embs @ txt_embs.T
    
    metrics = compute_recall_at_k(sim_matrix, ks=ks)
    return metrics
