"""
Evaluate alignment score (mean cosine similarity of paired image-text embeddings).
"""

import torch
from ..models.utils import tokenize_texts


@torch.no_grad()
def compute_alignment_score(model, dataloader, tokenizer, max_seq_len=77, device="cpu"):
    """
    Alignment score = mean cosine similarity between matched (image, text) pairs.
    Both embeddings are L2-normalized, so dot product == cosine similarity.
    Range: [-1, 1]. Higher is better.
    """
    model.eval()
    all_img, all_txt = [], []

    # Note: dataloader might be MultiDatasetUltrasound (returns 3 items) 
    # or LabeledUltrasoundDataset (returns 5 items).
    # We only care about the first two (image, caption).
    for batch in dataloader:
        images = batch[0].to(device)
        texts = batch[1]
        ids, mask = tokenize_texts(texts, tokenizer, max_length=max_seq_len, device=device)
        
        all_img.append(model.encode_image(images).cpu())
        all_txt.append(model.encode_text(ids, mask).cpu())

    img_embs = torch.cat(all_img)  # (N, D)
    txt_embs = torch.cat(all_txt)  # (N, D)

    # Diagonal elements are the matching image-text pairs
    paired_sim = (img_embs * txt_embs).sum(dim=1).mean().item()

    # Off-diagonal elements are the cross-pair similarities
    all_sim = img_embs @ txt_embs.T
    off_mask = ~torch.eye(img_embs.size(0), dtype=torch.bool)
    cross_sim = all_sim[off_mask].mean().item()

    return {
        "paired_similarity": paired_sim,
        "cross_pair_similarity": cross_sim
    }
