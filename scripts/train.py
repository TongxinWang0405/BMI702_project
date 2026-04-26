import os
import argparse
import yaml
import torch

from echocare_clip.data.loader import load_all_datasets, split_data, build_dataloaders
from echocare_clip.models.utils import build_model, get_tokenizer, tokenize_texts, get_max_seq_len

def train_one_epoch(model, dataloader, optimizer, tokenizer, cfg, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, texts, _) in enumerate(dataloader):
        images = images.to(device)
        ids, mask = tokenize_texts(texts, tokenizer, max_length=max_seq, device=device)
        
        optimizer.zero_grad()
        loss, _, _ = model(images, ids, mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
            print(f"  [E{epoch+1}] Batch {batch_idx+1}/{len(dataloader)}"
                  f" | loss={loss.item():.4f}"
                  f" | temp={model.log_temperature.exp().item():.4f}")
                  
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, tokenizer, cfg, device):
    model.eval()
    total_loss = 0.0
    for images, texts, _ in dataloader:
        images = images.to(device)
        ids, mask = tokenize_texts(texts, tokenizer, max_length=max_seq, device=device)
        loss, _, _ = model(images, ids, mask)
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train EchoCare-CLIP")
    parser.add_argument("--config", type=str, default="echocare_clip/configs/default.yaml")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number (1-8)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    if args.exp not in cfg["experiments"]:
        raise ValueError(f"Experiment {args.exp} not defined in config")
        
    exp_cfg = cfg["experiments"][args.exp]
    enc_type = exp_cfg["text_encoder"]
    max_seq  = get_max_seq_len(cfg, enc_type)
    print(f"=== Starting Experiment {args.exp}: {exp_cfg['name']} ===")
    print(f"Text encoder: {enc_type}  max_seq_len: {max_seq}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    full_df = load_all_datasets(cfg["data_root"])
    train_df, val_df, test_df = split_data(full_df, cfg)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, cfg, mode="train")

    # Build model
    model = build_model(
        cfg,
        text_encoder_type=enc_type,
        freeze_image=exp_cfg["freeze_image"],
        freeze_text=exp_cfg["freeze_text"],
        device=device
    )

    tokenizer = get_tokenizer(
        text_encoder_type=enc_type,
        model_name=cfg[f"{enc_type}_model_name"]
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["lr"]), 
        weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

    best_val = float('inf')
    os.makedirs(cfg["model_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["model_dir"], cfg["checkpoint_files"][args.exp])

    print(f"Training for {cfg['num_epochs']} epochs...")
    for epoch in range(cfg["num_epochs"]):
        t_loss = train_one_epoch(model, train_loader, optimizer, tokenizer, cfg, device, epoch)
        v_loss = validate_one_epoch(model, val_loader, tokenizer, cfg, device)
        scheduler.step()
        
        lr_now = scheduler.get_last_lr()[0]
        print(f">>> Epoch {epoch+1:2d}/{cfg['num_epochs']}  Train={t_loss:.4f}  Val={v_loss:.4f}  LR={lr_now:.2e}")
        
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Checkpoint] best val={best_val:.4f} -> {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
