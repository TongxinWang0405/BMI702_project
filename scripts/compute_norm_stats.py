import argparse
import yaml
import pprint
import os

from echocare_clip.data.normalization import compute_normalization_stats
from echocare_clip.data.loader import discover_datasets

def main():
    parser = argparse.ArgumentParser(description="Compute dataset mean and std")
    parser.add_argument("--config", type=str, default="echocare_clip/configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    datasets_found = discover_datasets(data_root)
    
    if not datasets_found:
        print(f"No datasets found in {data_root}")
        return

    # Collect all image paths (we can just load the CSVs and get the paths)
    from echocare_clip.data.loader import load_dataset_csv, resolve_image_path
    
    all_paths = []
    for name, csv_path in datasets_found:
        csv_dir = os.path.dirname(csv_path)
        df = load_dataset_csv(csv_path, name)
        for raw_path in df["image_path"]:
            p = resolve_image_path(raw_path, csv_dir)
            if p is not None:
                all_paths.append(p)
                
    print(f"Found {len(all_paths)} total images to compute stats.")
    
    mean, std = compute_normalization_stats(
        all_paths, 
        image_size=cfg["image_size"],
        batch_size=64,
        num_workers=cfg.get("num_workers", 2)
    )
    
    print("\nUpdate your config with:")
    print(f"img_mean: {mean}")
    print(f"img_std:  {std}")

if __name__ == "__main__":
    main()
