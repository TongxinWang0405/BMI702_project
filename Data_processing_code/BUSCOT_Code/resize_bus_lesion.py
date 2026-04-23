from pathlib import Path
from PIL import Image
import csv

# Input dataset root
root = Path("/Users/lanrr/Downloads/BUSCoT/BUS-Lesion")

# Output root
output_root = Path("/Users/lanrr/Downloads/BUSCoT_preprocessed/BUS-Lesion")
trainval_out = output_root / "trainval"
test_out = output_root / "test"
csv_path = output_root / "bus_lesion_manifest.csv"

# Target size
target_size = (256, 256)

for folder in [trainval_out, test_out]:
    folder.mkdir(parents=True, exist_ok=True)

rows = []
count = 0
skipped = 0

def resize_and_save(img_path: Path, out_path: Path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img.save(out_path, format="PNG")

for split_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    split_name = split_dir.name

    if split_name == "trainval":
        out_dir = trainval_out
    elif split_name == "test":
        out_dir = test_out
    else:
        continue

    for img_path in sorted(split_dir.glob("*.png")):
        out_path = out_dir / img_path.name

        try:
            resize_and_save(img_path, out_path)
            count += 1

            rows.append({
                "split": split_name,
                "file_name": img_path.name,
                "output_subfolder": out_dir.name,
                "output_file": out_path.name
            })

            if count % 500 == 0:
                print(f"Processed {count} images...")

        except Exception as e:
            skipped += 1
            print(f"Skipped {img_path}: {e}")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["split", "file_name", "output_subfolder", "output_file"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nDone.")
print(f"Images saved: {count}")
print(f"Skipped: {skipped}")
print(f"Manifest saved to: {csv_path}")
print(f"Output root: {output_root}")