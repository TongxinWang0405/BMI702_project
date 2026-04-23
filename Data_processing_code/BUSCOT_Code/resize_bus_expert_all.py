from pathlib import Path
from PIL import Image
import csv

# Input dataset root
root = Path("/Users/lanrr/Downloads/BUSCoT/BUS-Expert")

# Output root
output_root = Path("/Users/lanrr/Downloads/BUSCoT_preprocessed/BUS-Expert")
raw_out = output_root / "raw_images"
cropped_out = output_root / "cropped_images"
mask_out = output_root / "lesion_masks"
vis_out = output_root / "visboxmask_images"
csv_path = output_root / "bus_expert_manifest.csv"

# Target size
target_size = (256, 256)

for folder in [raw_out, cropped_out, mask_out, vis_out]:
    folder.mkdir(parents=True, exist_ok=True)

rows = []
counts = {
    "raw": 0,
    "cropped": 0,
    "lesion_mask": 0,
    "visboxmask": 0,
    "skipped": 0
}

def classify_file(file_path: Path) -> str:
    name = file_path.name
    if "@raw.png" in name:
        return "raw"
    if "@cropped.png" in name:
        return "cropped"
    if name.endswith("_VISBOXMASK.png"):
        return "visboxmask"
    if "@" in name and file_path.stem.split("@")[-1].isdigit():
        return "lesion_mask"
    return "other"

def resize_and_save(img_path: Path, out_path: Path, is_mask: bool):
    with Image.open(img_path) as img:
        if is_mask:
            img = img.resize(target_size, Image.Resampling.NEAREST)
        else:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        img.save(out_path, format="PNG")

for idx, case_dir in enumerate(sorted([p for p in root.iterdir() if p.is_dir()]), start=1):
    case_id = case_dir.name

    for img_path in sorted(case_dir.glob("*.png")):
        file_type = classify_file(img_path)

        if file_type == "other":
            counts["skipped"] += 1
            continue

        if file_type == "raw":
            out_path = raw_out / img_path.name
            is_mask = False
        elif file_type == "cropped":
            out_path = cropped_out / img_path.name
            is_mask = False
        elif file_type == "lesion_mask":
            out_path = mask_out / img_path.name
            is_mask = True
        elif file_type == "visboxmask":
            out_path = vis_out / img_path.name
            is_mask = True

        try:
            resize_and_save(img_path, out_path, is_mask=is_mask)
            counts[file_type] += 1

            rows.append({
                "case_id": case_id,
                "file_name": img_path.name,
                "image_type": file_type,
                "output_subfolder": out_path.parent.name,
                "output_file": out_path.name
            })

        except Exception as e:
            counts["skipped"] += 1
            print(f"Skipped {img_path}: {e}")

    if idx % 500 == 0:
        print(f"Processed {idx} case folders...")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["case_id", "file_name", "image_type", "output_subfolder", "output_file"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nDone.")
print(f"Raw images saved: {counts['raw']}")
print(f"Cropped images saved: {counts['cropped']}")
print(f"Lesion masks saved: {counts['lesion_mask']}")
print(f"VISBOXMASK images saved: {counts['visboxmask']}")
print(f"Skipped: {counts['skipped']}")
print(f"Manifest saved to: {csv_path}")
print(f"Output root: {output_root}")