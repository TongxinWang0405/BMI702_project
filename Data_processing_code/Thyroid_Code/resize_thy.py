from pathlib import Path
from PIL import Image


input_root = Path("/Users/lanrr/Downloads/Resize_Data/dataset thyroid")


output_root = Path("/Users/lanrr/Downloads/Resize_Data/dataset_thyroid_resized_png")


target_size = (256, 256)

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

count = 0
skipped = 0

for img_path in input_root.rglob("*"):
    if not img_path.is_file():
        continue
    if img_path.suffix.lower() not in valid_exts:
        continue

    try:
        rel_path = img_path.relative_to(input_root)
        out_subdir = output_root / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_path = out_subdir / f"{img_path.stem}.png"

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img.save(out_path, format="PNG")

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")

    except Exception as e:
        skipped += 1
        print(f"Skipped {img_path}: {e}")

print(f"\nDone. Saved {count} resized PNG images.")
print(f"Skipped: {skipped}")
print(f"Output folder: {output_root}")