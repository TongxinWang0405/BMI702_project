import os
import csv
import shutil
from PIL import Image

RESIZED_DIR  = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/resized"
COMBINED_DIR = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/final"
INPUT_CSV    = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/lung_captions.csv"
OUTPUT_CSV   = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/lung_captions.csv"

CLASSES = ["covid", "healthy", "other"]

os.makedirs(COMBINED_DIR, exist_ok=True)

# ── Build old→new filename mapping and save renamed PNGs ─────────────────────
old_to_new = {}

for cls in CLASSES:
    src_dir     = os.path.join(RESIZED_DIR, cls)
    image_files = sorted(f for f in os.listdir(src_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png")))

    for i, fname in enumerate(image_files, start=1):
        new_name = f"lung_{cls}_{i}.png"
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(COMBINED_DIR, new_name)

        img = Image.open(src_path).convert("RGB")
        img.save(dst_path, format="PNG")

        old_to_new[fname] = new_name

    print(f"{cls}: {len(image_files)} images renamed and saved as PNG")

# ── Update CSV ────────────────────────────────────────────────────────────────
with open(INPUT_CSV, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows   = list(reader)

updated = 0
for row in rows:
    new_name = old_to_new.get(row[0])
    if new_name:
        row[0] = new_name
        updated += 1

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nCSV updated: {updated}/{len(rows)} filenames renamed")
print(f"All images saved to: {COMBINED_DIR}")
print(f"CSV saved to:        {OUTPUT_CSV}")
