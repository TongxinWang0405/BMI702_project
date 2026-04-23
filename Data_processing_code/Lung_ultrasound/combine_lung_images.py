import os
import shutil

BASE_DIR = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/dataset"
OUTPUT_DIR = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/combined"

SPLITS = ["train", "test", "validation"]
CLASSES = ["covid", "healthy", "other"]

for cls in CLASSES:
    dst_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)

    for split in SPLITS:
        src_dir = os.path.join(BASE_DIR, split, cls)
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)

    count = len(os.listdir(dst_dir))
    print(f"{cls}: {count} images → {dst_dir}")

print("Done.")
