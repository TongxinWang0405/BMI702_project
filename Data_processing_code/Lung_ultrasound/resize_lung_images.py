from PIL import Image
import os

BASE_DIR = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/combined"
OUTPUT_DIR = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/resized"
TARGET_SIZE = 256

CLASSES = ["covid", "healthy", "other"]

for cls in CLASSES:
    src_dir = os.path.join(BASE_DIR, cls)
    dst_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)

    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Processing {cls}: {len(image_files)} images")

    for fname in image_files:
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")

        # Resize so longest side = 256, preserving aspect ratio
        w, h = img.size
        scale = TARGET_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Pad with black to make it 256x256
        padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        w, h = img.size
        padded.paste(img, ((TARGET_SIZE - w) // 2, (TARGET_SIZE - h) // 2))

        padded.save(os.path.join(dst_dir, fname))

    print(f"  Saved to {dst_dir}")

print("Done.")
