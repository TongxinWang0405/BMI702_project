from PIL import Image
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "AULI")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "resized_images")
TARGET_SIZE = (256, 256)

CATEGORIES = {
    "Benign":    "liver_benign",
    "Malignant": "liver_malignant",
    "Normal":    "liver_normal",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for folder, prefix in CATEGORIES.items():
    src_dir = os.path.join(BASE_DIR, folder, "image")
    if not os.path.isdir(src_dir):
        print(f"Skipping {folder}: folder not found at {src_dir}")
        continue

    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Processing {folder}: {len(image_files)} images")

    for fname in image_files:
        num = os.path.splitext(fname)[0]          # e.g. "100" from "100.jpg"
        ext = os.path.splitext(fname)[1].lower()  # e.g. ".jpg"

        src_path = os.path.join(src_dir, fname)
        dst_name = f"{prefix}_{num}{ext}"         # e.g. "liver_benign_100.jpg"
        dst_path = os.path.join(OUTPUT_DIR, dst_name)

        img = Image.open(src_path).convert("RGB")

        # Resize so longest side = 256, preserving aspect ratio
        w, h = img.size
        scale = 256 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Pad with black to make it 256x256
        padded = Image.new("RGB", (256, 256), (0, 0, 0))
        w, h = img.size
        padded.paste(img, ((256 - w) // 2, (256 - h) // 2))

        padded.save(dst_path)

    print(f"  Saved to {OUTPUT_DIR}")

print("Done.")
