from pathlib import Path
from PIL import Image
from collections import Counter

root = Path("/Users/lanrr/Downloads/Resize_Data/Common Carotid Artery Ultrasound Images Resized")

for subfolder in ["US images", "Expert mask images"]:
    folder = root / subfolder
    files = sorted(folder.glob("*.png"))
    print(f"\n{subfolder}: {len(files)} files")

    size_counts = Counter()
    for f in files:
        with Image.open(f) as img:
            size_counts[img.size] += 1

    print("Image sizes found:")
    for size, count in sorted(size_counts.items()):
        print(f"  {size}: {count}")