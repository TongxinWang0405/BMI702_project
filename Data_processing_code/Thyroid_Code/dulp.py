from pathlib import Path
from collections import Counter

root = Path("/Users/lanrr/Downloads/Resize_Data/dataset_thyroid_resized_png")

names = [p.name for p in root.rglob("*.png")]
counts = Counter(names)
dups = {name: c for name, c in counts.items() if c > 1}

print(f"Total PNG files: {len(names)}")
print(f"Unique PNG filenames: {len(counts)}")
print(f"Duplicate filenames: {len(dups)}")

if dups:
    print("\nExample duplicates:")
    for i, (name, c) in enumerate(sorted(dups.items())[:20], 1):
        print(f"{i}. {name}: {c}")