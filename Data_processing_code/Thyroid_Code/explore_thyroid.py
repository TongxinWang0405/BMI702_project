from pathlib import Path
from collections import Counter, defaultdict

root = Path("/Users/lanrr/Downloads/Resize_Data/dataset thyroid")

print(f"Dataset root: {root}\n")

if not root.exists():
    print("Root folder does not exist.")
    raise SystemExit

print("Top-level contents:")
for p in sorted(root.iterdir()):
    kind = "DIR " if p.is_dir() else "FILE"
    print(f"  {kind} {p.name}")

print("\nDetailed folder summary:")

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
all_files = [p for p in root.rglob("*") if p.is_file()]
image_files = [p for p in all_files if p.suffix.lower() in valid_exts]

print(f"Total files: {len(all_files)}")
print(f"Total image files: {len(image_files)}")

ext_counts = Counter(p.suffix.lower() for p in image_files)
print(f"Image extensions: {dict(ext_counts)}\n")

level1_dirs = [p for p in root.iterdir() if p.is_dir()]
for split_dir in sorted(level1_dirs):
    print(f"{split_dir.name}")
    split_files = [p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
    print(f"  Total image files: {len(split_files)}")

    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    for class_dir in sorted(class_dirs):
        class_files = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
        print(f"  {class_dir.name}: {len(class_files)} images")

        nested_dirs = [p for p in class_dir.rglob("*") if p.is_dir()]
        if nested_dirs:
            print(f"    Nested subfolders: {len(nested_dirs)}")
            for nd in nested_dirs[:5]:
                print(f"      {nd.relative_to(root)}")

        print("    Sample files:")
        for f in class_files[:5]:
            print(f"      {f.relative_to(root)}")
    print()