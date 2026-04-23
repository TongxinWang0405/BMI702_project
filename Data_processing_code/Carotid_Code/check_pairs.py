from pathlib import Path

root = Path("/Users/lanrr/Downloads/Resize_Data/Common Carotid Artery Ultrasound Images")
us_dir = root / "US images"
mask_dir = root / "Expert mask images"

us_files = {p.name for p in us_dir.glob("*.png")}
mask_files = {p.name for p in mask_dir.glob("*.png")}

only_in_us = sorted(us_files - mask_files)
only_in_mask = sorted(mask_files - us_files)

print(f"US images: {len(us_files)}")
print(f"Mask images: {len(mask_files)}")
print(f"Matched filenames: {len(us_files & mask_files)}")
print(f"Only in US: {len(only_in_us)}")
print(f"Only in masks: {len(only_in_mask)}")

if only_in_us[:10]:
    print("\nExamples only in US:")
    for x in only_in_us[:10]:
        print(x)

if only_in_mask[:10]:
    print("\nExamples only in masks:")
    for x in only_in_mask[:10]:
        print(x)