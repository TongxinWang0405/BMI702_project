from pathlib import Path
import csv
import random

random.seed(42)

input_root = Path("/Users/lanrr/Downloads/Resize_Data/dataset_thyroid_resized_png")
output_csv = Path("/Users/lanrr/Downloads/Resize_Data/thyroid_template_captions_with_paths_v2.csv")

templates = [
    "An ultrasound image of the {tissue} with {condition} findings.",
    "A B-mode ultrasound of the {tissue}, labeled {condition}.",
    "Sonographic appearance of {condition} {tissue}.",
    "This ultrasound image shows the {tissue} with a {condition} label.",
    "A clinical ultrasound scan of the {tissue}, categorized as {condition}.",
    "Echographic appearance of the {tissue} with {condition} features.",
    "This sonogram of the {tissue} is labeled {condition}.",
    "A diagnostic ultrasound image of the {tissue}, tagged as {condition}.",
    "Ultrasound of the {tissue} with {condition} appearance.",
    "A grayscale ultrasound image demonstrating {condition} changes in the {tissue}.",
]

def get_label(path_str: str) -> str:
    s = path_str.lower()
    if "normal thyroid" in s:
        return "normal"
    if "malignant" in s:
        return "malignant"
    if "benign" in s:
        return "benign"
    return "unknown"

def make_caption(tissue: str, condition: str) -> str:
    template = random.choice(templates)
    return template.format(tissue=tissue, condition=condition)

rows_out = []

for img_path in sorted(input_root.rglob("*.png")):
    rel_path = img_path.relative_to(input_root).as_posix()
    label = get_label(rel_path)
    caption = make_caption("thyroid", label)

    rows_out.append({
        "file_name": rel_path,
        "caption": caption,
        "Label": label,
        "caption_type": "tissue+condition"
    })

with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file_name", "caption", "Label", "caption_type"]
    )
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Saved CSV to: {output_csv}")
print(f"Total rows: {len(rows_out)}")
print("\nFirst 10 rows:")
for row in rows_out[:10]:
    print(row)