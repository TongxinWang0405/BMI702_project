import os
import csv
import random

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "AULI_resized_images")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "captions.csv")

TISSUE = "liver"

CONDITION_MAP = {
    "liver_benign":    "benign",
    "liver_malignant": "malignant",
    "liver_normal":    "normal",
}

TEMPLATES = [
    "An ultrasound image of {Tissue} with {Condition} findings.",
    "A B-mode ultrasound of {Tissue}, consistent with {Condition}.",
    "Sonographic appearance of {Condition} {Tissue}.",
    "This ultrasound demonstrates {Tissue} exhibiting features of {Condition}.",
    "A clinical ultrasound scan of {Tissue}, indicative of {Condition} pathology.",
    "Echographic findings of {Tissue} showing {Condition} characteristics.",
    "This sonogram of {Tissue} is consistent with a {Condition} diagnosis.",
    "A diagnostic ultrasound image of {Tissue}, with imaging features suggestive of {Condition}.",
    "Ultrasound of {Tissue} presenting sonographic signs of {Condition}.",
    "A grayscale ultrasound demonstrating {Condition} changes in {Tissue}.",
]

random.seed(42)

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

rows = []
for fname in image_files:
    # Derive condition from prefix (e.g. "liver_benign_1.jpg" → "liver_benign")
    prefix = "_".join(fname.split("_")[:2])
    condition = CONDITION_MAP.get(prefix)
    if condition is None:
        print(f"Warning: unrecognized prefix for {fname}, skipping.")
        continue

    template = random.choice(TEMPLATES)
    caption = template.replace("{Tissue}", TISSUE).replace("{Condition}", condition)
    rows.append((fname, caption))

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "caption"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")
