from pathlib import Path
import json
import csv

json_path = Path("/Users/lanrr/Downloads/BUSCoT/DatasetFiles/lesion_dataset.json")
output_csv = Path("/Users/lanrr/Downloads/BUSCoT_preprocessed/BUS-Lesion/bus_lesion_captions.csv")

def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()

def get_subtype_text(subtype_dict):
    if not subtype_dict:
        return ""
    first_key = next(iter(subtype_dict.keys()), "")
    if not first_key:
        return ""
    item = subtype_dict[first_key]
    histology = clean_text(item.get("histology", ""))
    return histology if histology else first_key

def build_metadata_caption(record):
    pathology = clean_text(record.get("pathology_histology", {}).get("pathology", ""))
    subtype = get_subtype_text(record.get("pathology_histology", {}).get("subtype", {}))

    us_report = record.get("us_report", {})
    birads = clean_text(us_report.get("BIRADS", ""))
    boundary = clean_text(us_report.get("LesionBoundary", ""))
    edge = clean_text(us_report.get("LesionEdge", ""))
    calcification = clean_text(us_report.get("LesionCalcificationFeatures", ""))
    echo = clean_text(us_report.get("EchoCharacteristics", ""))

    pieces = []

    if pathology:
        pieces.append(f"Breast ultrasound image of a {pathology.lower()} lesion.")
    else:
        pieces.append("Breast ultrasound image of a lesion.")

    if subtype:
        pieces.append(f"Histology subtype: {subtype}.")

    if birads:
        pieces.append(f"BIRADS {birads}.")

    if boundary:
        pieces.append(f"Boundary: {boundary}.")

    if edge:
        pieces.append(f"Edge: {edge}.")

    if echo:
        pieces.append(f"Echo pattern: {echo}.")

    if calcification:
        pieces.append(f"Calcification: {calcification}.")

    return " ".join(pieces)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for _, record in data.items():
    image_path = clean_text(record.get("image_path", ""))
    if not image_path:
        continue

    # convert BUS-Lesion/trainval/xxx.png -> trainval/xxx.png
    if image_path.startswith("BUS-Lesion/"):
        file_name = image_path[len("BUS-Lesion/"):]
    else:
        file_name = image_path

    pathology = clean_text(record.get("pathology_histology", {}).get("pathology", ""))

    caption_info = record.get("caption", {})
    full_caption = clean_text(caption_info.get("image_full_caption", ""))
    relative_caption = clean_text(caption_info.get("image_relative_caption", ""))
    reasoning = clean_text(record.get("us_report", {}).get("reasoning_response", ""))

    if full_caption:
        caption = full_caption
        caption_type = "full_caption"
    elif relative_caption:
        caption = relative_caption
        caption_type = "relative_caption"
    elif reasoning:
        caption = reasoning
        caption_type = "reasoning"
    else:
        caption = build_metadata_caption(record)
        caption_type = "metadata_template"

    rows.append({
        "file_name": file_name,
        "caption": caption,
        "Label": pathology,
        "caption_type": caption_type
    })

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file_name", "caption", "Label", "caption_type"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved CSV to: {output_csv}")
print(f"Total rows: {len(rows)}")

from collections import Counter
counts = Counter(row["caption_type"] for row in rows)
print("Caption type counts:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")

print("\nFirst 5 rows:")
for row in rows[:5]:
    print(row)