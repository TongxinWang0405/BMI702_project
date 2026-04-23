from pathlib import Path
import json
import csv

json_path = Path("/Users/lanrr/Downloads/BUSCoT/DatasetFiles/BUS-Expert_dataset.json")
image_root = Path("/Users/lanrr/Downloads/BUSCoT_preprocessed/BUS-Expert/cropped_images")
output_csv = Path("/Users/lanrr/Downloads/BUSCoT_preprocessed/BUS-Expert/bus_expert_captions.csv")

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

def extract_us_report_fields(us_report):
    # BUS-Expert stores lesion reports under keys like "0"
    if not isinstance(us_report, dict) or not us_report:
        return {}

    first_key = next(iter(us_report.keys()), None)
    if first_key is None:
        return {}

    if isinstance(us_report[first_key], dict):
        return us_report[first_key]

    return us_report

def build_metadata_caption(record):
    pathology = clean_text(record.get("pathology_histology", {}).get("pathology", ""))
    subtype = get_subtype_text(record.get("pathology_histology", {}).get("subtype", {}))

    report = extract_us_report_fields(record.get("us_report", {}))
    birads = clean_text(report.get("BIRADS", ""))
    boundary = clean_text(report.get("LesionBoundary", ""))
    edge = clean_text(report.get("LesionEdge", ""))
    calcification = clean_text(report.get("LesionCalcificationFeatures", ""))
    echo = clean_text(report.get("EchoCharacteristics", ""))

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

for case_id, record in data.items():
    cropped_rel = clean_text(record.get("image_file", {}).get("cropped_image", ""))
    if not cropped_rel:
        continue

    # from "000000/000000@cropped.png" -> "000000@cropped.png"
    file_name = Path(cropped_rel).name

    # only keep rows whose resized cropped image actually exists
    if not (image_root / file_name).exists():
        continue

    pathology = clean_text(record.get("pathology_histology", {}).get("pathology", ""))

    caption_info = record.get("caption", {})
    full_caption = clean_text(caption_info.get("image_full_caption", ""))
    relative_caption = clean_text(caption_info.get("image_relative_caption", ""))
    reasoning = clean_text(extract_us_report_fields(record.get("us_report", {})).get("reasoning_response", ""))

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