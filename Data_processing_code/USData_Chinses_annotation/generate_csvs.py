import json
import csv
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

organs = [
    ("Liver",   "Liver",  "new_Liver2.json",   "liver.csv"),
    ("Breast",  "Breast", "new_Mammary2.json",  "breast.csv"),
    ("Thyroid", "Thyroid","new_Thyroid2.json",  "thyroid.csv"),
]

for organ_name, img_folder, json_file, csv_file in organs:
    json_path = os.path.join(base_dir, json_file)
    # CSV goes inside the organ folder; image_path is just the filename
    csv_path  = os.path.join(base_dir, img_folder, csv_file)

    with open(json_path, encoding='utf-8-sig') as f:
        data = json.load(f)

    rows = []
    for split in ('train', 'val', 'test'):
        for entry in data.get(split, []):
            finding = entry['finding']
            for img_name in entry['image_path']:
                rows.append((img_name, finding))

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'annotation'])
        writer.writerows(rows)

    print(f"{organ_name}: {len(rows)} rows -> {img_folder}/{csv_file}")
