import os
import re
import csv
import random
import openpyxl

RESIZED_DIR   = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/resized"
METADATA_PATH = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/Lung Ultrasound/metadata.xlsx"
OUTPUT_CSV    = "/Users/tongxinsimac/Desktop/BMI 702/Code/Data Prcessing/lung_captions.csv"

CLASSES = ["covid", "healthy", "other"]

FOLDER_TO_CONDITION = {
    "covid":   "probably Covid",
    "healthy": "Healthy Lung",
    "other":   "diseased but probably not Covid",
}

# ── Location parser ───────────────────────────────────────────────────────────
def parse_location(fname):
    """Return a region string like 'upper posterior longitudinal view of the left lung',
    or None if location cannot be determined from the filename."""
    name = os.path.splitext(fname)[0].upper()
    # Strip the time suffix (-HH_MM_SS) and any trailing noise
    m = re.match(r'^(.+?)-\d+_\d+_\d+', name)
    prefix = m.group(1) if m else name
    tokens = re.split(r'[_\s]+', prefix)

    # Side
    side = None
    if any(t in ("LEFT", "LT") for t in tokens):
        side = "left"
    elif any(t in ("RT", "RIGHT") for t in tokens):
        side = "right"
    if side is None:
        return None                         # no side info → skip

    # Position (check substrings to handle typos/concat like LATUPPER)
    joined = " ".join(tokens)
    if any(t in ("ANT", "ANTERIOR") for t in tokens):
        position = "anterior"
    elif any(t in ("POST", "POSTERIOR") for t in tokens) or "LOWRE" in joined:
        position = "posterior"
    elif any("LAT" in t for t in tokens) or "LATERAL" in joined:
        position = "lateral"
    else:
        position = None

    # Level
    if any("UPPER" in t for t in tokens):
        level = "upper"
    elif any(t in ("LOWER", "LOWRE") for t in tokens):
        level = "lower"
    else:
        level = None

    # Orientation
    if any(t in ("LONG", "LONGITUDINAL") for t in tokens):
        orientation = "longitudinal"
    elif any(t in ("TRANS", "TRANSVERSE", "TYRANS", "TRASN") for t in tokens
             ) or "TRANSVERSE" in joined:
        orientation = "transverse"
    else:
        orientation = None

    parts = [p for p in [level, position, orientation] if p]
    region = " ".join(parts) if parts else ""
    return f"{region} view of the {side} lung".strip() if region else f"{side} lung"


# ── Zone column indices → region label (cols 10-98 = right lung, 106-194 = left lung)
IMG_COLS = [
    (10,  "upper posterior longitudinal view of the right lung"),
    (18,  "upper posterior transverse view of the right lung"),
    (26,  "lower posterior longitudinal view of the right lung"),
    (34,  "lower posterior transverse view of the right lung"),
    (42,  "upper anterior longitudinal view of the right lung"),
    (50,  "upper anterior transverse view of the right lung"),
    (58,  "lower anterior longitudinal view of the right lung"),
    (66,  "lower anterior transverse view of the right lung"),
    (74,  "upper lateral longitudinal view of the right lung"),
    (82,  "upper lateral transverse view of the right lung"),
    (90,  "lower lateral longitudinal view of the right lung"),
    (98,  "lower lateral transverse view of the right lung"),
    (106, "upper posterior longitudinal view of the left lung"),
    (114, "upper posterior transverse view of the left lung"),
    (122, "lower posterior longitudinal view of the left lung"),
    (130, "lower posterior transverse view of the left lung"),
    (138, "upper anterior longitudinal view of the left lung"),
    (146, "upper anterior transverse view of the left lung"),
    (154, "lower anterior longitudinal view of the left lung"),
    (162, "lower anterior transverse view of the left lung"),
    (170, "upper lateral longitudinal view of the left lung"),
    (178, "upper lateral transverse view of the left lung"),
    (186, "lower lateral longitudinal view of the left lung"),
    (194, "lower lateral transverse view of the left lung"),
]

METATEXT_TEMPLATES = [
    "Ultrasound of {Region} in {PatientInfo}. {Findings}. Assessment: {Condition}.",
    "Sonographic findings in {Region}: {Findings}. {PatientInfo}. Diagnosis: {Condition}.",
    "An ultrasound image of {Region} consistent with {Condition}. {Findings}.",
    "{Region} evaluated by sonography. {Findings}. Conclusion: {Condition}.",
    "{PatientInfo} underwent ultrasound imaging. {Findings} identified in {Region}, suggestive of {Condition}.",
    "Sonography: {Region}, {PatientInfo}. {Findings}. {Condition}.",
    "{PatientInfo}. Ultrasound of {Region} reveals {Findings}, consistent with {Condition}.",
    "{Condition} pattern on ultrasound. {Region} demonstrates {Findings}. {PatientInfo}.",
    "Sonographic examination of {Region}. {Findings}. {Condition}.",
    "Ultrasound performed on {PatientInfo}. Examination of {Region} revealed {Findings}. Impression: {Condition}.",
]

TIER2_TEMPLATES = [
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

# ── Build lookup: filename → metadata dict ────────────────────────────────────
wb = openpyxl.load_workbook(METADATA_PATH)
ws = wb.active

filename_to_meta = {}

for row in ws.iter_rows(min_row=2, values_only=True):
    age      = row[4]
    gender   = str(row[5]).lower() if row[5] else None
    pregnant = row[6]

    if age and gender:
        pat = f"a {int(age)}-year-old {gender} patient"
        if pregnant and str(pregnant).strip().lower() == "yes":
            pat = f"a {int(age)}-year-old pregnant {gender} patient"
    else:
        pat = "a patient"

    conclusion = row[202]
    condition  = str(conclusion).strip() if conclusion else "unspecified condition"

    for col_idx, region_label in IMG_COLS:
        fname = row[col_idx]
        if not fname:
            continue
        fname = str(fname).strip()

        findings_raw = row[col_idx + 1]
        findings = (f"{str(findings_raw).strip()} findings"
                    if findings_raw else "unremarkable findings")

        filename_to_meta[fname] = {
            "PatientInfo": pat,
            "Region":      region_label,
            "Findings":    findings,
            "Condition":   condition,
        }

print(f"Metadata lookup built: {len(filename_to_meta)} entries")

# ── Generate captions ─────────────────────────────────────────────────────────
rows = []
matched = unmatched_with_loc = unmatched_no_loc = 0

for cls in CLASSES:
    src_dir     = os.path.join(RESIZED_DIR, cls)
    image_files = sorted(f for f in os.listdir(src_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png")))

    for fname in image_files:
        meta = filename_to_meta.get(fname)

        if meta:
            matched += 1
            template = random.choice(METATEXT_TEMPLATES)
            caption  = (template
                        .replace("{PatientInfo}", meta["PatientInfo"])
                        .replace("{Region}",      meta["Region"])
                        .replace("{Findings}",    meta["Findings"])
                        .replace("{Condition}",   meta["Condition"]))
        else:
            condition = FOLDER_TO_CONDITION[cls]
            location  = parse_location(fname)
            tissue    = location if location else "lung"

            if location:
                unmatched_with_loc += 1
            else:
                unmatched_no_loc += 1

            template = random.choice(TIER2_TEMPLATES)
            caption  = (template
                        .replace("{Tissue}",    tissue)
                        .replace("{Condition}", condition))

        rows.append((fname, caption))

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "caption"])
    writer.writerows(rows)

print(f"Matched to metadata:          {matched}")
print(f"Unmatched — location parsed:  {unmatched_with_loc}")
print(f"Unmatched — no location:      {unmatched_no_loc}")
print(f"Total rows saved: {len(rows)} → {OUTPUT_CSV}")
