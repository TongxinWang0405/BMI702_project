from PIL import Image
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = [
    ('Liver',   'liver.csv'),
    ('Breast',  'breast.csv'),
    ('Thyroid', 'thyroid.csv'),
]

for folder, csv_file in FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg'))]
    print(f'{folder}: converting {len(image_files)} images...', flush=True)

    for i, fname in enumerate(image_files):
        src = os.path.join(folder_path, fname)
        dst = os.path.join(folder_path, os.path.splitext(fname)[0] + '.png')
        Image.open(src).save(dst, 'PNG')
        os.remove(src)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(image_files)} done...', flush=True)

    print(f'  {folder} images done.', flush=True)

    # Update image_path in CSV
    csv_path = os.path.join(folder_path, csv_file)
    with open(csv_path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))

    header = rows[0]
    data   = rows[1:]
    new_rows = [[os.path.splitext(r[0])[0] + '.png'] + r[1:] for r in data]

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_rows)

    print(f'  {csv_file} updated.', flush=True)

print('All done.')
