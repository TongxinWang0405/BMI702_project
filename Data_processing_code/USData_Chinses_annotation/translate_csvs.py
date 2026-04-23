import csv
import time
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='zh-CN', target='en')

def translate_batch(texts):
    """Translate a list of unique texts, returning a dict {chinese: english}."""
    cache = {}
    for i, text in enumerate(texts):
        if not text:
            cache[text] = ''
            continue
        try:
            cache[text] = translator.translate(text)
        except Exception as e:
            print(f"  Error on item {i}: {e} — retrying once...")
            time.sleep(2)
            try:
                cache[text] = translator.translate(text)
            except Exception:
                cache[text] = text  # fallback: keep original
        if (i + 1) % 100 == 0:
            print(f"  Translated {i+1}/{len(texts)}")
            time.sleep(0.5)  # be polite to Google
    return cache

organs = [
    ('Liver',   'Liver',   'liver.csv'),
    ('Breast',  'Breast',  'breast.csv'),
    ('Thyroid', 'Thyroid', 'thyroid.csv'),
]

for organ_name, folder, fname in organs:
    path = f'/Users/wangtongxin/Desktop/BMI 702/Code/Data Prcessing/USData/{folder}/{fname}'
    print(f'\n=== {organ_name} ===')

    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))

    header = rows[0]   # ['image_path', 'annotation']
    data   = rows[1:]

    unique_texts = list(dict.fromkeys(r[1] for r in data))  # preserve order, dedupe
    print(f'  {len(data)} rows, {len(unique_texts)} unique annotations — translating...')

    cache = translate_batch(unique_texts)

    # Build new rows: image_path | annotation (EN) | annotation_Chinese
    new_rows = [[r[0], cache[r[1]], r[1]] for r in data]
    new_header = ['image_path', 'annotation', 'annotation_Chinese']

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    print(f'  Done. Sample EN: {new_rows[0][1]}')

print('\nAll done.')
