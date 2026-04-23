import csv
import time
import sys
import signal
from deep_translator import GoogleTranslator

def translate_with_timeout(text, timeout=10):
    """Translate with a timeout to avoid hanging on a single request."""
    result = [None]
    error  = [None]

    def _do():
        try:
            t = GoogleTranslator(source='zh-CN', target='en')
            result[0] = t.translate(text)
        except Exception as e:
            error[0] = e

    import threading
    thread = threading.Thread(target=_do)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None, TimeoutError(f"Translation timed out after {timeout}s")
    return result[0], error[0]

organs = [
    ('Breast',  'Breast',  'breast.csv'),
    ('Thyroid', 'Thyroid', 'thyroid.csv'),
]

for organ_name, folder, fname in organs:
    path = f'/Users/wangtongxin/Desktop/BMI 702/Code/Data Prcessing/USData/{folder}/{fname}'
    print(f'\n=== {organ_name} ===', flush=True)

    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))

    header = rows[0]
    data   = rows[1:]

    if len(header) != 2:
        print(f'  Already translated, skipping.', flush=True)
        continue

    unique_texts = list(dict.fromkeys(r[1] for r in data))
    print(f'  {len(data)} rows, {len(unique_texts)} unique annotations', flush=True)

    cache = {}
    for i, text in enumerate(unique_texts):
        if not text:
            cache[text] = ''
            continue
        translated, err = translate_with_timeout(text, timeout=10)
        if err or translated is None:
            # retry once
            print(f'  Warning item {i}: {err} — retrying...', flush=True)
            time.sleep(2)
            translated, err = translate_with_timeout(text, timeout=15)
            if translated is None:
                print(f'  Skipping item {i}, keeping original.', flush=True)
                translated = text
        cache[text] = translated
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(unique_texts)} translated...', flush=True)
        time.sleep(0.05)

    print(f'  Writing {len(data)} rows...', flush=True)
    new_rows = [[r[0], cache[r[1]], r[1]] for r in data]
    new_header = ['image_path', 'annotation', 'annotation_Chinese']

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    print(f'  Done. Sample: {new_rows[0][1]}', flush=True)

print('\nAll done.', flush=True)
