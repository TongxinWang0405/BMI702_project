from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SIZE = 256
FOLDERS = ['Liver', 'Breast', 'Thyroid']

for folder in FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'{folder}: {len(image_files)} images', flush=True)

    for i, fname in enumerate(image_files):
        fpath = os.path.join(folder_path, fname)
        img = Image.open(fpath).convert('RGB')

        # Resize so longest side = 256, preserving aspect ratio
        w, h = img.size
        scale = TARGET_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Pad with black to make it 256x256
        padded = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        w, h = img.size
        padded.paste(img, ((TARGET_SIZE - w) // 2, (TARGET_SIZE - h) // 2))

        padded.save(fpath)

        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(image_files)} done...', flush=True)

    print(f'  {folder} complete.', flush=True)

print('All done.')
