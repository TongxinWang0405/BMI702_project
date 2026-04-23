from pathlib import Path
from PIL import Image


root = Path("/Users/lanrr/Downloads/Resize_Data/Common Carotid Artery Ultrasound Images")


us_input = root / "US images"
mask_input = root / "Expert mask images"


output_root = Path("/Users/lanrr/Downloads/Resize_Data/Common Carotid Artery Ultrasound Images Resized")
us_output = output_root / "US images"
mask_output = output_root / "Expert mask images"


target_size = (256, 256)

def resize_and_save(input_dir: Path, output_dir: Path, is_mask: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_path in sorted(input_dir.glob("*.png")):
        out_path = output_dir / img_path.name

        try:
            with Image.open(img_path) as img:
                if is_mask:
                    img = img.resize(target_size, Image.Resampling.NEAREST)
                else:
                    img = img.convert("RGB")
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

                img.save(out_path, format="PNG")

            count += 1
            if count % 100 == 0:
                print(f"{input_dir.name}: processed {count} images")

        except Exception as e:
            print(f"Skipped {img_path.name}: {e}")

    print(f"Finished {input_dir.name}. Saved {count} images to {output_dir}")

print("Resizing ultrasound images...")
resize_and_save(us_input, us_output, is_mask=False)

print("\nResizing expert mask images...")
resize_and_save(mask_input, mask_output, is_mask=True)

print(f"\nAll done. Output folder: {output_root}")