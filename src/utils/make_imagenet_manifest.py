import argparse
import csv
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("train_dir", type=Path)
args = parser.parse_args()

root = args.train_dir.resolve(strict=True)
classes = sorted(
    path for path in root.iterdir()
    if path.is_dir() and re.fullmatch(r"n\d{8}", path.name)
)
if len(classes) != 1000:
    raise SystemExit(f"Expected 1,000 class folders in {root}, found {len(classes)}")

output_dir = Path("src/artifacts/data")
output_dir.mkdir(parents=True, exist_ok=True)
image_count = 0

with (
    (output_dir / "imagenet_train.csv").open("w", newline="") as image_file,
    (output_dir / "imagenet_classes.csv").open("w", newline="") as class_file,
):
    images = csv.writer(image_file)
    class_ids = csv.writer(class_file)
    images.writerow(["image_path", "label"])
    class_ids.writerow(["label", "synset"])

    for label, folder in enumerate(classes):
        paths = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if not paths:
            raise SystemExit(f"No images found in {folder}")
        class_ids.writerow([label, folder.name])
        images.writerows((path, label) for path in paths)
        image_count += len(paths)

print(f"Wrote {image_count:,} images and {len(classes):,} classes to {output_dir}")