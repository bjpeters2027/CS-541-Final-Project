"""Index ImageNet images and class names directly out of the Kaggle competition
zip, without ever calling extractall. Turing's guidance is to read members
straight out of the archive with zipfile instead of unpacking ~1.3M files onto
the shared filesystem -- this script (and dataprep.archive_dataset at load time)
is how that's honored end to end.

Expects the zip downloaded by scripts/01_fetch_imagenet.py to contain, at top
level: LOC_synset_mapping.txt, LOC_val_solution.csv, and
ILSVRC/Data/CLS-LOC/{train,val}/... (the Kaggle "imagenet-object-localization-
challenge" layout).
"""
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGENET_RAW_DIR, MANIFESTS_DIR

TRAIN_PREFIX = "ILSVRC/Data/CLS-LOC/train/"
VAL_PREFIX = "ILSVRC/Data/CLS-LOC/val/"
SYNSET_MAPPING_NAME = "LOC_synset_mapping.txt"
VAL_SOLUTION_NAME = "LOC_val_solution.csv"

TRAIN_MEMBER_RE = re.compile(re.escape(TRAIN_PREFIX) + r"(n\d+)/[^/]+\.JPEG$")


def find_archive() -> Path:
    zips = sorted(IMAGENET_RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No .zip found in {IMAGENET_RAW_DIR}. Run scripts/01_fetch_imagenet.py first."
        )
    return zips[0]


def load_synset_mapping(archive: zipfile.ZipFile) -> dict[str, str]:
    mapping = {}
    with archive.open(SYNSET_MAPPING_NAME) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            wnid, _, names = line.strip().partition(" ")
            if wnid:
                mapping[wnid] = names.split(",")[0].strip()
    return mapping


def load_val_labels(archive: zipfile.ZipFile) -> dict[str, str]:
    labels = {}
    with archive.open(VAL_SOLUTION_NAME) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
        for row in reader:
            labels[row["ImageId"]] = row["PredictionString"].split(" ")[0]
    return labels


def main() -> None:
    archive_path = find_archive()
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        synset_to_name = load_synset_mapping(archive)
        val_labels = load_val_labels(archive)

        for name in archive.namelist():
            if name.startswith(TRAIN_PREFIX) and name.endswith(".JPEG"):
                match = TRAIN_MEMBER_RE.match(name)
                if not match:
                    continue
                rows.append({
                    "image_id": Path(name).stem,
                    "split": "train",
                    "wnid": match.group(1),
                    "archive_path": str(archive_path),
                    "member_path": name,
                })
            elif name.startswith(VAL_PREFIX) and name.endswith(".JPEG"):
                image_id = Path(name).stem
                wnid = val_labels.get(image_id)
                if wnid is None:
                    continue
                rows.append({
                    "image_id": image_id,
                    "split": "val",
                    "wnid": wnid,
                    "archive_path": str(archive_path),
                    "member_path": name,
                })

    if not rows:
        raise RuntimeError(
            f"Found 0 images in {archive_path}. The zip's internal layout may not "
            "match the expected Kaggle structure -- check with `unzip -l` (don't extract)."
        )

    manifest = pd.DataFrame(rows)
    manifest["class_name"] = manifest["wnid"].map(synset_to_name)

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFESTS_DIR / "imagenet.parquet", index=False)

    classes = (
        manifest[["wnid", "class_name"]]
        .drop_duplicates()
        .sort_values("wnid")
        .reset_index(drop=True)
    )
    classes.to_parquet(MANIFESTS_DIR / "imagenet_classes.parquet", index=False)

    print(f"Indexed {len(manifest)} images across {len(classes)} classes -> {MANIFESTS_DIR}")
    unresolved = manifest["class_name"].isna().sum()
    if unresolved:
        print(f"warning: {unresolved} images have a wnid with no resolved class name")


if __name__ == "__main__":
    main()
