"""Stage ImageNet validation images as a small training/evaluation subset zip.

Requires access to ILSVRC/imagenet-1k on Hugging Face and HF_TOKEN or a saved
Hugging Face login. Only validation shards are downloaded. The resulting zip
keeps the layout expected by scripts/02_build_imagenet_manifest.py.

Run this from a login-node shell, not inside an sbatch job -- compute nodes on
many HPC clusters have no outbound internet access.
"""
import os
import sys
from pathlib import Path
import csv
import io
import random
import zipfile
from collections import defaultdict

import requests
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGENET_RAW_DIR, RANDOM_SEED

SEEN_CLASSES = 800
IMAGES_PER_CLASS = 30


def main() -> None:
    IMAGENET_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if any(IMAGENET_RAW_DIR.glob("*.zip")):
        print(f"Found an existing zip in {IMAGENET_RAW_DIR}, skipping download.")
        return

    if not 1 <= IMAGES_PER_CLASS < 40:
        raise ValueError("IMAGES_PER_CLASS must leave room for validation and test images (1-39).")
    snapshot = snapshot_download(
        "ILSVRC/imagenet-1k", repo_type="dataset",
        allow_patterns="data/validation-*.parquet",
        local_dir=IMAGENET_RAW_DIR / "validation_shards",
        token=os.environ.get("HF_TOKEN"), max_workers=2,
    )
    shards = sorted(Path(snapshot).glob("data/validation-*.parquet"))
    if not shards:
        raise RuntimeError("No ImageNet validation shards found.")
    archive_path = IMAGENET_RAW_DIR / "imagenet_validation_subset.partial"
    with zipfile.ZipFile(archive_path, "w") as archive:
        metadata_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/"
        mapping = requests.get(metadata_url + "imagenet_metadata.txt", timeout=30)
        mapping.raise_for_status()
        synsets = requests.get(metadata_url + "imagenet_lsvrc_2015_synsets.txt", timeout=30)
        synsets.raise_for_status()
        descriptions = dict(line.split("\t", 1) for line in mapping.content.decode().splitlines())
        archive.writestr("LOC_synset_mapping.txt", "".join(
            f"{class_id} {descriptions[class_id]}\n" for class_id in synsets.content.decode().splitlines()
        ))
        classes = sorted(line.split()[0] for line in archive.read("LOC_synset_mapping.txt").decode().splitlines() if line.strip())
        rng = random.Random(RANDOM_SEED)
        seen_classes = classes.copy()
        rng.shuffle(seen_classes)
        seen = set(seen_classes[:SEEN_CLASSES])
        roles = {}
        for class_id in classes:
            roles[class_id] = (
                ["train"] * IMAGES_PER_CLASS + ["validation"] * 10
                + ["test_seen"] * (40 - IMAGES_PER_CLASS)
            ) if class_id in seen else ["test_unseen"] * 50
            rng.shuffle(roles[class_id])
        val_rows, split_rows = io.StringIO(), io.StringIO()
        val_writer, split_writer = csv.writer(val_rows), csv.writer(split_rows)
        val_writer.writerow(["ImageId", "PredictionString"])
        split_writer.writerow(["ImageId", "partition"])
        counts = defaultdict(int)
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["image", "label"]):
                for row in batch.to_pylist():
                    class_id = classes[row["label"]]
                    index = counts[class_id]
                    partition = roles[class_id][index]
                    image_id = Path(row["image"]["path"] or f"{class_id}_{index:02d}.JPEG").stem
                    member = f"ILSVRC/Data/CLS-LOC/train/{class_id}/{image_id}.JPEG" if partition == "train" else f"ILSVRC/Data/CLS-LOC/val/{image_id}.JPEG"
                    archive.writestr(member, row["image"]["bytes"])
                    if partition != "train":
                        val_writer.writerow([image_id, class_id])
                    split_writer.writerow([image_id, partition])
                    counts[class_id] += 1
            print(f"Packed {shard.name}", flush=True)
        if any(counts[class_id] != 50 for class_id in classes):
            raise ValueError("Expected 50 validation-source images per ImageNet class.")
        archive.writestr("LOC_val_solution.csv", val_rows.getvalue())
        archive.writestr("image_partitions.csv", split_rows.getvalue())
    archive_path.replace(IMAGENET_RAW_DIR / "imagenet_subset.zip")
    print(
        f"Downloaded to {IMAGENET_RAW_DIR}. Leave the .zip compressed -- "
        "do not extract it; scripts/02_build_imagenet_manifest.py reads directly from it."
    )


if __name__ == "__main__":
    main()
