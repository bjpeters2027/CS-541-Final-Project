"""Sanity checks across the whole data-prep pipeline. Run after every stage.

Checks: manifest sizes, seen/unseen disjointness, every split label has a BERT
embedding (word2vec OOV is only a warning), and a handful of images per
manifest actually decode straight out of their zip archive.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBEDDINGS_DIR, MANIFESTS_DIR, SPLITS_DIR
from dataprep.archive_dataset import ArchiveBackedImageDataset


def check_manifests() -> None:
    paths = sorted(MANIFESTS_DIR.glob("*.parquet"))
    if not paths:
        print(f"no manifests found in {MANIFESTS_DIR} yet")
        return
    for path in paths:
        df = pd.read_parquet(path)
        print(f"{path.name}: {len(df)} rows")


def check_splits_disjoint() -> None:
    seen_path = SPLITS_DIR / "seen.json"
    if not seen_path.exists():
        print(f"no splits found in {SPLITS_DIR} yet")
        return
    seen_names = {c["class_name"].strip().lower() for c in json.loads(seen_path.read_text())["classes"]}
    for path in sorted(SPLITS_DIR.glob("unseen_*.json")):
        unseen = json.loads(path.read_text())
        overlap = seen_names & {c["class_name"].strip().lower() for c in unseen}
        assert not overlap, f"{path.name} overlaps seen classes: {overlap}"
    print("disjointness check passed")


def check_embeddings_complete() -> None:
    embeddings_path = EMBEDDINGS_DIR / "label_embeddings.npz"
    if not embeddings_path.exists():
        print(f"no embeddings found in {EMBEDDINGS_DIR} yet")
        return
    data = np.load(embeddings_path, allow_pickle=True)
    labels, word2vec, bert = data["labels"], data["word2vec"], data["bert"]
    missing_bert = [l for l, v in zip(labels, bert) if v is None]
    assert not missing_bert, f"missing BERT embeddings: {missing_bert}"
    missing_w2v = [l for l, v in zip(labels, word2vec) if v is None]
    if len(missing_w2v):
        print(f"warning: {len(missing_w2v)} labels have no word2vec embedding (OOV): {list(missing_w2v)}")
    print(f"embeddings cover {len(labels)} labels")


def spot_check_images(manifest_path, n: int = 5) -> None:
    manifest = pd.read_parquet(manifest_path)
    sample = manifest.sample(min(n, len(manifest)), random_state=0)
    dataset = ArchiveBackedImageDataset(sample)
    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert image is not None
    print(f"{manifest_path.name}: decoded {len(sample)} sample images OK")


def main() -> None:
    check_manifests()
    check_splits_disjoint()
    check_embeddings_complete()
    for path in sorted(MANIFESTS_DIR.glob("*.parquet")):
        if "classes" in path.stem:
            continue
        spot_check_images(path)


if __name__ == "__main__":
    main()
