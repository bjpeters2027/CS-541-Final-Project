import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, default_collate

from config import MANIFESTS_DIR, SPLITS_DIR
from dataprep.archive_dataset import ArchiveBackedImageDataset
from dataprep.transforms import build_transform


def collate_labeled_images(batch):
    images, labels = default_collate([
        (image, int(metadata["label"])) for image, metadata in batch
    ])
    return images, labels


def close_loaders(*loaders: DataLoader) -> None:
    for loader in loaders:
        dataset = loader.dataset
        if not isinstance(dataset, ArchiveBackedImageDataset):
            raise TypeError("Expected an archive-backed dataset.")
        dataset.close()


def load_data(data_config: dict, batch_size: int, seed: int = 541):
    """Return train, held-out test, and validation loaders in that order.

    Official ImageNet ``val`` images are reserved for the final test. Validation
    images are held out per class from the official training partition.
    Labels follow the order of entries in seen_classes, matching embedding rows.
    """
    val_split = data_config["val_split"]
    if not 0 < val_split < 1:
        raise ValueError("data.val_split must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    manifest_path = Path(data_config["manifest"])
    split_path = Path(data_config["seen_classes"])
    if not manifest_path.is_absolute() and not manifest_path.exists():
        manifest_path = MANIFESTS_DIR / manifest_path
    if not split_path.is_absolute() and not split_path.exists():
        split_path = SPLITS_DIR / split_path

    with split_path.open(encoding="utf-8") as file:
        split = json.load(file)
    classes = split["classes"] if isinstance(split, dict) else split
    wnids = [entry["wnid"] for entry in classes]
    if not wnids or len(set(wnids)) != len(wnids):
        raise ValueError("seen_classes must contain unique, nonempty ImageNet classes.")

    manifest = pd.read_parquet(manifest_path)
    required = {"wnid", "split", "archive_path", "member_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required - set(manifest.columns))}")
    manifest = manifest.loc[manifest["wnid"].isin(wnids)].copy()
    if manifest.duplicated(["archive_path", "member_path"]).any():
        raise ValueError("Manifest contains duplicate images across partitions.")
    manifest["label"] = manifest["wnid"].map({wnid: i for i, wnid in enumerate(wnids)})
    training = manifest.loc[manifest["split"] == "train"]
    test = manifest.loc[manifest["split"] == "val"]
    if test.empty:
        raise ValueError("Manifest must contain held-out official 'val' images for testing.")

    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    indices_by_class = training.groupby("wnid").groups
    for wnid in wnids:
        indices = np.asarray(indices_by_class.get(wnid, []))
        if len(indices) < 2:
            raise ValueError(f"Class {wnid!r} needs at least two training images.")
        indices = rng.permutation(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * val_split)))
        val_indices.extend(indices[:count])
        train_indices.extend(indices[count:])

    def loader(frame: pd.DataFrame, train: bool) -> DataLoader:
        return DataLoader(
            ArchiveBackedImageDataset(frame, transform=build_transform(train)),
            batch_size=batch_size,
            shuffle=train,
            num_workers=data_config.get("num_workers", 0),
            generator=torch.Generator().manual_seed(seed),
            collate_fn=collate_labeled_images,
        )

    return (
        loader(manifest.loc[train_indices], True),
        loader(test, False),
        loader(manifest.loc[val_indices], False),
    )
