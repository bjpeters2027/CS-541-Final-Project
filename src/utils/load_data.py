import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import MANIFESTS_DIR, SPLITS_DIR
from dataprep.archive_dataset import ArchiveBackedImageDataset
from dataprep.transforms import build_transform
from src.utils.label_embeddings import load_class_split

def load_data(data_config, batch_size, seed):
    """
    Load the seen-class ImageNet partitions (chosen by scripts/01_fetch_imagenet.py) as DataLoaders.

    Args:
        data_config (dict): The "data" section of a model config. "manifest" and "seen_classes"
            are resolved inside the data prep manifests/ and splits/ folders unless absolute.
        batch_size (int): Number of samples per batch.
        seed (int): Seed for shuffling the training data.

    Returns:
        tuple: Training, validation, and seen-class test DataLoaders, and the number of seen classes.
            Labels are indices into the seen classes, in seen.json order.
    """

    manifest = pd.read_parquet(MANIFESTS_DIR / data_config["manifest"])
    seen_classes = load_class_split(SPLITS_DIR / data_config["seen_classes"])
    class_index = {entry["wnid"]: index for index, entry in enumerate(seen_classes)}

    loaders = []
    for partition, train in (("train", True), ("validation", False), ("test_seen", False)):
        rows = manifest[manifest["partition"] == partition].copy()
        labels = rows["wnid"].map(class_index)
        if rows.empty or labels.isna().any():
            raise ValueError(f"The {partition} partition is empty or contains classes missing from seen classes.")
        rows["label"] = labels.astype(int)
        dataset = ArchiveBackedImageDataset(rows, build_transform(train), target_column="label")
        generator = torch.Generator().manual_seed(seed) if train else None
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=data_config["num_workers"], generator=generator))
    return (*loaders, len(seen_classes))
