from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

from src.utils.enums.path_enums import Paths


class ImageManifestDataset(Dataset):
    def __init__(self, rows, image_root, transform):
        self.samples = list(
            rows[["image_path", "label"]].itertuples(index=False, name=None)
        )
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(self.image_root / path) as image:
            return self.transform(image.convert("RGB")), int(label)


def load_data(filename, batch_size, test_split=0.2, validation_split=0.1, *, train_fraction=1.0):
    if not 0 < train_fraction <= 1:
        raise ValueError("Training fraction must be greater than 0 and at most 1.")

    manifest = Path(Paths.DATA) / filename
    if not manifest.is_file() and not manifest.suffix:
        manifest = manifest.with_suffix(".csv")

    rows = pd.read_csv(manifest, dtype={"image_path": "string", "label": "Int64"})
    if not {"image_path", "label"}.issubset(rows.columns):
        raise ValueError(f"{manifest} must contain image_path and label columns.")
    if rows["image_path"].isna().any() or rows["image_path"].str.strip().eq("").any():
        raise ValueError(f"{manifest} contains an empty image path.")
    if rows["label"].isna().any() or (rows["label"] < 0).any():
        raise ValueError(f"{manifest} labels must be nonnegative integers.")

    held_out_fraction = test_split + validation_split
    if test_split <= 0 or validation_split <= 0 or held_out_fraction >= 1:
        raise ValueError("Test and validation fractions must be positive and sum to less than 1.")

    train_rows, held_out_rows = train_test_split(
        rows, test_size=held_out_fraction, random_state=42
    )
    test_rows, val_rows = train_test_split(
        held_out_rows,
        test_size=validation_split / held_out_fraction,
        random_state=42,
    )
    if train_fraction < 1:
        train_rows, _ = train_test_split(
            train_rows,
            train_size=train_fraction,
            random_state=42,
            stratify=train_rows["label"],
        )

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    return (
        DataLoader(
            ImageManifestDataset(train_rows, manifest.parent, transform),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            ImageManifestDataset(test_rows, manifest.parent, transform),
            batch_size=batch_size,
        ),
        DataLoader(
            ImageManifestDataset(val_rows, manifest.parent, transform),
            batch_size=batch_size,
        ),
    )
