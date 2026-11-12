"""Dataset that reads images directly out of zip/tar archives, never extracting
them to individual files on disk.

Turing's storage guidance is explicit about this: unpacking a dataset with a
million-plus small files "puts serious strain on the file server," and the
documented fix is to read members straight out of the archive instead. This
covers both archive kinds we actually use: zip (ImageNet, AWA2) via Python's
zipfile, and plain uncompressed tar (CUB, SUN, after scripts/03 converts their
.tar.gz once) via tarfile -- tarfile only gets true random-access seeks on an
uncompressed tar, not a .tar.gz, which is why that conversion happens upfront
instead of re-decompressing per read.
"""
import tarfile
import threading
import zipfile
from io import BytesIO

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ArchiveBackedImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self._local = threading.local()

    def _archive(self, path: str):
        cache = getattr(self._local, "archives", None)
        if cache is None:
            cache = {}
            self._local.archives = cache
        archive = cache.get(path)
        if archive is None:
            archive = zipfile.ZipFile(path, "r") if path.endswith(".zip") else tarfile.open(path, "r")
            cache[path] = archive
        return archive

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _read_bytes(archive, member_path: str) -> bytes:
        if isinstance(archive, zipfile.ZipFile):
            with archive.open(member_path) as f:
                return f.read()
        member = archive.extractfile(member_path)
        return member.read()

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        archive = self._archive(row.archive_path)
        image = Image.open(BytesIO(self._read_bytes(archive, row.member_path))).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row.to_dict()
