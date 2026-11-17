"""Stage the three out-of-domain eval datasets (AWA2, CUB-200-2011, SUN
attributes) plus Xian et al.'s official "proposed split" (PS) class lists,
which are what guarantee these classes don't overlap with ImageNet-1k.

Images come from each dataset's own official source (verified directly, not a
possibly-stale Kaggle mirror):
  - AWA2: a .zip -- left compressed; scripts/04 reads it directly with zipfile.
  - CUB / SUN: a .tar.gz each. tar.gz can't be read with random access without
    re-decompressing from the start every time, which doesn't scale to
    per-image reads during training. So each is decompressed exactly once to
    a plain, uncompressed .tar (one file in, one file out -- the individual
    images are never written to disk separately) so dataprep.archive_dataset
    can later seek members directly.

Run this from a login-node shell, not inside an sbatch job -- compute nodes on
many HPC clusters have no outbound internet access.
"""
import gzip
import random
import shutil
import sys
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import requests
from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DOMAIN_DOWNLOAD_URLS,
    DOMAIN_RAW_DIRS,
    PS_SPLIT_FILES,
    RANDOM_SEED,
    XLSA17_DATASET_DIRS,
    XLSA17_URL,
    ZSL_SPLITS_DIR,
)

IMAGES_PER_CLASS = 20
MAX_BYTES = 7_000_000_000


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(response.raw, f, length=1024 * 1024)


def write_subset(source, domain: str, destination: Path) -> None:
    is_zip = isinstance(source, zipfile.ZipFile)
    classify = import_module("scripts.04_build_domain_manifests").CLASSIFY[domain]
    groups = defaultdict(list)
    members = source.infolist() if is_zip else source.getmembers()
    for member in members:
        name = member.filename if is_zip else member.name
        class_name = classify(name)
        if class_name is not None:
            groups[class_name].append((name, member))
    rng = random.Random(RANDOM_SEED)
    selected = []
    for class_name in sorted(groups):
        candidates = sorted(groups[class_name], key=lambda entry: entry[0])
        selected.extend(rng.sample(candidates, min(IMAGES_PER_CLASS, len(candidates))))
    if is_zip:
        for name, _ in selected.copy():
            license_name = name.replace("/JPEGImages/", "/licenses/").rsplit(".", 1)[0] + ".txt"
            try:
                selected.append((license_name, source.getinfo(license_name)))
            except KeyError:
                pass
    size = sum((m.file_size if is_zip else m.size) + 4096 for _, m in selected) + 10240
    if not is_zip:
        size += Path(source.name).stat().st_size
    if size >= MAX_BYTES:
        raise ValueError(f"{domain}: source plus subset would exceed 7 GB; reduce IMAGES_PER_CLASS.")
    partial = destination.with_suffix(".partial")
    open_archive = zipfile.ZipFile if is_zip else tarfile.open
    with open_archive(partial, "w") as output:
        for name, member in selected:
            if is_zip:
                output.writestr(name, source.read(member))
            else:
                with source.extractfile(member) as image:
                    output.addfile(member, image)
    partial.replace(destination)


def fetch_domain_archive(domain: str, url: str) -> None:
    raw_dir = DOMAIN_RAW_DIRS[domain]
    raw_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".zip" if url.endswith(".zip") else ".tgz"
    archive_path = raw_dir / f"{domain}{suffix}"
    tar_path = raw_dir / f"{domain}.tar"
    subset_path = raw_dir / f"{domain}_subset{'.zip' if suffix == '.zip' else '.tar'}"

    if any(p.suffix in (".zip", ".tar", ".tgz", ".gz") and p != subset_path for p in raw_dir.iterdir()):
        raise RuntimeError(f"Move or delete old archives in {raw_dir} before creating the subset.")
    if subset_path.exists():
        print(f"{domain}: already staged in {raw_dir}, skipping download.")
        return

    if suffix == ".zip":
        print(f"{domain}: downloading a subset from {url} -> {subset_path}")
        with requests.Session() as session, RemoteZip(url, session=session, timeout=300) as source:
            write_subset(source, domain, subset_path)
        return

    with tempfile.TemporaryDirectory(dir=raw_dir) as staging:
        archive_path = Path(staging) / archive_path.name
        tar_path = Path(staging) / tar_path.name
        print(f"{domain}: downloading {url} -> {archive_path}")
        download(url, archive_path)
        print(f"{domain}: decompressing to a plain .tar for random-access reads")
        with gzip.open(archive_path, "rb") as src, open(tar_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        archive_path.unlink()  # re-downloadable; no need to keep both forms
        with tarfile.open(tar_path, "r") as source:
            write_subset(source, domain, subset_path)


def fetch_proposed_splits() -> None:
    ZSL_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Only the small class-list files are fetched, via HTTP range requests.
        source = RemoteZip(XLSA17_URL, timeout=300)
    except requests.RequestException as exc:
        print(
            f"Could not reach {XLSA17_URL} ({exc}). Download xlsa17.zip manually and "
            f"place xlsa17/data/<DATASET>/{{trainvalclasses,testclasses}}.txt under "
            f"{ZSL_SPLITS_DIR}/<domain>/ for domain in {list(XLSA17_DATASET_DIRS)}.",
            file=sys.stderr,
        )
        return

    with source:
        for domain, dataset_dir in XLSA17_DATASET_DIRS.items():
            out_dir = ZSL_SPLITS_DIR / domain
            out_dir.mkdir(parents=True, exist_ok=True)
            for filename in PS_SPLIT_FILES:
                data = source.read(f"xlsa17/data/{dataset_dir}/{filename}")
                (out_dir / filename).write_bytes(data)
            print(f"{domain}: proposed-split class lists -> {out_dir}")


def main() -> None:
    for domain, url in DOMAIN_DOWNLOAD_URLS.items():
        fetch_domain_archive(domain, url)
    fetch_proposed_splits()


if __name__ == "__main__":
    main()
