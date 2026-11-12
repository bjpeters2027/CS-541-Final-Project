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
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DOMAIN_DOWNLOAD_URLS,
    DOMAIN_RAW_DIRS,
    PS_SPLIT_FILES,
    XLSA17_DATASET_DIRS,
    XLSA17_URL,
    ZSL_SPLITS_DIR,
)
from dataprep.http_zip import fetch_member, list_members


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(response.raw, f, length=1024 * 1024)


def fetch_domain_archive(domain: str, url: str) -> None:
    raw_dir = DOMAIN_RAW_DIRS[domain]
    raw_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".zip" if url.endswith(".zip") else ".tgz"
    archive_path = raw_dir / f"{domain}{suffix}"
    tar_path = raw_dir / f"{domain}.tar"

    if archive_path.exists() or tar_path.exists():
        print(f"{domain}: already staged in {raw_dir}, skipping download.")
        return

    print(f"{domain}: downloading {url} -> {archive_path}")
    download(url, archive_path)

    if suffix == ".tgz":
        print(f"{domain}: decompressing to a plain .tar for random-access reads")
        with gzip.open(archive_path, "rb") as src, open(tar_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        archive_path.unlink()  # re-downloadable; no need to keep both forms


def fetch_proposed_splits() -> None:
    ZSL_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        members = list_members(XLSA17_URL)
    except requests.RequestException as exc:
        print(
            f"Could not reach {XLSA17_URL} ({exc}). Download xlsa17.zip manually and "
            f"place xlsa17/data/<DATASET>/{{trainvalclasses,testclasses}}.txt under "
            f"{ZSL_SPLITS_DIR}/<domain>/ for domain in {list(XLSA17_DATASET_DIRS)}.",
            file=sys.stderr,
        )
        return

    for domain, dataset_dir in XLSA17_DATASET_DIRS.items():
        out_dir = ZSL_SPLITS_DIR / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in PS_SPLIT_FILES:
            member_name = f"xlsa17/data/{dataset_dir}/{filename}"
            offset, csize, ctype = members[member_name]
            data = fetch_member(XLSA17_URL, offset, csize, ctype)
            (out_dir / filename).write_bytes(data)
        print(f"{domain}: proposed-split class lists -> {out_dir}")


def main() -> None:
    for domain, url in DOMAIN_DOWNLOAD_URLS.items():
        fetch_domain_archive(domain, url)
    fetch_proposed_splits()


if __name__ == "__main__":
    main()
