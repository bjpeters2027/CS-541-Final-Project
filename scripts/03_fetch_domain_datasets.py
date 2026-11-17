"""Stage a small evaluation subset of the three out-of-domain datasets (AWA2,
CUB-200-2011, SUN attributes), plus Xian et al.'s official "proposed split"
(PS) class lists, which are what guarantee these classes don't overlap with
ImageNet-1k.

Only the PS test (unseen) classes are ever used for evaluation, so only those
are kept, capped at config.DOMAIN_IMAGES_PER_CLASS randomly chosen images per
class. Each domain's subset is written to raw/<domain>/<domain>.zip using the
original archive's member paths, so scripts/04 indexes it unchanged.

Images come from each dataset's own official source (verified directly, not a
possibly-stale Kaggle mirror):
  - AWA2: a ~14GB .zip whose server supports HTTP range requests, so only the
    chosen images are downloaded (see dataprep/http_zip.py).
  - CUB / SUN: a ~1.2GB / ~1.8GB .tar.gz each, which has no random access, so
    each is streamed once and only the chosen images are kept. Nothing else is
    written to disk.

Run this from a login-node shell, not inside an sbatch job -- compute nodes on
many HPC clusters have no outbound internet access.
"""
import random
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DOMAIN_DOWNLOAD_URLS,
    DOMAIN_IMAGES_PER_CLASS,
    DOMAIN_RAW_DIRS,
    PS_SPLIT_FILES,
    RANDOM_SEED,
    XLSA17_DATASET_DIRS,
    XLSA17_URL,
    ZSL_SPLITS_DIR,
)
from dataprep.domains import CLASSIFY
from dataprep.http_zip import open_remote_zip


def fetch_proposed_splits() -> None:
    missing = [
        domain for domain in XLSA17_DATASET_DIRS
        if not all((ZSL_SPLITS_DIR / domain / name).exists() for name in PS_SPLIT_FILES)
    ]
    if not missing:
        return
    try:
        with open_remote_zip(XLSA17_URL) as archive:
            for domain in missing:
                out_dir = ZSL_SPLITS_DIR / domain
                out_dir.mkdir(parents=True, exist_ok=True)
                for filename in PS_SPLIT_FILES:
                    member_name = f"xlsa17/data/{XLSA17_DATASET_DIRS[domain]}/{filename}"
                    (out_dir / filename).write_bytes(archive.read(member_name))
                print(f"{domain}: proposed-split class lists -> {out_dir}")
    except requests.RequestException as exc:
        sys.exit(
            f"Could not reach {XLSA17_URL} ({exc}). Download xlsa17.zip manually and "
            f"place xlsa17/data/<DATASET>/{{trainvalclasses,testclasses}}.txt under "
            f"{ZSL_SPLITS_DIR}/<domain>/ for domain in {list(XLSA17_DATASET_DIRS)}."
        )


def load_unseen_classes(domain: str) -> set[str]:
    path = ZSL_SPLITS_DIR / domain / "testclasses.txt"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def select_from_remote_zip(url: str, domain: str, unseen: set[str], rng: random.Random):
    classify = CLASSIFY[domain]
    with open_remote_zip(url) as archive:
        by_class = defaultdict(list)
        for name in sorted(archive.namelist()):
            if classify(name) in unseen:
                by_class[classify(name)].append(name)
        chosen = [
            name
            for class_name in sorted(by_class)
            for name in rng.sample(by_class[class_name], min(DOMAIN_IMAGES_PER_CLASS, len(by_class[class_name])))
        ]
        print(f"{domain}: fetching {len(chosen)} images from {len(by_class)} unseen classes")
        for index, name in enumerate(chosen, 1):
            yield name, archive.read(name)
            if index % 100 == 0:
                print(f"{domain}: {index}/{len(chosen)}", flush=True)


def select_from_streamed_tgz(url: str, domain: str, unseen: set[str], rng: random.Random):
    # Reservoir-sample each class during a single pass, since a .tar.gz can only
    # be read front to back.
    classify = CLASSIFY[domain]
    reservoirs = defaultdict(list)
    seen_counts = defaultdict(int)
    print(f"{domain}: streaming {url} (only unseen-class images are kept)")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                class_name = classify(member.name) if member.isfile() else None
                if class_name not in unseen:
                    continue
                seen_counts[class_name] += 1
                reservoir = reservoirs[class_name]
                if len(reservoir) < DOMAIN_IMAGES_PER_CLASS:
                    reservoir.append((member.name, archive.extractfile(member).read()))
                else:
                    slot = rng.randrange(seen_counts[class_name])
                    if slot < DOMAIN_IMAGES_PER_CLASS:
                        reservoir[slot] = (member.name, archive.extractfile(member).read())
    for class_name in sorted(reservoirs):
        yield from sorted(reservoirs[class_name])


def fetch_domain_subset(domain: str, url: str) -> None:
    raw_dir = DOMAIN_RAW_DIRS[domain]
    raw_dir.mkdir(parents=True, exist_ok=True)
    subset_path = raw_dir / f"{domain}.zip"
    if subset_path.exists():
        print(f"{domain}: subset already staged at {subset_path}, skipping download.")
        return
    stale = [path for path in raw_dir.iterdir() if path.suffix in (".zip", ".tar", ".tgz")]
    if stale:
        sys.exit(f"{domain}: remove the previously staged full archive(s) {stale} first; "
                 "scripts/04 reads whichever archive it finds in that folder.")

    unseen = load_unseen_classes(domain)
    rng = random.Random(f"{RANDOM_SEED}-{domain}")
    select = select_from_remote_zip if url.endswith(".zip") else select_from_streamed_tgz

    partial_path = raw_dir / f"{domain}.partial"
    count = 0
    # JPEGs are already compressed, so store them as-is.
    with zipfile.ZipFile(partial_path, "w", zipfile.ZIP_STORED) as subset:
        for name, data in select(url, domain, unseen, rng):
            subset.writestr(name, data)
            count += 1
    if count == 0:
        partial_path.unlink()
        raise RuntimeError(
            f"{domain}: no images matched the {len(unseen)} unseen classes in "
            f"testclasses.txt. The archive layout may not match dataprep/domains.py."
        )
    partial_path.replace(subset_path)
    size_mb = subset_path.stat().st_size / 1e6
    print(f"{domain}: {count} images ({size_mb:.0f} MB) -> {subset_path}")


def main() -> None:
    fetch_proposed_splits()
    for domain, url in DOMAIN_DOWNLOAD_URLS.items():
        fetch_domain_subset(domain, url)


if __name__ == "__main__":
    main()
