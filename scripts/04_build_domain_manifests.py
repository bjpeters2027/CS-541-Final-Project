"""Index each domain dataset's images directly out of its archive (no
extraction -- see dataprep/archive_dataset.py), and mark which classes are
unseen under Xian et al.'s official proposed split (scripts/03 already pulled
testclasses.txt / trainvalclasses.txt for each domain from their benchmark
release).

Member paths are mapped to class names by dataprep.domains, which documents
each archive's layout.
"""
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DOMAIN_RAW_DIRS, MANIFESTS_DIR, ZSL_SPLITS_DIR
from dataprep.domains import CLASSIFY


def denormalize_awa2(class_name: str) -> str:
    return class_name.replace("_", " ").replace("+", " ")


def denormalize_cub(class_name: str) -> str:
    # "043.Yellow_bellied_Flycatcher" -> "Yellow bellied Flycatcher"
    return class_name.split(".", 1)[-1].replace("_", " ")


def denormalize_sun(class_name: str) -> str:
    return class_name.replace("_", " ")


DENORMALIZE = {"awa2": denormalize_awa2, "cub": denormalize_cub, "sun": denormalize_sun}


def find_archive(domain: str) -> tuple[Path, str]:
    raw_dir = DOMAIN_RAW_DIRS[domain]
    for path in raw_dir.glob("*"):
        if path.suffix in (".zip", ".tar"):
            return path, "zip" if path.suffix == ".zip" else "tar"
    raise FileNotFoundError(f"No archive found in {raw_dir}. Run scripts/03_fetch_domain_datasets.py first.")


def iter_members(archive_path: Path, archive_kind: str):
    if archive_kind == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            yield from archive.namelist()
    else:
        with tarfile.open(archive_path, "r") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    yield member.name


def load_unseen_classes(domain: str) -> set[str]:
    path = ZSL_SPLITS_DIR / domain / "testclasses.txt"
    if not path.exists():
        print(f"warning: no testclasses.txt for {domain} at {path} -- run scripts/03 first")
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def build_domain_manifest(domain: str) -> None:
    archive_path, archive_kind = find_archive(domain)
    classify = CLASSIFY[domain]
    denormalize = DENORMALIZE[domain]
    unseen_raw_names = load_unseen_classes(domain)

    rows = []
    for member_path in iter_members(archive_path, archive_kind):
        raw_class_name = classify(member_path)
        if raw_class_name is None:
            continue
        rows.append({
            "image_id": Path(member_path).stem,
            "archive_path": str(archive_path),
            "member_path": member_path,
            "raw_class_name": raw_class_name,
            "class_name": denormalize(raw_class_name),
            "is_unseen_ps": raw_class_name in unseen_raw_names,
        })

    if not rows:
        raise RuntimeError(
            f"Found 0 images in {archive_path} for {domain}. "
            "The archive's internal layout may not match what this script expects."
        )

    manifest = pd.DataFrame(rows)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFESTS_DIR / f"{domain}.parquet", index=False)

    n_unseen_classes = manifest.loc[manifest["is_unseen_ps"], "class_name"].nunique()
    print(
        f"{domain}: indexed {len(manifest)} images, "
        f"{manifest['class_name'].nunique()} classes ({n_unseen_classes} unseen under PS) -> "
        f"{MANIFESTS_DIR / f'{domain}.parquet'}"
    )


def main() -> None:
    for domain in DOMAIN_RAW_DIRS:
        build_domain_manifest(domain)


if __name__ == "__main__":
    main()
