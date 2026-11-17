"""Index each domain dataset's images directly out of its archive (no
extraction -- see dataprep/archive_dataset.py), and mark which classes are
unseen under Xian et al.'s official proposed split (scripts/03 already pulled
testclasses.txt / trainvalclasses.txt for each domain from their benchmark
release).

Archive layouts, confirmed directly against each official source rather than
assumed:
  - AWA2 (.zip):  Animals_with_Attributes2/JPEGImages/<class>/<class>_<n>.jpg
                  (class names use "+" for multi-word, e.g. "blue+whale" --
                  matches testclasses.txt exactly)
  - CUB  (.tar):  CUB_200_2011/images/<class>/<image>.jpg
                  (class folder names match testclasses.txt exactly, e.g.
                  "043.Yellow_bellied_Flycatcher")
  - SUN  (.tar):  images/<first_letter_of_class>/<class>/<image>.jpg for most
                  classes, but compound indoor/outdoor classes get an extra
                  folder level instead of an underscore, e.g. the class
                  "quonset_hut_outdoor" lives at images/q/quonset_hut/outdoor/
                  -- confirmed by inspecting the real archive, not assumed.
                  images/outliers/... and images/misc/... are review/unsorted
                  buckets, not real classes, and are skipped.
"""
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DOMAIN_RAW_DIRS, MANIFESTS_DIR, ZSL_SPLITS_DIR

AWA2_MEMBER_RE = re.compile(r"Animals_with_Attributes2/JPEGImages/([^/]+)/[^/]+\.jpg$", re.IGNORECASE)
CUB_MEMBER_RE = re.compile(r"CUB_200_2011/images/([^/]+)/[^/]+\.jpg$", re.IGNORECASE)
SUN_SIMPLE_RE = re.compile(r"^images/[^/]/([^/]+)/[^/]+\.jpg$", re.IGNORECASE)
SUN_COMPOUND_RE = re.compile(r"^images/[^/]/([^/]+)/([^/]+)/[^/]+\.jpg$", re.IGNORECASE)


def classify_simple(pattern: re.Pattern):
    def classify(member_path: str) -> str | None:
        match = pattern.search(member_path)
        return match.group(1) if match else None
    return classify


def classify_sun(member_path: str) -> str | None:
    if member_path.startswith("images/outliers/") or member_path.startswith("images/misc/"):
        return None
    match = SUN_COMPOUND_RE.match(member_path)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    match = SUN_SIMPLE_RE.match(member_path)
    return match.group(1) if match else None


CLASSIFY = {
    "awa2": classify_simple(AWA2_MEMBER_RE),
    "cub": classify_simple(CUB_MEMBER_RE),
    "sun": classify_sun,
}


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
