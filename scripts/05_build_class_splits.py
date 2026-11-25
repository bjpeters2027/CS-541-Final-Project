"""Build the seen/unseen class splits used throughout training and evaluation.

Seen classes come from ImageNet; two kinds of unseen classes are produced so
in-domain and out-of-domain generalization can be compared separately:
  - unseen_imagenet: ImageNet classes held out of training (same domain as seen)
  - unseen_<domain>: AWA2 / CUB / SUN classes under their official Xian et al.
    proposed split, which is verified to have zero overlap with ImageNet-1k.

Which ImageNet classes are seen was decided by scripts/01_fetch_imagenet.py
and recorded per image in the manifest's partition column: seen classes have
train images, unseen classes only have test_unseen images.

Disjointness is asserted explicitly (by class id and by normalized label
string) rather than assumed -- this is the exact leakage bug Xian et al. flag
in the original (non-proposed) AWA split.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DOMAIN_RAW_DIRS, MANIFESTS_DIR, RANDOM_SEED, SPLITS_DIR


def normalize(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def build_imagenet_splits() -> tuple[list[dict], list[dict]]:
    manifest = pd.read_parquet(MANIFESTS_DIR / "imagenet.parquet")

    def classes_with(partition: str) -> list[dict]:
        rows = manifest.loc[manifest["partition"] == partition, ["wnid", "class_name", "synonyms"]]
        return rows.drop_duplicates().sort_values("wnid").to_dict("records")

    return classes_with("train"), classes_with("test_unseen")


def build_domain_split(domain: str) -> list[dict]:
    manifest_path = MANIFESTS_DIR / f"{domain}.parquet"
    if not manifest_path.exists():
        print(f"skip {domain}: no manifest at {manifest_path} yet (run scripts/04 first)")
        return []
    manifest = pd.read_parquet(manifest_path)
    return (
        manifest.loc[manifest["is_unseen_ps"], ["class_name"]]
        .drop_duplicates()
        .to_dict("records")
    )


def assert_disjoint(seen: list[dict], *unseen_groups: list[dict]) -> None:
    seen_names = {normalize(c["class_name"]) for c in seen}
    seen_wnids = {c.get("wnid") for c in seen if c.get("wnid")}
    for group in unseen_groups:
        for c in group:
            name = normalize(c["class_name"])
            assert name not in seen_names, (
                f"leakage: {c['class_name']!r} appears in both seen and unseen sets"
            )
            if c.get("wnid"):
                assert c["wnid"] not in seen_wnids, f"leakage: wnid {c['wnid']} in both sets"


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    seen, unseen_imagenet = build_imagenet_splits()
    unseen_by_domain = {d: build_domain_split(d) for d in DOMAIN_RAW_DIRS}

    assert_disjoint(seen, unseen_imagenet, *unseen_by_domain.values())

    (SPLITS_DIR / "seen.json").write_text(
        json.dumps({"k": len(seen), "seed": RANDOM_SEED, "classes": seen}, indent=2)
    )
    (SPLITS_DIR / "unseen_imagenet.json").write_text(json.dumps(unseen_imagenet, indent=2))
    for domain, classes in unseen_by_domain.items():
        (SPLITS_DIR / f"unseen_{domain}.json").write_text(json.dumps(classes, indent=2))

    counts = " ".join(f"{d}={len(c)}" for d, c in unseen_by_domain.items())
    print(f"seen={len(seen)} unseen_imagenet={len(unseen_imagenet)} {counts}")
    print(f"wrote splits -> {SPLITS_DIR}")


if __name__ == "__main__":
    main()
