"""Check class identity by WordNet ID, falling back to normalized names."""

from collections.abc import Iterable, Mapping


def normalize(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def assert_disjoint(
    seen: Iterable[Mapping[str, str]],
    *unseen_groups: Iterable[Mapping[str, str]],
) -> None:
    seen_wnids = set()
    seen_names = set()
    seen_names_without_wnid = set()
    for entry in seen:
        name = normalize(entry["class_name"])
        seen_names.add(name)
        if entry.get("wnid"):
            seen_wnids.add(entry["wnid"])
        else:
            seen_names_without_wnid.add(name)

    for group in unseen_groups:
        for entry in group:
            wnid = entry.get("wnid")
            name = normalize(entry["class_name"])
            if wnid and wnid in seen_wnids:
                raise ValueError(
                    f"leakage: wnid {wnid!r} ({entry['class_name']!r}) "
                    "appears in both seen and unseen sets"
                )
            names_to_check = seen_names_without_wnid if wnid else seen_names
            if name in names_to_check:
                raise ValueError(
                    f"leakage: normalized label {name!r} appears in both seen "
                    "and unseen sets without comparable WordNet IDs"
                )
