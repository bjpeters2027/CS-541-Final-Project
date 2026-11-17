"""Map an archive member path to its raw class name for each domain dataset.

Shared by scripts/03 (to pick which images to keep while fetching) and
scripts/04 (to index the staged archives), so both agree on class names.

Archive layouts, confirmed directly against each official source rather than
assumed:
  - AWA2 (.zip):  Animals_with_Attributes2/JPEGImages/<class>/<class>_<n>.jpg
                  (class names use "+" for multi-word, e.g. "blue+whale" --
                  matches testclasses.txt exactly)
  - CUB  (.tgz):  CUB_200_2011/images/<class>/<image>.jpg
                  (class folder names match testclasses.txt exactly, e.g.
                  "043.Yellow_bellied_Flycatcher")
  - SUN  (.tgz):  images/<first_letter_of_class>/<class>/<image>.jpg for most
                  classes, but compound indoor/outdoor classes get an extra
                  folder level instead of an underscore, e.g. the class
                  "quonset_hut_outdoor" lives at images/q/quonset_hut/outdoor/
                  -- confirmed by inspecting the real archive, not assumed.
                  images/outliers/... and images/misc/... are review/unsorted
                  buckets, not real classes, and are skipped.
"""
import re

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
