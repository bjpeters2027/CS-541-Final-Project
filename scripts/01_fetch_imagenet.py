"""Stage the raw ImageNet (ILSVRC2012) archive into scratch storage, compressed.

ImageNet access is credential-gated, so this script doesn't scrape anything --
it drives the Kaggle CLI if credentials are present, and otherwise prints the
manual options and exits. The downloaded zip is left exactly as Kaggle serves
it: scripts/02_build_imagenet_manifest.py reads straight out of it with
zipfile, so nothing here ever gets extracted.

Run this from a login-node shell, not inside an sbatch job -- compute nodes on
many HPC clusters have no outbound internet access.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGENET_RAW_DIR

KAGGLE_COMPETITION = "imagenet-object-localization-challenge"

MANUAL_INSTRUCTIONS = f"""
No Kaggle credentials found (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).

Option A - Kaggle (recommended, matches what scripts/02 expects):
  1. Create an API token at https://www.kaggle.com/settings -> "Create New Token".
  2. Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or export
     KAGGLE_USERNAME / KAGGLE_KEY.
  3. Re-run this script.

Option B - Academic Torrents:
  1. Get the ILSVRC2012 torrent from https://academictorrents.com (search
     "ImageNet Large Scale Visual Recognition Challenge 2012").
  2. Download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar into:
       {IMAGENET_RAW_DIR}
  3. You'll also need LOC_synset_mapping.txt and LOC_val_solution.csv (the
     wnid -> class-name and val-image -> label mappings scripts/02 expects).
     Both ship inside the Kaggle "{KAGGLE_COMPETITION}" dataset, so even on
     the torrent path it's worth pulling just those two small files from
     Kaggle rather than re-deriving them from the old devkit.
  4. scripts/02_build_imagenet_manifest.py currently assumes a single Kaggle
     zip; if you go this route you'll need to adjust it for the tar layout.
"""


def have_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> None:
    IMAGENET_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if any(IMAGENET_RAW_DIR.glob("*.zip")):
        print(f"Found an existing zip in {IMAGENET_RAW_DIR}, skipping download.")
        return

    if shutil.which("kaggle") is None:
        print("The 'kaggle' CLI isn't installed. Add it with `uv add kaggle`.", file=sys.stderr)
        sys.exit(1)

    if not have_kaggle_credentials():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", KAGGLE_COMPETITION,
            "-p", str(IMAGENET_RAW_DIR),
        ],
        check=True,
    )
    print(
        f"Downloaded to {IMAGENET_RAW_DIR}. Leave the .zip compressed -- "
        "do not extract it; scripts/02_build_imagenet_manifest.py reads directly from it."
    )


if __name__ == "__main__":
    main()
