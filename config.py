"""Central paths and constants for the data prep pipeline.

Defaults match WPI Turing's storage conventions (docs.turing.wpi.edu/best-practices/storing_data):
raw, publicly re-downloadable archives go to scratch (not backed up, fine to lose);
code and small derived artifacts (manifests/splits/embeddings) go to home (backed up).
Every root is overridable via env vars so the same scripts run on a local machine too.
"""
import os
from pathlib import Path

USER = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

SCRATCH_ROOT = Path(os.environ.get("DATAPREP_SCRATCH_ROOT", f"/scratch/{USER}"))
HOME_ROOT = Path(os.environ.get("DATAPREP_HOME_ROOT", Path(__file__).resolve().parent))

RAW_ROOT = Path(os.environ.get("DATAPREP_RAW_ROOT", SCRATCH_ROOT / "raw"))
DATA_ROOT = Path(os.environ.get("DATAPREP_DATA_ROOT", HOME_ROOT / "data"))

MANIFESTS_DIR = DATA_ROOT / "manifests"
SPLITS_DIR = DATA_ROOT / "splits"
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"
ZSL_SPLITS_DIR = DATA_ROOT / "zsl_splits"  # tiny Xian et al. proposed-split files, committed to repo

IMAGENET_RAW_DIR = RAW_ROOT / "imagenet"
DOMAIN_RAW_DIRS = {
    "awa2": RAW_ROOT / "awa2",
    "cub": RAW_ROOT / "cub",
    "sun": RAW_ROOT / "sun",
}

# Official, verified sources (not Kaggle mirrors -- these datasets' own project
# pages are more reliable than guessing which Kaggle re-upload is current).
DOMAIN_DOWNLOAD_URLS = {
    "awa2": "https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
    "cub": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
    "sun": "https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz",
}

# Xian et al.'s official "proposed split" (PS) release -- the source of truth for
# which classes are unseen/test for each domain, verified to have zero overlap
# with ImageNet-1k. scripts/03 pulls just the two small class-list files per
# domain out of this zip via HTTP range requests (it's ~900MB total, almost all
# of it pre-extracted ResNet101 features we don't need).
XLSA17_URL = "https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
XLSA17_DATASET_DIRS = {"awa2": "AWA2", "cub": "CUB", "sun": "SUN"}
PS_SPLIT_FILES = ("trainvalclasses.txt", "testclasses.txt")

RANDOM_SEED = int(os.environ.get("DATAPREP_SEED", "541"))

DEFAULT_K_SEEN_CLASSES = 1000  # how many of ImageNet's 1000 classes to actually train on

WORD2VEC_MODEL_NAME = "word2vec-google-news-300"
BERT_MODEL_NAME = "bert-base-uncased"
BERT_TEMPLATE = "a photo of a {label}."

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

for _dir in (MANIFESTS_DIR, SPLITS_DIR, EMBEDDINGS_DIR, ZSL_SPLITS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
