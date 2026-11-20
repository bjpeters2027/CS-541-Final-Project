# How to train our model on Turing

## 1. Connect to Turing and clone the repo

From your local terminal, replace YOUR_WPI_USERNAME with your WPI username.

ssh YOUR_WPI_USERNAME@turing.wpi.edu

On Turing, clone the dev branch if you do not already have the project.

git clone --branch dev https://github.com/bjpeters2027/CS-541-Final-Project.git
cd ~/CS-541-Final-Project

## 2. Set up Python and Kaggle

Start an interactive Slurm session to download and prepare the data on a compute node.

srun --partition=academic --ntasks=1 --cpus-per-task=2 \
  --mem=8G --time=03:00:00 --pty bash -l

From that session, load Python, create the virtual environment, and install the dependencies. On subsequent sessions, load the same Python module and activate the existing environment.

cd ~/CS-541-Final-Project
module load python/3.12.10
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

Sign in to Kaggle. If Kaggle denies the download, check that your account has accepted the competition rules.

kaggle auth login

## 3. Download and extract ImageNet

Skip this section if the images are already extracted under ~/archive/ImageNet/ILSVRC/Data/CLS-LOC/train/.

DATA_DIR="$HOME/archive/ImageNet"
mkdir -p "$DATA_DIR"

kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p "$DATA_DIR"

Extract the training images from the ZIP using Python.

python -u - <<'PY'
from pathlib import Path
from zipfile import ZipFile

archive = Path.home() / "archive/ImageNet"
zip_path = archive / "imagenet-object-localization-challenge.zip"
prefix = "ILSVRC/Data/CLS-LOC/train/"
image_suffixes = {".jpeg", ".jpg", ".png"}

with ZipFile(zip_path) as source:
    members = [
        entry for entry in source.infolist()
        if entry.filename.startswith(prefix)
        and not entry.is_dir()
        and Path(entry.filename).suffix.lower() in image_suffixes
    ]
    if not members:
        raise SystemExit(f"No training images found under {prefix}")

    print(f"Extracting {len(members):,} images to {archive}")
    for count, entry in enumerate(members, start=1):
        source.extract(entry, archive)
        if count % 50_000 == 0:
            print(f"Extracted {count:,} / {len(members):,}")

print(f"Training folder: {archive / prefix}")
PY

## 4. Build the manifests and test the data loader

Run from the repo root with .venv active.

cd ~/CS-541-Final-Project
python -m src.utils.make_imagenet_manifest \
  "$HOME/archive/ImageNet/ILSVRC/Data/CLS-LOC/train"
python -m src.utils.check_imagenet_batch

The manifest command writes src/artifacts/data/imagenet_train.csv with image paths and numeric labels. It also writes src/artifacts/data/imagenet_classes.csv with the corresponding label-to-synset mapping. Keep these files together. Embedding tensor row i must correspond to numeric label i in these manifests.

The no-embedding baseline can go straight to Step 7 after the batch check passes.

## 5. Generate class embeddings

### Create the class-name CSV

Generate ~/archive/ImageNet/imagenet_class_names.csv by matching the training manifest’s synset IDs to the published names. The join uses synset IDs because the published index and this project can assign different numeric labels. ImageNet class index

python - <<'PY'
import csv
import json
from pathlib import Path
from urllib.request import urlopen

classes_path = Path("src/artifacts/data/imagenet_classes.csv")
output = Path.home() / "archive/ImageNet/imagenet_class_names.csv"
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

with urlopen(url, timeout=30) as response:
    index = json.load(response)
names = {synset: name.replace("_", " ") for synset, name in index.values()}

with classes_path.open(newline="") as file:
    classes = sorted(csv.DictReader(file), key=lambda row: int(row["label"]))

if [int(row["label"]) for row in classes] != list(range(1000)):
    raise SystemExit(f"Expected class IDs 0 through 999 in {classes_path}")

synsets = [row["synset"] for row in classes]
missing = [synset for synset in synsets if not names.get(synset)]
if missing:
    raise SystemExit(f"Missing names for {len(missing)} synsets: {missing[:5]}")

with output.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["synset", "name"])
    writer.writerows((synset, names[synset]) for synset in synsets)

print(f"Wrote {len(synsets)} class names to {output}")
PY

### Generate BERT embeddings

This step uses bert-base-uncased. The first run may need network access to download the tokenizer and model. It saves one 768-dimensional vector per class in numeric label order. For BERT training, skip the Word2Vec step.

python - <<'PY'
import csv
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

archive = Path.home() / "archive/ImageNet"
classes_path = Path("src/artifacts/data/imagenet_classes.csv")
names_path = archive / "imagenet_class_names.csv"
output = archive / "embeddings/bert.pt"

with classes_path.open(newline="") as file:
    classes = sorted(csv.DictReader(file), key=lambda row: int(row["label"]))
with names_path.open(newline="") as file:
    names = {row["synset"]: row["name"].strip() for row in csv.DictReader(file)}

if [int(row["label"]) for row in classes] != list(range(1000)):
    raise SystemExit(f"Expected class IDs 0 through 999 in {classes_path}")
missing = [row["synset"] for row in classes if not names.get(row["synset"])]
if missing:
    raise SystemExit(f"Missing names for {len(missing)} synsets: {missing[:5]}")

labels = [names[row["synset"]] for row in classes]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()
rows = []

with torch.inference_mode():
    for start in range(0, len(labels), 32):
        batch = tokenizer(
            labels[start:start + 32],
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens = batch.pop("special_tokens_mask").bool()
        mask = (batch["attention_mask"].bool() & ~special_tokens).unsqueeze(-1)
        hidden = model(**batch).last_hidden_state
        rows.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))

embeddings = torch.cat(rows)
output.parent.mkdir(parents=True, exist_ok=True)
torch.save(embeddings, output)
print(f"Wrote {output} with shape {tuple(embeddings.shape)}")
PY

### Generate Word2Vec embeddings if needed

Place GoogleNews-vectors-negative300.bin in ~/archive/ImageNet/ before running this step. Load the full vocabulary in a compute session with enough memory.

python - <<'PY'
import csv
import re
from pathlib import Path

import torch
from gensim.models import KeyedVectors

archive = Path.home() / "archive/ImageNet"
with Path("src/artifacts/data/imagenet_classes.csv").open(newline="") as file:
    classes = sorted(csv.DictReader(file), key=lambda row: int(row["label"]))
with (archive / "imagenet_class_names.csv").open(newline="") as file:
    names = {row["synset"]: row["name"].strip() for row in csv.DictReader(file)}

if [int(row["label"]) for row in classes] != list(range(1000)):
    raise SystemExit("Expected class IDs 0 through 999 in imagenet_classes.csv")

vectors = KeyedVectors.load_word2vec_format(
    str(archive / "GoogleNews-vectors-negative300.bin"), binary=True
)
rows = []

for row in classes:
    synset = row["synset"]
    label = names.get(synset)
    if not label:
        raise SystemExit(f"Missing class name for {synset}")

    tokens = re.findall(r"[A-Za-z0-9]+", label)
    phrase = "_".join(tokens)
    matches = [word for word in (phrase, phrase.lower()) if word in vectors][:1]
    if not matches:
        matches = [
            word for token in tokens
            for word in (token if token in vectors else token.lower(),)
            if word in vectors
        ]
    if not matches:
        raise SystemExit(f"No Word2Vec vocabulary match for {synset}: {label!r}")

    rows.append(torch.stack([torch.from_numpy(vectors[word]) for word in matches]).mean(0))

output = archive / "embeddings/word2vec.pt"
output.parent.mkdir(parents=True, exist_ok=True)
embeddings = torch.stack(rows)
torch.save(embeddings, output)
print(f"Wrote {output} with shape {tuple(embeddings.shape)}")
PY

## 6. Configure the embedding paths

Edit src/configs/cnn_embed_config.yaml. Place both file paths under model.embedding_paths and remove any duplicate top-level embedding_paths block. Use absolute paths. Keep the other model and training settings in the file.

To print a YAML fragment with the paths for your account, run:

cat <<EOF
model:
  backbone: "resnet18"
  weights: "IMAGENET1K_V1"
  embedding_paths:
    Word2Vec: "$HOME/archive/ImageNet/embeddings/word2vec.pt"
    BERT: "$HOME/archive/ImageNet/embeddings/bert.pt"
EOF

Before submitting BERT, verify that the config parses and points to a readable tensor.

python - <<'PY'
import os

from src.utils.load_config import load_config

path = load_config("cnn_embed_config")["model"]["embedding_paths"]["BERT"]
if not path or not os.path.isfile(path) or not os.access(path, os.R_OK):
    raise SystemExit(f"BERT tensor is missing or unreadable: {path!r}")
print(f"BERT tensor: {path}")
PY

## 7. Submit training

If you are still in the interactive compute session, run exit to return to the login node. Submit from the repo root. Run the command for the experiment you want to start.

cd ~/CS-541-Final-Project

# No-embedding baseline
sbatch --partition=academic scripts/train_cnn_noembed.sh

# BERT with cross-entropy
sbatch --partition=academic scripts/train_cnn_bert_ent.sh

Save the job ID printed by sbatch. The BERT launcher uses python/3.12.10 by default. If your virtual environment was built with a different module, supply that module at submission.

PYTHON_MODULE=python/3.12.10 sbatch scripts/train_cnn_bert_ent.sh

### Train on 30% of the training split

Use the existing ImageNet manifest from Step 4. Submit the baseline or BERT experiment from the repo root.

```bash
PYTHON_MODULE=python/3.12.10 TRAIN_FRACTION=0.30 \
  sbatch --partition=academic scripts/train_cnn_noembed.sh

TRAIN_FRACTION=0.30 sbatch --partition=academic scripts/train_cnn_bert_ent.sh
```

The trainer samples 30% of the training split with seed 42 and stratifies by ImageNet class. With the default 70/20/10 split, this uses about 21% of the manifest for training and evaluates on the original test split. The job log reports the image counts, and the model checkpoint records the fraction. The default fraction is 1.0.

## 8. Monitor training and find results

Replace 1234567 with the submitted job ID.

JOB_ID=1234567
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" -X --format=JobID,JobName%24,State,ExitCode,Elapsed
tail -n 30 "train-${JOB_ID}.out" "train-${JOB_ID}.err"

If a job disappears from squeue, use sacct to find its final state. To list jobs from yesterday onward, pass an explicit date.

sacct -u "$USER" -S "$(date -d yesterday +%F)" -X \
  --format=JobID,JobName%24,State,ExitCode,Elapsed

The trainer prints an Epoch line after completing both training and evaluation for that epoch. It writes the metrics and model after the final epoch.

| Run | Metrics | Model |
|---|---|---|
| Baseline | src/artifacts/results/cnn_noembed_<jobid>_metrics.json | src/artifacts/models/cnn_noembed_<jobid>_model.pt |
| BERT cross-entropy | src/artifacts/results/cnn_bert_ent_<jobid>_metrics.json | src/artifacts/models/cnn_bert_ent_<jobid>_model.pt |

The default configuration runs 20 epochs. Use the time to the first completed epoch to estimate whether the run fits within the 12-hour Slurm limit. For a short end-to-end check, lower training_args.epochs in the experiment’s config before submitting. A timeout before the final epoch leaves no saved metrics or model with the current trainer.

I used the Python 3.12.10 module from the existing guide as the launcher default. I haven’t run this launcher on Turing.
