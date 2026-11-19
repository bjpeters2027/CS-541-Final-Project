# How to train our model on Turing

## Step 1: SSHing into Turing
Run this command on a local terminal
```bash
ssh <wpi_username>@turing.wpi.edu
```
This will ask you for a password, this is your WPI account password

## Step 2: Clone the Repo
Run this command in your Turing terminal
```bash
git clone --branch dev https://github.com/bjpeters2027/CS-541-Final-Project.git
```

## Step 3: Download the Data
Follow these steps to download the ImageNet Data

### Step 3.1: Spin up a Slurm Terminal
Run this command in your Turing terminal to start a slurm session. this is to avoid downloading all the data on the login partition
```bash
srun --partition=academic --ntasks=1 --cpus-per-task=2 --mem=4G --time=03:00:00 --pty bash -l
```

### Step 3.2: Prep Kaggle
Go into the the repo
```bash
cd CS-541-Final-Project/
```
Then, create a virtual environment on the right python version:

First load python 3.12.10:
```bash
module load python/3.12.10
```
Then, create and source your venv
```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

Install the necessary packages
```bash
pip install -r ./requirements.txt
```

Finally, authenticate Kaggle using this command (I reccomend going with option A from the output)
```bash
kaggle auth login
```

### Step 3.3: Create the Data Directory
Run these commands to make the directories for the data. Turing docs say to use ```~/archive``` for data storage
```bash
ARCHIVE_DIR="/home/bjpeters/archive"
DATA_DIR="$ARCHIVE_DIR/ImageNet"
ZIP="$DATA_DIR/imagenet-object-localization-challenge.zip"
mkdir -p "$DATA_DIR"
```

### Step 3.4: Download the Data (fr this time)
This command will finally download the data 
```bash
kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p "$DATA_DIR"
```

This took about 30 minutes

Now you need to unzip it but turing doesnt have the unzip function so you need to run this:
```bash
python -u - <<'PY'
from pathlib import Path
from zipfile import ZipFile

archive = Path.home() / "archive" / "ImageNet"
zip_path = archive / "imagenet-object-localization-challenge.zip"
prefix = "ILSVRC/Data/CLS-LOC/train/"

with ZipFile(zip_path) as z:
    members = [
        entry for entry in z.infolist()
        if entry.filename.startswith(prefix)
        and not entry.is_dir()
        and Path(entry.filename).suffix.lower() in {".jpeg", ".jpg", ".png"}
    ]

    if not members:
        print("First 20 archive entries:")
        for entry in z.infolist()[:20]:
            print(entry.filename)
        raise SystemExit(f"No training images found under {prefix}")

    print(f"Extracting {len(members):,} images to {archive}")
    for count, entry in enumerate(members, start=1):
        z.extract(entry, archive)
        if count % 50_000 == 0:
            print(f"Extracted {count:,} / {len(members):,}")

print(f"Training folder: {archive / prefix}")
PY
```
This will also take a while btw


## Step 4: Prep for Training
To get ready for training, we need to make the manifest and class order. Run this from the repo root

```bash
export TRAIN_DIR="$HOME/archive/ImageNet/ILSVRC/Data/CLS-LOC/train"

python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["TRAIN_DIR"]).resolve(strict=True)
classes = sorted(
    p for p in root.iterdir()
    if p.is_dir() and p.name.startswith("n") and p.name[1:].isdigit()
)
if len(classes) != 1000:
    raise SystemExit(f"Expected 1,000 ImageNet classes in {root}, found {len(classes)}")

output = Path("src/artifacts/data")
output.mkdir(parents=True, exist_ok=True)

with (output / "imagenet_classes.csv").open("w", newline="") as class_file, \
     (output / "imagenet_train.csv").open("w", newline="") as image_file:
    class_writer = csv.writer(class_file)
    image_writer = csv.writer(image_file)
    class_writer.writerow(["label", "synset"])
    image_writer.writerow(["image_path", "label"])

    for label, folder in enumerate(classes):
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if not images:
            raise SystemExit(f"No images found in {folder}")
        class_writer.writerow([label, folder.name])
        image_writer.writerows((str(image), label) for image in images)

print(f"Wrote manifests to {output}")
PY
```

Now run this:
```bash
export CLASS_NAMES="$HOME/archive/ImageNet/imagenet_class_names.csv"
export WORD2VEC_BIN="$HOME/archive/ImageNet/GoogleNews-vectors-negative300.bin"
export EMBEDDING_DIR="$HOME/archive/ImageNet/embeddings"

python - <<'PY'
import csv
import os
import re
from pathlib import Path

import torch
from gensim.models import KeyedVectors
from transformers import AutoModel, AutoTokenizer

with open("src/artifacts/data/imagenet_classes.csv", newline="") as handle:
    classes = sorted(csv.DictReader(handle), key=lambda row: int(row["label"]))

with open(os.environ["CLASS_NAMES"], newline="") as handle:
    names = {row["synset"]: row["name"].strip() for row in csv.DictReader(handle)}

missing = [row["synset"] for row in classes if not names.get(row["synset"])]
if missing:
    raise SystemExit(f"Missing names for {len(missing)} classes, including {missing[:5]}")

labels = [names[row["synset"]] for row in classes]
output = Path(os.environ["EMBEDDING_DIR"])
output.mkdir(parents=True, exist_ok=True)

vectors = KeyedVectors.load_word2vec_format(os.environ["WORD2VEC_BIN"], binary=True)
word2vec_rows = []
for row, label in zip(classes, labels):
    tokens = re.findall(r"[A-Za-z0-9]+", label)
    phrase = "_".join(tokens)
    candidates = [
        word for word in (phrase, phrase.lower()) if word in vectors
    ][:1]
    if not candidates:
        candidates = [
            word for token in tokens
            for word in (token if token in vectors else token.lower(),)
            if word in vectors
        ]
    if not candidates:
        raise SystemExit(f"No Word2Vec vocabulary match for {row['synset']}: {label!r}")
    word2vec_rows.append(
        torch.stack([torch.from_numpy(vectors[word]) for word in candidates]).mean(0)
    )

torch.save(torch.stack(word2vec_rows), output / "word2vec.pt")
del vectors

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()
bert_rows = []

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
        bert_rows.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))

torch.save(torch.cat(bert_rows), output / "bert.pt")
print(f"Wrote embedding tensors to {output}")
PY
```

Now test a batch before running the whole thing:
```bash
python - <<'PY'
from src.utils.load_data import load_data

train, test, val = load_data("imagenet_train.csv", batch_size=8)
images, labels = next(iter(train))
assert tuple(images.shape) == (8, 3, 224, 224), images.shape
assert int(labels.min()) >= 0 and int(labels.max()) < 1000, labels
print("Batch:", images.shape, labels.tolist())
print("Split sizes:", len(train.dataset), len(test.dataset), len(val.dataset))
PY
```