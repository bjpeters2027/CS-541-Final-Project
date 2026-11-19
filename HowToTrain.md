# How to train our models on Turing

## 1. Connect and clone the repository

From your own computer, connect from the WPI network and use your WPI credentials when prompted. Replace YOUR_WPI_USERNAME with your username. Turing access instructions
```bash
ssh YOUR_WPI_USERNAME@turing.wpi.edu
```

On the Turing login node:
```bash
cd "$HOME"
git clone --branch dev https://github.com/bjpeters2027/CS-541-Final-Project.git
cd CS-541-Final-Project
```

If you already cloned dev, run git pull --ff-only from that checkout before making the personal config changes below.

## 2. Start a Slurm session and install dependencies

Request a compute shell from the login node. Use it for installs, downloads, extraction, and manifest generation. If the allocation expires, start another and reload the module and environment before continuing.
```bash
srun --partition=academic --ntasks=1 --cpus-per-task=2 \
  --mem=4G --time=03:00:00 --pty bash -l
```

Inside the allocation:

```bash
cd "$HOME/CS-541-Final-Project"
module load python/3.12.10
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```


## 3. Download and extract ImageNet

Accept the ImageNet competition rules in a browser on your own computer. Then authenticate from your Slurm shell with the command below and follow the prompts using your local browser. Ask the team whether there is already an approved, compute-readable shared copy before downloading another one. ImageNet competition data · Kaggle CLI authentication
```bash
kaggle auth login --no-launch-browser
```

Check your scratch quota before downloading. You’ll need room for both the ZIP and the extracted images.

```bash
export DATA_DIR="/scratch/$USER/ImageNet"
mkdir -p "$DATA_DIR"

kaggle competitions download \
  -c imagenet-object-localization-challenge \
  -p "$DATA_DIR"
```

If Slurm stops the download, retry in a new allocation using the same destination and omit --force so the CLI can attempt to resume. Once the download finishes, extract the training images with Python.

```bash
python -u - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile

root = Path(os.environ["DATA_DIR"])
prefix = "ILSVRC/Data/CLS-LOC/train/"
suffixes = {".jpeg", ".jpg", ".png"}

with ZipFile(root / "imagenet-object-localization-challenge.zip") as archive:
    entries = [
        entry for entry in archive.infolist()
        if entry.filename.startswith(prefix)
        and not entry.is_dir()
        and Path(entry.filename).suffix.lower() in suffixes
    ]
    if not entries:
        raise SystemExit(f"No training images found under {prefix}")

    print(f"Extracting {len(entries):,} training images")
    for count, entry in enumerate(entries, start=1):
        archive.extract(entry, root)
        if count % 50_000 == 0:
            print(f"Extracted {count:,} / {len(entries):,}")

print(f"Training images: {root / prefix}")
PY
```
## 4. Generate the manifests and check a batch

Still inside Slurm, run from the repository root. If you’re using a shared extraction, set TRAIN_DIR to its train directory.
```bash
cd "$HOME/CS-541-Final-Project"
export TRAIN_DIR="$DATA_DIR/ILSVRC/Data/CLS-LOC/train"

python - <<'PY'
import csv
import os
import re
from pathlib import Path

root = Path(os.environ["TRAIN_DIR"]).resolve(strict=True)
classes = sorted(
    p for p in root.iterdir()
    if p.is_dir() and re.fullmatch(r"n\d{8}", p.name)
)
if len(classes) != 1000:
    raise SystemExit(f"Expected 1,000 class folders in {root}, found {len(classes)}")

output = Path("src/artifacts/data")
output.mkdir(parents=True, exist_ok=True)
count = 0

with (
    (output / "imagenet_classes.csv").open("w", newline="") as class_file,
    (output / "imagenet_train.csv").open("w", newline="") as image_file,
):
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
        count += len(images)

print(f"Wrote {count:,} images across {len(classes):,} classes")
if count != 1_281_167:
    raise SystemExit("Unexpected image count for full ImageNet. Check extraction before training.")
PY
```
Both CNN configs on dev already use imagenet_train.csv. Keep the generated CSVs out of Git. They’re reproducible, and the image manifest contains paths specific to your Turing checkout.
```bash
python - <<'PY'
import torch
from src.utils.load_data import load_data

train, test, val = load_data("imagenet_train.csv", batch_size=8)
images, labels = next(iter(train))

assert tuple(images.shape) == (8, 3, 224, 224), images.shape
assert labels.dtype == torch.long, labels.dtype
assert ((labels >= 0) & (labels < 1000)).all(), labels

print("Batch passed:", tuple(images.shape), labels.tolist())
print("Split sizes:", len(train.dataset), len(test.dataset), len(val.dataset))
PY
```
These splits are sampled from the extracted ImageNet training images. Report the resulting metrics as image-level holdout results. The official validation set and zero-shot evaluation require separate splits.

## 5. Set up embeddings if you’re running BERT or Word2Vec

Skip this step for the baseline. For the embedding runs, use the team’s generated bert.pt and word2vec.pt if available. Check that their rows match imagenet_classes.csv in label order. Sharing the same tensors also keeps the label representations consistent across runs.

If you’re generating the tensors with the Python block in the draft, first obtain the team’s imagenet_class_names.csv with synset,name columns and the pretrained GoogleNews-vectors-negative300.bin file. The draft’s generator expects both files and needs access to BERT’s pretrained files on first use. Run it in Slurm using scratch paths and more memory than the 4 GB data-prep allocation for the Word2Vec load.

```bash
export CLASS_NAMES="$DATA_DIR/imagenet_class_names.csv"
export WORD2VEC_BIN="$DATA_DIR/GoogleNews-vectors-negative300.bin"
export EMBEDDING_DIR="$DATA_DIR/embeddings"
```

In src/configs/cnn_embed_config.yaml, remove the extra top-level embedding_paths block. Put your actual absolute paths in the existing block under model. YAML does not expand $USER in these strings. You can leave the other entry empty if you’re running only one embedding type.
```JSON
model:
  backbone: "resnet18"
  weights: "IMAGENET1K_V1"
  embedding_paths:
    Word2Vec: "/scratch/YOUR_WPI_USERNAME/ImageNet/embeddings/word2vec.pt"
    BERT: "/scratch/YOUR_WPI_USERNAME/ImageNet/embeddings/bert.pt"
```

In each embedding batch script you plan to use, change module load python to module load python/3.12.10 and include $SLURM_JOB_ID in EXPERIMENT_NAME. For example, the BERT cross-entropy script should use:
```bash
EXPERIMENT_NAME="cnn_bert_ent_${SLURM_JOB_ID}"
```

## 6. Submit training

Exit the interactive Slurm shell and submit from the project root on the login node. For the first baseline job, set training_args.epochs to 1 in src/configs/cnn_noembed_config.yaml. A full ImageNet epoch is still substantial. Use its runtime to decide whether the planned run fits the partition’s time limit.

```bash
exit
cd "$HOME/CS-541-Final-Project"
export PYTHON_MODULE=python/3.12.10

sbatch --partition=academic scripts/train_cnn_noembed.sh
```

Each model config requests pretrained ResNet-18 weights. The first run needs network access to download them or an existing Torch cache visible on the GPU node.

After the baseline succeeds, submit whichever embedding runs your team needs:

| Experiment | Submit from the repository root |
|---|---|
| BERT + cross-entropy | sbatch --partition=academic scripts/train_cnn_bert_ent.sh |
| BERT + cosine loss | sbatch --partition=academic scripts/train_cnn_bert_cos.sh |
| Word2Vec + cross-entropy | sbatch --partition=academic scripts/train_cnn_word2vec_ent.sh |
| Word2Vec + cosine loss | sbatch --partition=academic scripts/train_cnn_word2vec_cos.sh |

## 7. Monitor jobs and collect results

Write down the job ID printed by sbatch. From the repository root on the login node, replace YOUR_JOB_ID with that number:
```bash
JOB_ID=YOUR_JOB_ID
squeue -u "$USER"
tail -n 30 "train-$JOB_ID.out" "train-$JOB_ID.err"
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
```

PD means pending and R means running. A job can disappear from squeue after success or failure. Use the top-level job row in sacct to confirm COMPLETED and 0:0. Turing Slurm guide

For the baseline, verify both result files:
```bash
EXPERIMENT="cnn_noembed_${JOB_ID}"
ls -lh \
  "src/artifacts/results/${EXPERIMENT}_metrics.json" \
  "src/artifacts/models/${EXPERIMENT}_model.pt"
```

For an embedding run, use its EXPERIMENT_NAME instead. The JSON contains per-epoch training loss, test loss, and test accuracy. The .pt file contains the trained model checkpoint and its configuration. Both files are saved at the end of training. Share the metrics JSON with the team and record the job ID, Git commit (git rev-parse --short HEAD), and settings alongside it so others can reproduce the run.