#!/usr/bin/env bash
#SBATCH --job-name=cs541_bert_ent
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"

module load "${PYTHON_MODULE:-python/3.12.10}"
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python - <<'PY'
from pathlib import Path

import torch

from src.utils.enums.path_enums import Paths
from src.utils.load_config import load_config

config = load_config("cnn_embed_config")
manifest = Path(Paths.DATA) / config["data"]["datafile"]
bert_path = config["model"]["embedding_paths"]["BERT"]

if not manifest.is_file():
    raise SystemExit(f"Missing ImageNet manifest: {manifest}")
if not bert_path or not Path(bert_path).is_file():
    raise SystemExit(f"Missing BERT tensor: {bert_path!r}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable")

embeddings = torch.load(bert_path, map_location="cpu", weights_only=True)
if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (1000, 768):
    raise SystemExit(f"Expected a BERT tensor with shape (1000, 768): {bert_path}")

print(f"Manifest: {manifest}")
print(f"BERT tensor: {bert_path}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python -u -m src.train \
    "cnn_bert_ent_${SLURM_JOB_ID}" \
    cnn_embed \
    2 \
    1