
#!/usr/bin/env bash

#SBATCH --job-name=cs541_train
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

set -euo pipefail

# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

MODEL_NAME="cnn_embed"
EXPERIMENT_NAME="cnn_bert_cos_MM-DD-YYYY"
EMBEDDING_MODEL="bert"
LOSS_FUNCTION="cosine"

# ==========================================================
# CONVERT CONFIGURATION TO NUMERIC ARGUMENTS
# ==========================================================

case "$EMBEDDING_MODEL" in
    word2vec)
        EMBEDDING_CHOICE=1
        ;;
    bert)
        EMBEDDING_CHOICE=2
        ;;
    *)
        echo "Invalid embedding model: $EMBEDDING_MODEL" >&2
        exit 1
        ;;
esac

case "$LOSS_FUNCTION" in
    cross_entropy)
        LOSS_CHOICE=1
        ;;
    cosine)
        LOSS_CHOICE=2
        ;;
    *)
        echo "Invalid loss function: $LOSS_FUNCTION" >&2
        exit 1
        ;;
esac

# ==========================================================
# ENVIRONMENT SETUP
# ==========================================================

cd "${SLURM_SUBMIT_DIR:?Submit from project root}"

module load python

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ==========================================================
# JOB INFORMATION
# ==========================================================

echo "=========================================="
echo "SLURM JOB INFORMATION"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Working directory: $PWD"
echo "Model: $MODEL_NAME"
echo "Experiment: $EXPERIMENT_NAME"
echo "Embedding: $EMBEDDING_MODEL"
echo "Loss: $LOSS_FUNCTION"
echo "=========================================="

python --version
nvidia-smi

# ==========================================================
# VERIFY GPU
# ==========================================================

python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
PY

# ==========================================================
# RUN TRAINING
# ==========================================================

srun python -u -m src.train \
    "$EXPERIMENT_NAME" \
    "$MODEL_NAME" \
    "$EMBEDDING_CHOICE" \
    "$LOSS_CHOICE"

echo "Training completed successfully."
