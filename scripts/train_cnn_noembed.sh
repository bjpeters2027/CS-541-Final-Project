#!/usr/bin/env bash
#SBATCH --job-name=cs541_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

set -euo pipefail

# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

MODEL_NAME="cnn_noembed"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cnn_noembed_$(date +%m-%d-%Y_%H-%M-%S)}"
EMBEDDING_MODEL="N/A"
LOSS_FUNCTION="N/A"

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
    N/A)
        # cnn_noembed has no embedding model and always uses cross-entropy.
        EMBEDDING_CHOICE=0
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
    N/A)
        # cnn_noembed has no embedding model and always uses cross-entropy.
        LOSS_CHOICE=0
        ;;
    *)
        echo "Invalid loss function: $LOSS_FUNCTION" >&2
        exit 1
        ;;
esac

# ==========================================================
# ENVIRONMENT SETUP
# ==========================================================

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Requesting a 10-minute interactive GPU allocation; press Ctrl+C after verifying training starts."
    exec srun --partition=academic --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --gres=gpu:1 --time=00:10:00 --pty \
        bash "$PWD/scripts/$(basename "${BASH_SOURCE[0]}")" "$@"
fi

module load python
module load cuda
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"

if [[ -n "${PYTHON_MODULE:-}" ]]; then
    module load "$PYTHON_MODULE"
fi

source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.cuda.get_device_name(0))'

python -u -m src.train \
    "$EXPERIMENT_NAME" \
    "$MODEL_NAME" \
    "$EMBEDDING_CHOICE" \
    "$LOSS_CHOICE"

echo "Training completed successfully."
srun python -u -m src.train \
    "cnn_noembed_${SLURM_JOB_ID}" \
    cnn_noembed \
    0 \
    0
