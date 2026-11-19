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

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"

if [[ -n "${PYTHON_MODULE:-}" ]]; then
    module load "$PYTHON_MODULE"
fi

source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.cuda.get_device_name(0))'

srun python -u -m src.train \
    "cnn_noembed_${SLURM_JOB_ID}" \
    cnn_noembed \
    0 \
    0