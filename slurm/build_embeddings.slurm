#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=16g
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100"
#SBATCH --job-name dataprep-embeddings
#SBATCH --partition short
#SBATCH --time 0-02:00:00
#SBATCH --output logs/%x-%j.out
# TODO: add --account=<your_slurm_account> if required (sacctmgr show user $USER)

module load python
module load cuda
module load uv

uv run scripts/06_build_label_embeddings.py
uv run scripts/07_validate_dataprep.py
