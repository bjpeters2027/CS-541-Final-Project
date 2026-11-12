#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=16g
#SBATCH --job-name dataprep-manifests
#SBATCH --partition short
#SBATCH --time 0-06:00:00
#SBATCH --output logs/%x-%j.out
# TODO: add --account=<your_slurm_account> if required (sacctmgr show user $USER)

module load uv

uv run scripts/02_build_imagenet_manifest.py
uv run scripts/04_build_domain_manifests.py
uv run scripts/05_build_class_splits.py
