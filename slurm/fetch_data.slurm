#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=8g
#SBATCH --job-name dataprep-fetch
#SBATCH --partition short
#SBATCH --time 0-04:00:00
#SBATCH --output logs/%x-%j.out
# TODO: add --account=<your_slurm_account> if your course allocation requires one
# (check with: sacctmgr show user $USER)

# NOTE: many HPC clusters block outbound internet from compute nodes. If this
# job fails to reach Kaggle/academictorrents/mpi-inf.mpg.de, run
# scripts/01_fetch_imagenet.py and scripts/03_fetch_domain_datasets.py directly
# from a login-node shell instead of via sbatch.

module load uv

uv run scripts/01_fetch_imagenet.py
uv run scripts/03_fetch_domain_datasets.py
