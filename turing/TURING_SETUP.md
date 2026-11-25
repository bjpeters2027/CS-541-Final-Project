# Turing setup

This folder is a snapshot of the repository, with the original source files,
comments, configurations, and training scripts unchanged. It contains the four
completed image archives, class splits, proposed split lists, label embeddings,
and the portable pretrained ResNet-18 checkpoint.

The Mac .venv, Git metadata, credentials, Python caches, Mac metadata, download
intermediates, and old manifests are excluded. The data/manifests directory is
intentionally empty until it is rebuilt on Turing.

## 1. Upload from your Mac

Run these commands from the original repository root on your Mac. They upload
the CONTENTS of turing/ into Final_Project, rather than creating another turing/
directory inside Final_Project. Image archives go directly to scratch.

    ssh trgarofalo@turing.wpi.edu 'mkdir -p /home/trgarofalo/Deep_Learning/Final_Project /scratch/trgarofalo/raw'

    rsync -avP --exclude='data/raw/' ./turing/ \
        trgarofalo@turing.wpi.edu:/home/trgarofalo/Deep_Learning/Final_Project/

    rsync -avP ./turing/data/raw/ \
        trgarofalo@turing.wpi.edu:/scratch/trgarofalo/raw/

These commands do not delete existing remote files. Old manifests at the
destination will be overwritten by the indexing commands below.

If you already uploaded the entire bundle to Final_Project, you can stage its
archives on the login node instead:

    mkdir -p /scratch/trgarofalo/raw
    rsync -avP /home/trgarofalo/Deep_Learning/Final_Project/data/raw/ \
        /scratch/trgarofalo/raw/

The scratch location must contain these files:

    /scratch/trgarofalo/raw/imagenet/imagenet_subset.zip
    /scratch/trgarofalo/raw/awa2/awa2_subset.zip
    /scratch/trgarofalo/raw/cub/cub_subset.tar
    /scratch/trgarofalo/raw/sun/sun_subset.tar

Keep the archives intact; image loading reads their members directly.

## 2. Login node: create the Linux environment

Do this on the login node, before opening sinteractive. Dependency installation
requires internet access and does not require a GPU allocation.

The --clear option below replaces an existing .venv, including a Mac environment
that might have been copied over previously. It does not change source or data.

    cd /home/trgarofalo/Deep_Learning/Final_Project
    module load python
    module load cuda
    module load uv

    uv venv --python 3.11 --clear .venv
    source .venv/bin/activate

    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    uv pip install -r requirements.txt
    uv pip check

    python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda); assert torch.version.cuda is not None'

The first installation command selects CUDA-enabled PyTorch wheels. A CUDA build
is required even though the login node itself may report no available GPU.
Do not use a CPU-only PyTorch installation for training.

## 3. Login node: install the included ResNet checkpoint and rebuild manifests

The following copies the included checkpoint into the cache used by the current
Turing environment. Training will then find it without a network download.

    python - <<'PY'
    from pathlib import Path
    import shutil
    import torch

    source = Path("data/pretrained/resnet18-f37072fd.pth")
    destination = Path(torch.hub.get_dir()) / "checkpoints" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print("Pretrained checkpoint ready:", destination)
    PY

Rebuild the lightweight indexes so archive paths refer to Turing:

    export DATAPREP_RAW_ROOT="/scratch/trgarofalo/raw"
    export DATAPREP_DATA_ROOT="$PWD/data"

    python scripts/02_build_imagenet_manifest.py
    python scripts/04_build_domain_manifests.py

Expected outputs:

- ImageNet: 50,000 images, with 24,000 train, 8,000 validation,
  8,000 test_seen, and 10,000 test_unseen images.
- AWA2: 1,000 images across 50 classes; 10 proposed-split unseen classes.
- CUB: 4,000 images across 200 classes; 50 proposed-split unseen classes.
- SUN: 14,346 images across 723 parsed classes; 72 proposed-split unseen classes.

The copied splits already contain 800 seen and 200 unseen ImageNet classes.
The copied embeddings cover all 1,132 candidate labels with no missing vectors.
The training tensors have shapes (800, 300) and (800, 768), in seen.json order.

Scripts 01, 03, 05, and 06 do not need to be rerun. Dataset credentials and the
full Word2Vec/BERT download caches are unnecessary for this prepared data.

## 4. Compute node: verify the GPU and test training

Start this from the project root on the login node. This explicitly requests
one GPU, four CPUs, and 32 GB RAM for a 20-minute interactive session:

    cd /home/trgarofalo/Deep_Learning/Final_Project
    srun --partition=academic --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=32G --gres=gpu:1 --time=00:20:00 --pty bash

If you already have a GPU sinteractive session, use that session instead.
The SBATCH comments inside the training scripts do not allocate resources when
a script is executed with bash inside an existing interactive session.

Inside the compute-node shell:

    cd /home/trgarofalo/Deep_Learning/Final_Project
    module load python
    module load cuda
    source .venv/bin/activate

    python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.cuda.get_device_name(0))'
    PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p 'test_cnn_*.py'

    SLURM_SUBMIT_DIR="$PWD" bash scripts/train_cnn_bert_cos.sh

Expect CUDA available: True, Device: cuda, 750 training batches, and 250 validation
batches. The progress bars show that images are being loaded and processed.

This runs the normal 20-epoch training path within your interactive allocation.
For a startup check, stop with Ctrl+C once progress is confirmed, then type exit
to release the interactive session. Checkpoints are saved only after all epochs;
an interrupted startup check does not save a partially trained model.

You can also start the script directly from the login node with bash; it requests
a ten-minute interactive GPU allocation automatically.

## 5. Login node: submit the full training job

After leaving the interactive session:

    cd /home/trgarofalo/Deep_Learning/Final_Project
    sbatch scripts/train_cnn_bert_cos.sh
    squeue -u trgarofalo

Choose the matching script for each experiment:

| Script | Model | Loss |
| --- | --- | --- |
| scripts/train_cnn_bert_cos.sh | BERT projection | Cosine embedding |
| scripts/train_cnn_bert_ent.sh | BERT projection | Cross-entropy |
| scripts/train_cnn_word2vec_cos.sh | Word2Vec projection | Cosine embedding |
| scripts/train_cnn_word2vec_ent.sh | Word2Vec projection | Cross-entropy |
| scripts/train_cnn_noembed.sh | Classifier baseline | Cross-entropy |

Batch scripts request one GPU, four CPUs, and 32 GB RAM for up to twelve hours.
They train for the 20 epochs configured in src/configs/. Each run gets a timestamped
experiment name. A full completed run writes its checkpoint to src/artifacts/models/
and its training/validation metrics to src/artifacts/results/.

Batch stdout goes to train-JOBID.out; progress bars and stderr go to
train-JOBID.err. For example, replace JOBID with the submitted job number:

    tail -f train-JOBID.out train-JOBID.err

## References

- [Turing Python environments](https://docs.turing.wpi.edu/software/python/)
- [Turing package management](https://docs.turing.wpi.edu/best-practices/python_package_management/)
- [Turing storage](https://docs.turing.wpi.edu/best-practices/storing_data/)
- [Turing interactive and batch jobs](https://docs.turing.wpi.edu/getting-started/how_to_run/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)

