# CNN embedding model

The embedding model uses the same frozen pretrained ResNet-18 as the baseline
and trains a linear projection into a fixed Word2Vec or BERT label space.
Only the projection is trainable; backbone BatchNorm statistics remain frozen.

Prepare each label lookup table as a floating-point tensor with shape
`(number_of_candidate_classes, embedding_dimension)` and save it with
`torch.save(tensor, path)`. Row `i` must correspond to dataset class ID `i`.
Use embeddings from one frozen text model consistently for training and evaluation.
Rows must be finite and nonzero. The module normalizes them and stores them as a
non-trainable buffer in its state dictionary.

Set `model.embedding_paths.Word2Vec` and/or `model.embedding_paths.BERT` in
[the embedding configuration](src/configs/cnn_embed_config.yaml) to those files.
Paths are relative to the working directory unless absolute. They are intentionally
empty until label embeddings are supplied. Initialization fails explicitly for an
empty path; it does not substitute random embeddings or download a text model.
Default ResNet weights may download on first use; set `model.weights` to `null`
for offline architecture tests with an untrained backbone.

`initialize_model(Embedding.WORD2VEC, config)` (or `Embedding.BERT`) builds the model.
`model(images)` returns cosine-similarity logits for cross-entropy training and
classification. `model.encode_images(images)` returns normalized projected vectors.
The training loop uses these vectors and the corresponding label rows for
`CosineEmbeddingLoss`, with positive-pair targets.

For zero-shot evaluation, call `model(images, candidate_embeddings)` with another
ordered label tensor from the same text embedding space, including unseen labels.
The candidate count may differ from training; the embedding dimension must match.
Predicted indices refer to rows of the supplied vocabulary. This does not retrain
or change the stored training vocabulary.

Run the embedding tests from the project root (PowerShell):

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cnn_embed.py
```

## Training and Optuna hyperparameter tuning

Install dependencies into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Training now reads the archive-backed ImageNet parquet manifest and `seen.json`
produced by the data preparation scripts. Configure `data.manifest` and
`data.seen_classes` as explicit paths, or filenames resolved under
`config.MANIFESTS_DIR` and `config.SPLITS_DIR`. Class IDs follow the **order of
entries in `seen.json`**, so embedding tensor rows must follow that same order.
Set the baseline's `model.num_classes` to the number of seen classes.

Split generation and validation compare classes by WordNet ID when both entries
have one. Different ImageNet synsets can share a display name (for example,
the Cardigan dog and cardigan clothing); that alone is not class leakage.
When either entry lacks an ID, as in the domain dataset splits, normalized
class names are compared instead. Genuine overlap raises an explicit error.
This does not disambiguate labels for text embeddings: the label-name-keyed
embedding preparation still needs a separate disambiguation policy before
experiments involving these ambiguous labels.

The official ImageNet `val` partition is reserved for final testing.
`data.val_split` holds out a seeded fraction of each seen class's **training**
images for validation (at least one image each for training and validation).
Unseen classes never enter these loaders. Each class needs at least two training
images. Training uses random crops/flips; validation and testing are deterministic.

Optuna minimizes **last-epoch validation loss** using TPE sampling. Each trial
starts a fresh model with the same seed and image partitions. Epoch count stays
fixed at `training_args.epochs`; only learning rate (log scale) and batch size are
tuned. Defaults are 20 trials, learning rates from `1e-5` to `1e-2`, and batch sizes
16, 32, or 64. Trials run sequentially and never evaluate test images.

Run from the repository root:

```powershell
# Baseline model
.\.venv\Scripts\python.exe -m src.tune cnn_noembed --trials 20 --output-dir src\artifacts\tuning\baseline

# Embedding model: 1=Word2Vec, 2=BERT; loss 1=cross entropy, 2=cosine
.\.venv\Scripts\python.exe -m src.tune cnn_embed --embedding 1 --loss 2 --trials 20 --output-dir src\artifacts\tuning\word2vec_cosine
```

Use `--config path\to\config.yaml` for a custom configuration, `--lr-min` and
`--lr-max` for bounds, and `--batch-sizes 16 32` to restrict GPU memory use.
Failed trials surface their errors and stop the command; they are not assigned
fabricated scores. Choose a separate study for each model/embedding/loss setup;
validation losses from different loss functions are not directly comparable.

Each output directory contains:

- `study.db`: persistent Optuna SQLite study, including failed trials.
- `trials.csv`: trial parameters, scores, and states after a successful run.
- `best_trial.json`: winning parameters and validation metrics.
- `best_config.yaml`: original configuration with the winning learning rate and
  batch size substituted. Source configurations are not overwritten.

Repeating an identical command resumes the study and runs **additional** trials.
Changing its configuration/search space requires another output directory or
`--study-name`. Keep the referenced manifests, splits, and embeddings unchanged
when resuming. Seeded runs preserve splits and initialization; GPU kernels and
resumed sampler sequences are not guaranteed to match an uninterrupted run.

Finally, train with the winning configuration and evaluate the held-out test set
once, after training:

```powershell
.\.venv\Scripts\python.exe -m src.train baseline_best cnn_noembed --config src\artifacts\tuning\baseline\best_config.yaml
.\.venv\Scripts\python.exe -m src.train word2vec_best cnn_embed 1 2 --config src\artifacts\tuning\word2vec_cosine\best_config.yaml
```

Normal training writes a checkpoint under `src/artifacts/models` and metrics under
`src/artifacts/results`. Per-epoch metrics are now `train_losses`, `val_losses`,
and `val_accuracies`; final test metrics are the scalars `test_loss` and
`test_accuracy`, replacing the old per-epoch `test_losses`/`test_accuracies`.
Tuning does not save model checkpoints; the final command creates one.

Run model, loader, and tuning tests (synthetic images; no model downloads):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```