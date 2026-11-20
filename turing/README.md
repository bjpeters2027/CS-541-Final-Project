# CNN embedding model

The embedding model uses the same frozen pretrained ResNet-18 as the baseline
and trains a linear projection into a fixed Word2Vec or BERT label space.
Only the projection is trainable; backbone BatchNorm statistics remain frozen.

Each label lookup table is a floating-point tensor with shape
`(number_of_candidate_classes, embedding_dimension)` saved with `torch.save`.
Row `i` corresponds to seen class `i` in `splits/seen.json` order.
`scripts/06_build_label_embeddings.py` writes them as `seen_Word2Vec.pt` and
`seen_BERT.pt` from frozen text models. Rows must be finite and nonzero. The
module normalizes them and stores them as a non-trainable buffer in its state
dictionary.

`model.embedding_paths.Word2Vec` and `model.embedding_paths.BERT` in
[the embedding configuration](src/configs/cnn_embed_config.yaml) point to those
files, relative to the data prep `embeddings/` folder unless absolute.
Initialization fails explicitly for an empty path; it does not substitute random
embeddings or download a text model.
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

# Data preparation

Run the scripts from the project root, in order. Scripts 01 and 03 download
data, so run them from a login node. Downloads go to `/scratch/$USER/raw` and
derived files (manifests, splits, embeddings) to `data/`; override these with
`DATAPREP_RAW_ROOT` and `DATAPREP_DATA_ROOT`.

1. `scripts/01_fetch_imagenet.py`: packs the ImageNet validation images and
   chooses the 800 seen and 200 unseen classes. Each seen class gets 30 train, 10 validation, and 10
   test_seen images; every unseen-class image is test_unseen. Needs a Hugging
   Face account that has accepted the `ILSVRC/imagenet-1k` terms: run
   `hf auth login` once, or `export HF_TOKEN=...` (a `.env` file is not loaded
   automatically). No other script needs credentials.
2. `scripts/02_build_imagenet_manifest.py`: indexes those images and partitions.
3. `scripts/03_fetch_domain_datasets.py`: stages 20 images per class of AWA2,
   CUB, and SUN, plus the Xian et al. proposed split that marks which classes are
   unseen, for transfer evaluation. Move or delete older full archives first.
4. `scripts/04_build_domain_manifests.py`: indexes the domain images.
5. `scripts/05_build_class_splits.py`: writes the seen/unseen class splits.
6. `scripts/06_build_label_embeddings.py`: builds the Word2Vec and BERT label embeddings.

Train with `python experiment.py <cnn_embed|cnn_noembed> train` (or the
`scripts/train_*.sh` jobs), and evaluate a saved model with
`python -m src.evalutate <model_name>`. Evaluation reports seen-class accuracy
for every model, plus unseen-class, generalized zero-shot, and AWA2/CUB/SUN
transfer accuracy for embedding models.
