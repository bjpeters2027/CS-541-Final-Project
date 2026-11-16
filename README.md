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