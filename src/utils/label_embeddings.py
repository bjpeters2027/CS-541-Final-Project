import json
from pathlib import Path

import numpy as np
import torch

from src.utils.enums.embedding_enums import Embedding


def load_class_split(path):
    """
    Load a class split written by scripts/05_build_class_splits.py.

    Args:
        path (str | Path): Path to seen.json or an unseen_*.json file.

    Returns:
        list[dict]: Ordered class entries, each with a "class_name" (and "wnid" for ImageNet).
    """
    data = json.loads(Path(path).read_text())
    return data["classes"] if isinstance(data, dict) else data


def load_label_embeddings(archive_path, embedding, class_names):
    """
    Look up frozen text embeddings for class names in the archive written by
    scripts/06_build_label_embeddings.py.

    Args:
        archive_path (str | Path): Path to label_embeddings.npz.
        embedding (Embedding | str): Which text embedding model's vectors to use.
        class_names (list[str]): Class names, in the order the rows should follow.

    Returns:
        tuple[torch.Tensor, list[str]]: A (num_kept, embedding_dim) tensor and the kept class
        names. Names without a vector (Word2Vec out-of-vocabulary) are skipped.
    """
    embedding = Embedding(embedding)
    archive = np.load(archive_path, allow_pickle=True)
    vectors = dict(zip(archive["labels"], archive[embedding.value.lower()]))
    kept = [name for name in class_names if vectors.get(name) is not None]
    missing = [name for name in class_names if vectors.get(name) is None]
    if missing:
        print(f"warning: no {embedding.value} embedding for {len(missing)} classes, skipped: {missing}")
    rows = np.stack([np.asarray(vectors[name], dtype=np.float32) for name in kept])
    return torch.from_numpy(rows), kept
