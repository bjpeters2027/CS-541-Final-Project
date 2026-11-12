"""Build frozen Word2Vec + BERT embeddings for every class label across all splits.

Per Methods.md, both embedding models stay frozen for the rest of the project,
so this is the only place either one is ever loaded -- training code just
reads the resulting lookup table (embeddings/label_embeddings.npz).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BERT_MODEL_NAME, BERT_TEMPLATE, EMBEDDINGS_DIR, SPLITS_DIR, WORD2VEC_MODEL_NAME


def load_all_labels() -> list[str]:
    labels: set[str] = set()
    for path in SPLITS_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        classes = data["classes"] if isinstance(data, dict) and "classes" in data else data
        for c in classes:
            labels.add(c["class_name"])
    return sorted(labels)


def build_word2vec_embeddings(labels: list[str]) -> dict[str, np.ndarray]:
    import gensim.downloader as gensim_api

    model = gensim_api.load(WORD2VEC_MODEL_NAME)
    embeddings: dict[str, np.ndarray] = {}
    oov = []
    for label in labels:
        words = [w.lower() for w in label.replace(",", " ").split() if w]
        vectors = [model[w] for w in words if w in model]
        if not vectors:
            oov.append(label)
            continue
        embeddings[label] = np.mean(vectors, axis=0)
    if oov:
        print(f"word2vec OOV ({len(oov)}): {oov}")
    return embeddings


def build_bert_embeddings(labels: list[str]) -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for label in labels:
            text = BERT_TEMPLATE.format(label=label)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            embeddings[label] = pooled.squeeze(0).cpu().numpy()
    return embeddings


def main() -> None:
    labels = load_all_labels()
    if not labels:
        raise RuntimeError(f"No labels found in {SPLITS_DIR}. Run scripts/05 first.")

    word2vec = build_word2vec_embeddings(labels)
    bert = build_bert_embeddings(labels)

    missing_bert = [l for l in labels if l not in bert]
    if missing_bert:
        raise RuntimeError(f"BERT embedding failed for: {missing_bert}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBEDDINGS_DIR / "label_embeddings.npz",
        labels=np.array(labels, dtype=object),
        word2vec=np.array([word2vec.get(l) for l in labels], dtype=object),
        bert=np.array([bert[l] for l in labels], dtype=object),
    )

    missing_w2v = sum(1 for l in labels if l not in word2vec)
    print(f"embedded {len(labels)} labels ({missing_w2v} missing word2vec/OOV) -> {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
