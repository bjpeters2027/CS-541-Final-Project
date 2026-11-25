"""Build frozen Word2Vec + BERT embeddings for every class label across all splits.

Per Methods.md, both embedding models stay frozen for the rest of the project,
so this is the only place either one is ever loaded. Writes:
  - embeddings/label_embeddings.npz: every label, for zero-shot evaluation
  - embeddings/seen_<Word2Vec|BERT>.pt: one row per seen class, in seen.json
    order, which CNNEmbed loads as its fixed training vocabulary
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BERT_MODEL_NAME, BERT_TEMPLATE, EMBEDDINGS_DIR, SPLITS_DIR, WORD2VEC_MODEL_NAME
from src.utils.enums.embedding_enums import Embedding
from src.utils.label_embeddings import load_class_split, load_label_embeddings

# Hand-written Word2Vec text for classes whose name and synonyms are all missing
# from the GoogleNews vocabulary (agreed as a group). BERT still embeds the class
# name itself.
WORD2VEC_TEXT_OVERRIDES = {
    "coucal": "ground cuckoo",
    "jacamar": "iridescent tropical bird",
    "kuvasz": "Hungarian sheepdog",
    "EntleBucher": "Swiss cattle dog",
    "cuirass": "steel body armor",
    "pickelhaube": "spiked helmet",
    "agaric": "gilled mushroom",
    "gyromitra": "false morel",
    "limpkin": "wading bird",
    "dowitcher": "long-billed shorebird",
    "groenendael": "Belgian sheepdog",
    "Sayornis": "tyrant flycatcher",  # CUB
}


def load_all_labels() -> dict[str, list[str]]:
    """Returns {class name: alternative names}; only ImageNet classes have alternatives."""
    labels: dict[str, list[str]] = {}
    for path in SPLITS_DIR.glob("*.json"):
        for c in load_class_split(path):
            synonyms = c.get("synonyms") or ""
            labels[c["class_name"]] = [s.strip() for s in synonyms.split(",") if s.strip()]
    return dict(sorted(labels.items()))


def word2vec_vectors(model, name: str) -> list[np.ndarray]:
    # GoogleNews vectors are case-sensitive; try each word as written, lowercased, then capitalized.
    vectors = []
    for word in re.findall(r"[A-Za-z0-9]+", name):
        for form in (word, word.lower(), word.capitalize()):
            if form in model:
                vectors.append(model[form])
                break
    return vectors


def build_word2vec_embeddings(labels: dict[str, list[str]]) -> dict[str, np.ndarray]:
    import gensim.downloader as gensim_api

    model = gensim_api.load(WORD2VEC_MODEL_NAME)
    embeddings: dict[str, np.ndarray] = {}
    oov = []
    for label, synonyms in labels.items():
        # Use the first of the class name, its synonyms, and its override that has any in-vocabulary word.
        names = [label, *synonyms]
        if label in WORD2VEC_TEXT_OVERRIDES:
            names.append(WORD2VEC_TEXT_OVERRIDES[label])
        for name in names:
            vectors = word2vec_vectors(model, name)
            if vectors:
                break
        if not vectors:
            oov.append(label)
            continue
        embeddings[label] = np.mean(vectors, axis=0)
    if oov:
        print(f"word2vec OOV ({len(oov)}): {oov}")
    return embeddings


def build_bert_embeddings(labels) -> dict[str, np.ndarray]:
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
    synonyms = load_all_labels()
    if not synonyms:
        raise RuntimeError(f"No labels found in {SPLITS_DIR}. Run scripts/05 first.")
    labels = list(synonyms)

    word2vec = build_word2vec_embeddings(synonyms)
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

    seen_names = [c["class_name"] for c in load_class_split(SPLITS_DIR / "seen.json")]
    incomplete = []
    for embedding in Embedding:
        rows, kept = load_label_embeddings(EMBEDDINGS_DIR / "label_embeddings.npz", embedding, seen_names)
        if len(kept) != len(seen_names):
            incomplete.append(embedding.value)
            continue
        torch.save(rows, EMBEDDINGS_DIR / f"seen_{embedding.value}.pt")
    if incomplete:
        # The other embedding's training tensor is still written, so its experiments can run.
        raise RuntimeError(f"Every seen class needs an embedding to train on; seen_*.pt not written for {incomplete}.")


if __name__ == "__main__":
    main()
