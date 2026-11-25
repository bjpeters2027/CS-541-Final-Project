import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import EMBEDDINGS_DIR, MANIFESTS_DIR, SPLITS_DIR
from dataprep.archive_dataset import ArchiveBackedImageDataset
from dataprep.transforms import build_transform

from src.cnn_embed import CNNEmbed

from src.utils.enums.model_save_enums import Saves
from src.utils.enums.path_enums import Paths

from src.utils.label_embeddings import load_class_split, load_label_embeddings
from src.utils.load_model import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAINS = ("awa2", "cub", "sun")

def accuracy(model, rows, labels, batch_size, num_workers, candidate_embeddings=None):
    """
    Top-1 accuracy of the model on the given manifest rows.

    Args:
        model (nn.Module): The model being evaluated.
        rows (pd.DataFrame): Manifest rows of the images to classify.
        labels (pd.Series): Index of each row's true class among the candidates.
        batch_size (int): Number of samples per batch.
        num_workers (int): DataLoader worker processes.
        candidate_embeddings (torch.Tensor, optional): Candidate vocabulary for a CNNEmbed model.
            Defaults to None, which scores against the classes the model was trained on.
    """

    dataset = ArchiveBackedImageDataset(rows.assign(label=labels.astype(int)), build_transform(train=False), target_column="label")
    correct = 0
    with torch.no_grad():
        for inputs, targets in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
            inputs = inputs.to(device)
            logits = model(inputs) if candidate_embeddings is None else model(inputs, candidate_embeddings.to(device))
            correct += (logits.argmax(dim=1).cpu() == targets).sum().item()
    return correct / len(dataset)

def zero_shot_accuracy(model, rows, vocabulary, archive_path, embedding, batch_size, num_workers):
    """
    Accuracy when each image is scored against the text embeddings of a candidate vocabulary.

    Args:
        rows (pd.DataFrame): Manifest rows with a "class_name" column.
        vocabulary (list[str]): Candidate class names. Names without an embedding are skipped,
            along with their images.
    """

    candidates, kept = load_label_embeddings(archive_path, embedding, vocabulary)
    class_index = {name: index for index, name in enumerate(kept)}
    rows = rows[rows["class_name"].isin(class_index)]
    return accuracy(model, rows, rows["class_name"].map(class_index), batch_size, num_workers, candidates)

def evaluate(model_name, batch_size=64):
    """
    Evaluate a saved model on the accuracy measurements from the experimental setup and save
    them to a JSON file.

    Every model gets seen-class accuracy. Embedding models also get unseen-class (zero-shot)
    accuracy on held-out ImageNet classes, generalized zero-shot accuracy (unseen images scored
    against seen and unseen classes together), and transfer accuracy on AWA2 / CUB / SUN.

    Args:
        model_name (str): Name of the saved model file, without the .pt extension.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
    """

    model = load_model(model_name).to(device)
    save = torch.load(f"{Paths.MODELS}/{model_name}.pt", map_location="cpu", weights_only=True)
    config = save[Saves.CONFIG]
    data_config = config["data"]
    num_workers = data_config["num_workers"]

    manifest = pd.read_parquet(MANIFESTS_DIR / data_config["manifest"])
    seen_classes = load_class_split(SPLITS_DIR / data_config["seen_classes"])
    seen_index = {entry["wnid"]: index for index, entry in enumerate(seen_classes)}

    test_seen = manifest[manifest["partition"] == "test_seen"]
    results = {"seen_accuracy": accuracy(model, test_seen, test_seen["wnid"].map(seen_index), batch_size, num_workers)}

    if isinstance(model, CNNEmbed):
        archive_path = EMBEDDINGS_DIR / config["model"]["embedding_archive"]
        embedding = save[Saves.EMBEDDING]
        seen_names = [entry["class_name"] for entry in seen_classes]
        unseen_names = [entry["class_name"] for entry in load_class_split(SPLITS_DIR / "unseen_imagenet.json")]
        test_unseen = manifest[manifest["partition"] == "test_unseen"]

        results["unseen_accuracy"] = zero_shot_accuracy(model, test_unseen, unseen_names, archive_path, embedding, batch_size, num_workers)
        results["generalized_zero_shot_accuracy"] = zero_shot_accuracy(model, test_unseen, seen_names + unseen_names, archive_path, embedding, batch_size, num_workers)

        for domain in DOMAINS:
            domain_manifest = MANIFESTS_DIR / f"{domain}.parquet"
            if not domain_manifest.exists():
                print(f"Skipping {domain}: no manifest at {domain_manifest}.")
                continue
            rows = pd.read_parquet(domain_manifest)
            rows = rows[rows["is_unseen_ps"]]
            vocabulary = [entry["class_name"] for entry in load_class_split(SPLITS_DIR / f"unseen_{domain}.json")]
            results[f"transfer_accuracy_{domain}"] = zero_shot_accuracy(model, rows, vocabulary, archive_path, embedding, batch_size, num_workers)

    Path(Paths.RESULTS).mkdir(parents=True, exist_ok=True)
    with open(f"{Paths.RESULTS}/{model_name}_eval.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evalutate <model_name>")
        sys.exit(1)
    print(json.dumps(evaluate(sys.argv[1]), indent=2))
