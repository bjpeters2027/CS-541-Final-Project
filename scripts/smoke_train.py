"""Exercise data preparation and all four embedding/loss combinations without downloads."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def create_fixtures(raw: Path, data: Path) -> None:
    image = io.BytesIO()
    Image.new("RGB", (256, 256), (80, 120, 160)).save(image, format="JPEG")
    image_bytes = image.getvalue()
    imagenet = raw / "imagenet"
    imagenet.mkdir(parents=True)
    with zipfile.ZipFile(imagenet / "synthetic.zip", "w") as archive:
        archive.writestr(
            "LOC_synset_mapping.txt",
            "".join(f"n{i:08d} synthetic class {i}\n" for i in range(4)),
        )
        val_rows = ["ImageId,PredictionString"]
        for i in range(4):
            wnid = f"n{i:08d}"
            for j in range(6):
                archive.writestr(
                    f"ILSVRC/Data/CLS-LOC/train/{wnid}/{wnid}_{j}.JPEG", image_bytes
                )
            for j in range(2):
                image_id = f"ILSVRC2012_val_{i}_{j}"
                archive.writestr(f"ILSVRC/Data/CLS-LOC/val/{image_id}.JPEG", image_bytes)
                val_rows.append(f"{image_id},{wnid} 0 0 10 10")
        archive.writestr("LOC_val_solution.csv", "\n".join(val_rows))

    members = {
        "awa2": ["Animals_with_Attributes2/JPEGImages/red+fox/red+fox_1.jpg",
                 "Animals_with_Attributes2/JPEGImages/blue+whale/blue+whale_1.jpg"],
        "cub": ["CUB_200_2011/images/001.Black_bird/bird.jpg",
                "CUB_200_2011/images/002.White_bird/bird.jpg"],
        "sun": ["images/k/kitchen/kitchen.jpg", "images/h/hut/outdoor/hut.jpg"],
    }
    split_names = {
        "awa2": ["red+fox", "blue+whale"],
        "cub": ["001.Black_bird", "002.White_bird"],
        "sun": ["kitchen", "hut_outdoor"],
    }
    for domain, names in members.items():
        folder = raw / domain
        folder.mkdir(parents=True)
        if domain == "awa2":
            with zipfile.ZipFile(folder / f"{domain}.zip", "w") as archive:
                for name in names:
                    archive.writestr(name, image_bytes)
        else:
            with tarfile.open(folder / f"{domain}.tar", "w") as archive:
                for name in names:
                    member = tarfile.TarInfo(name)
                    member.size = len(image_bytes)
                    archive.addfile(member, io.BytesIO(image_bytes))
        split_dir = data / "zsl_splits" / domain
        split_dir.mkdir(parents=True)
        (split_dir / "testclasses.txt").write_text("\n".join(split_names[domain]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "src" / "artifacts" / "smoke")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="cs541-smoke-") as directory:
        temp = Path(directory)
        raw, data = temp / "raw", temp / "data"
        os.environ["DATAPREP_RAW_ROOT"] = str(raw)
        os.environ["DATAPREP_DATA_ROOT"] = str(data)
        create_fixtures(raw, data)
        for script, extra in (
            ("02_build_imagenet_manifest.py", []),
            ("04_build_domain_manifests.py", []),
            ("05_build_class_splits.py", ["--k", "2"]),
        ):
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), *extra], check=True, cwd=ROOT
            )

        from config import EMBEDDINGS_DIR
        from src.evalutate import evaluate
        from src.train import initializer, train_experiment
        from src.utils.label_embeddings import load_class_split
        from src.utils.load_model import load_model

        labels = sorted({
            entry["class_name"]
            for path in (data / "splits").glob("*.json")
            for entry in load_class_split(path)
        })
        rng = np.random.default_rng(541)
        embedding_path = EMBEDDINGS_DIR / "label_embeddings.npz"
        np.savez(
            embedding_path, labels=np.array(labels),
            word2vec=rng.normal(size=(len(labels), 300)).astype("float32"),
            bert=rng.normal(size=(len(labels), 768)).astype("float32"),
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "07_validate_dataprep.py")],
            check=True, cwd=ROOT,
        )
        report = {"synthetic": True, "runs": {}}
        expected = {}
        probe = torch.zeros(1, 3, 224, 224)
        for embedding, embedding_choice in (("Word2Vec", "1"), ("BERT", "2")):
            for loss, loss_choice in (("cross_entropy", "1"), ("cosine", "2")):
                name = f"smoke_{embedding.lower()}_{loss}"
                configuration = {
                    "seed": 541, "output_dir": str(output),
                    "model": {"backbone": "resnet18", "weights": None,
                              "embedding_archive": str(embedding_path)},
                    "data": {"manifest": str(data / "manifests" / "imagenet.parquet"),
                             "seen_classes": str(data / "splits" / "seen.json"),
                             "val_split": 1 / 3, "num_workers": 0},
                    "training_args": {"batch_size": 2, "epochs": 1, "device": "cpu"},
                    "hyperparameters": {"learning_rate": 0.001},
                }
                config_path = temp / f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
                initialized = initializer("cnn_embed", embedding_choice, loss_choice, config_path)
                model = initialized[1]
                before = model.projection.weight.detach().clone()
                backbone_before = {key: value.clone() for key, value in model.backbone.state_dict().items()}
                result = train_experiment(name, *initialized)
                if torch.equal(before, model.projection.weight):
                    raise AssertionError("Training did not update the projection.")
                for key, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(value, backbone_before[key], rtol=0, atol=0)
                with torch.no_grad():
                    expected[result["checkpoint"]] = model(probe).clone()
                result["zero_shot"] = evaluate(
                    result["checkpoint"], data / "manifests" / "awa2.parquet",
                    data / "splits" / "unseen_awa2.json", embedding_path, embedding,
                    batch_size=2,
                )
                report["runs"][name] = result

    # Reload only after temporary archives, split files, and embeddings are gone.
    for checkpoint, logits in expected.items():
        with torch.no_grad():
            torch.testing.assert_close(load_model(checkpoint)(probe), logits, rtol=0, atol=0)
    report["reload_without_source_data"] = True
    report_path = output / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: four offline training runs, frozen backbones, zero-shot scoring, and reloads.")
    print(f"Synthetic results only; these accuracies do not measure model quality.\nReport: {report_path}")


if __name__ == "__main__":
    main()
