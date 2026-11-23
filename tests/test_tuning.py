from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import optuna
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.train import initializer, test_model as evaluate_model, train_experiment
from src.tune import tune_hyperparameters
from src.utils.load_data import close_loaders, load_data


class TinyBackbone(nn.Module):
    output_dim = 3

    def forward(self, images):
        return images.mean(dim=(-2, -1))


class TuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        archive_path = self.root / "images.zip"
        rows = []
        with zipfile.ZipFile(archive_path, "w") as archive:
            for wnid, color in (("n1", "red"), ("n2", "blue"), ("unseen", "green")):
                for partition, count in (("train", 5), ("val", 2)):
                    for i in range(count):
                        member = f"{wnid}_{partition}_{i}.png"
                        image = BytesIO()
                        Image.new("RGB", (32, 32), color).save(image, format="PNG")
                        archive.writestr(member, image.getvalue())
                        rows.append({
                            "wnid": wnid, "split": partition,
                            "archive_path": str(archive_path), "member_path": member,
                        })
        self.manifest_path = self.root / "manifest.parquet"
        pd.DataFrame(rows).to_parquet(self.manifest_path, index=False)
        split_path = self.root / "seen.json"
        split_path.write_text(json.dumps({"classes": [{"wnid": "n2"}, {"wnid": "n1"}]}))
        self.config = {
            "seed": 541,
            "model": {"backbone": "resnet18", "weights": None, "num_classes": 2},
            "data": {
                "manifest": str(self.manifest_path), "seen_classes": str(split_path),
                "val_split": 0.4, "num_workers": 0,
            },
            "training_args": {"batch_size": 4, "epochs": 1},
            "hyperparameters": {"learning_rate": 0.001},
        }
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.storages = []
        self.actual_create_study = optuna.create_study
        self.addCleanup(self.close_storages)

    def create_study(self, **kwargs):
        storage = optuna.storages.RDBStorage(kwargs.pop("storage"))
        self.storages.append(storage)
        return self.actual_create_study(storage=storage, **kwargs)

    def close_storages(self):
        for storage in self.storages:
            storage.remove_session()
            storage.engine.dispose()

    @staticmethod
    def tiny_model(config):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))

    def test_loaders_are_disjoint_reproducible_and_follow_label_order(self):
        first = load_data(self.config["data"], batch_size=4, seed=541)
        second = load_data(self.config["data"], batch_size=2, seed=541)
        self.addCleanup(close_loaders, *first, *second)
        partitions = []
        for left, right, count in zip(first, second, (6, 4, 4)):
            self.assertEqual(len(left.dataset), count)
            pd.testing.assert_frame_equal(left.dataset.manifest, right.dataset.manifest)
            partitions.append(set(left.dataset.manifest["member_path"]))
            self.assertEqual(set(left.dataset.manifest["wnid"]), {"n1", "n2"})
            expected_labels = left.dataset.manifest["wnid"].map({"n2": 0, "n1": 1})
            pd.testing.assert_series_equal(
                left.dataset.manifest["label"], expected_labels, check_names=False
            )
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertEqual(set(first[1].dataset.manifest["split"]), {"val"})
        self.assertEqual(set(first[2].dataset.manifest["split"]), {"train"})
        images, labels = next(iter(first[0]))
        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(labels.dtype, torch.int64)
        # Windows DataLoader workers serialize datasets using spawn.
        restored = pickle.loads(pickle.dumps(first[0].dataset))
        self.addCleanup(restored.close)
        self.assertEqual(restored[0][1], first[0].dataset[0][1])

    def test_loader_rejects_leakage_and_insufficient_training_data(self):
        frame = pd.read_parquet(self.manifest_path)
        duplicate = frame.iloc[[0]].copy()
        duplicate["split"] = "val"
        pd.concat([frame, duplicate]).to_parquet(self.manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate images"):
            load_data(self.config["data"], 2)
        frame = frame.drop(frame.index[(frame["wnid"] == "n1") & (frame["split"] == "train")][1:])
        frame.to_parquet(self.manifest_path, index=False)
        with self.assertRaisesRegex(ValueError, "at least two training images"):
            load_data(self.config["data"], 2)

    def test_evaluation_weights_partial_batches_by_sample_count(self):
        logits = torch.tensor([[5., 0.], [5., 0.], [5., 0.]])
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        with patch("src.train.device", torch.device("cpu")):
            loss, accuracy = evaluate_model(nn.Identity(), loader, nn.CrossEntropyLoss())
        self.assertAlmostEqual(loss, nn.CrossEntropyLoss()(logits, labels).item(), places=6)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_real_optuna_trials_persist_resume_and_never_evaluate_test_images(self):
        output = self.root / "study"
        starts = []

        def factory(config):
            model = self.tiny_model(config)
            starts.append(model[-1].weight.detach().clone())
            return model

        def validation_only(model, loader, criterion):
            self.assertEqual(set(loader.dataset.manifest["split"]), {"train"})
            return evaluate_model(model, loader, criterion)

        options = {
            "config_path": self.config_path, "n_trials": 2,
            "learning_rate_min": 0.0001, "learning_rate_max": 0.01,
            "batch_sizes": (2, 4),
        }
        original_config = deepcopy(self.config)
        with (
            patch("src.train.initialize_model_noembed", side_effect=factory),
            patch("src.tune.optuna.create_study", side_effect=self.create_study),
            patch("src.train.test_model", side_effect=validation_only) as evaluate,
            patch("src.train.device", torch.device("cpu")),
            patch("src.tune.device", torch.device("cpu")),
        ):
            study = tune_hyperparameters("cnn_noembed", output, **options)
            self.assertEqual(len(study.trials), 2)
            self.assertEqual(evaluate.call_count, 2)
            self.assertTrue(all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
            self.assertEqual(study.best_value, min(trial.value for trial in study.trials))
            torch.testing.assert_close(starts[0], starts[1], rtol=0, atol=0)
            with (output / "best_config.yaml").open() as file:
                best_config = yaml.safe_load(file)
            self.assertEqual(best_config["hyperparameters"]["learning_rate"], study.best_params["learning_rate"])
            self.assertEqual(best_config["training_args"]["batch_size"], study.best_params["batch_size"])
            self.assertEqual(best_config["training_args"]["epochs"], 1)
            self.assertEqual(len(pd.read_csv(output / "trials.csv")), 2)
            summary = json.loads((output / "best_trial.json").read_text())
            self.assertEqual(summary["validation_loss"], study.best_value)
            options["n_trials"] = 1
            resumed = tune_hyperparameters("cnn_noembed", output, **options)
            self.assertEqual(len(resumed.trials), 3)
            self.assertEqual(len(pd.read_csv(output / "trials.csv")), 3)
            options["batch_sizes"] = (8,)
            with self.assertRaisesRegex(ValueError, "setup changed"):
                tune_hyperparameters("cnn_noembed", output, **options)
        self.assertEqual(yaml.safe_load(self.config_path.read_text()), original_config)

    def test_embedding_tuning_supports_both_loss_functions(self):
        embedding_path = self.root / "embeddings.pt"
        torch.save(torch.tensor([[0., 0., 1.], [1., 0., 0.]]), embedding_path)
        self.config["model"]["embedding_paths"] = {"Word2Vec": str(embedding_path)}
        self.config_path.write_text(yaml.safe_dump(self.config))
        with (
            patch("src.cnn_embed.FrozenResNet18", return_value=TinyBackbone()),
            patch("src.tune.optuna.create_study", side_effect=self.create_study),
            patch("src.train.device", torch.device("cpu")),
            patch("src.tune.device", torch.device("cpu")),
        ):
            for loss in ("1", "2"):
                with self.subTest(loss=loss):
                    study = tune_hyperparameters(
                        "cnn_embed", self.root / f"embedding_{loss}",
                        config_path=self.config_path, embedding_choice="1",
                        loss_choice=loss, n_trials=1, batch_sizes=(2,),
                    )
                    self.assertTrue(torch.isfinite(torch.tensor(study.best_value)))
                    self.assertEqual(study.best_trial.state, optuna.trial.TrialState.COMPLETE)

    def test_training_failure_is_persisted_and_propagated(self):
        def fail_training(model, train_loader, *args):
            next(iter(train_loader))
            raise RuntimeError("training failed")

        with (
            patch("src.train.initialize_model_noembed", side_effect=self.tiny_model),
            patch("src.tune.optuna.create_study", side_effect=self.create_study),
            patch("src.tune.train_model", side_effect=fail_training),
            patch("src.tune.device", torch.device("cpu")),
        ):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                tune_hyperparameters(
                    "cnn_noembed", self.root / "failed", config_path=self.config_path,
                    n_trials=1, batch_sizes=(2,),
                )
        study = optuna.load_study(study_name="hyperparameters", storage=self.storages[-1])
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.FAIL)
        self.assertFalse((self.root / "failed" / "best_config.yaml").exists())

    def test_normal_training_saves_validation_and_one_final_test_result(self):
        with (
            patch("src.train.initialize_model_noembed", side_effect=self.tiny_model),
            patch("src.train.device", torch.device("cpu")),
            patch("src.train.Paths", SimpleNamespace(RESULTS=str(self.root / "results"))),
            patch("src.utils.load_model.Paths", SimpleNamespace(MODELS=str(self.root / "models"))),
            patch("src.train.test_model", wraps=evaluate_model) as evaluate,
        ):
            initialized = initializer("cnn_noembed", config_path=self.config_path)
            train_experiment("tiny", *initialized)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(set(evaluate.call_args_list[0].args[1].dataset.manifest["split"]), {"train"})
        self.assertEqual(set(evaluate.call_args_list[1].args[1].dataset.manifest["split"]), {"val"})
        metrics = json.loads((self.root / "results" / "tiny_metrics.json").read_text())
        self.assertEqual(len(metrics["val_losses"]), 1)
        self.assertTrue(0 <= metrics["test_accuracy"] <= 1)
        self.assertTrue((self.root / "models" / "tiny_model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
