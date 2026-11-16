import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from src.cnn_embed import CNNEmbed, initialize_model
from src.train import forward_and_loss, initializer, test_model as evaluate_model
from src.utils.enums.embedding_enums import Embedding
from src.utils.enums.model_name_enums import Models


class CNNEmbedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.labels = torch.randn(3, 5, requires_grad=True)
        self.model = CNNEmbed(self.labels, weights=None)

    def tearDown(self) -> None:
        torch.set_rng_state(self.rng_state)

    def test_shapes_and_cosine_logits(self) -> None:
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                images = torch.randn(batch_size, 3, 64, 64)
                embeddings = self.model.encode_images(images)
                self.assertEqual(embeddings.shape, (batch_size, 5))
                torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(batch_size))
                logits = self.model(images)
                self.assertEqual(logits.shape, (batch_size, 3))
                torch.testing.assert_close(logits, embeddings @ self.model.class_embeddings.T)
                self.assertTrue(torch.isfinite(logits).all())

    def test_both_losses_train_only_projection(self) -> None:
        for criterion in (nn.CrossEntropyLoss(), nn.CosineEmbeddingLoss()):
            with self.subTest(loss=type(criterion).__name__):
                self.model.train()
                before_backbone = copy.deepcopy(self.model.backbone.state_dict())
                before_labels = self.model.class_embeddings.clone()
                before_projection = self.model.projection.weight.detach().clone()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
                optimizer.zero_grad()
                images = torch.randn(2, 3, 64, 64)
                labels = torch.tensor([0, 2])
                logits, loss = forward_and_loss(self.model, images, labels, criterion)
                self.assertEqual(logits.shape, (2, 3))
                self.assertTrue(torch.isfinite(loss))
                if isinstance(criterion, nn.CosineEmbeddingLoss):
                    expected = (
                        1 - logits[torch.arange(labels.numel()), labels]
                    ).mean()
                else:
                    expected = criterion(logits, labels)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before_projection, self.model.projection.weight))
                self.assertFalse(self.model.class_embeddings.requires_grad)
                self.assertIsNone(self.labels.grad)
                torch.testing.assert_close(before_labels, self.model.class_embeddings)
                for name, value in self.model.backbone.state_dict().items():
                    self.assertTrue(torch.equal(before_backbone[name], value), name)
                for parameter in self.model.backbone.parameters():
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)
                self.assertTrue(self.model.projection.training)
                self.assertTrue(
                    all(not module.training for module in self.model.backbone.modules())
                )

    def test_evaluation_supports_both_losses(self) -> None:
        batches = [(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))]
        with patch("src.train.device", torch.device("cpu")):
            for criterion in (nn.CrossEntropyLoss(), nn.CosineEmbeddingLoss()):
                loss, accuracy = evaluate_model(self.model, batches, criterion)
                self.assertGreaterEqual(loss, 0)
                self.assertTrue(0 <= accuracy <= 1)

    def test_zero_shot_vocabulary_and_nearest_neighbor(self) -> None:
        images = torch.randn(2, 3, 64, 64)
        before_labels = self.model.class_embeddings.clone()
        candidates = torch.randn(7, 5)
        candidates[:2] = self.model.encode_images(images).detach()
        logits = self.model(images, candidates)
        self.assertEqual(logits.shape, (2, 7))
        torch.testing.assert_close(logits.max(dim=1).values, torch.ones(2))
        self.assertEqual(logits.argmax(dim=1).tolist(), [0, 1])
        torch.testing.assert_close(before_labels, self.model.class_embeddings)
        with self.assertRaises(ValueError):
            self.model(images, torch.randn(7, 6))

    def test_state_dict_round_trip(self) -> None:
        images = torch.randn(1, 3, 64, 64)
        expected = self.model(images)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(self.model.state_dict(), path)
            restored = CNNEmbed(torch.randn(3, 5), weights=None)
            restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_invalid_embeddings_fail_explicitly(self) -> None:
        for embeddings, error in (
            ([], TypeError),
            (torch.ones(3), ValueError),
            (torch.ones(0, 5), ValueError),
            (torch.ones(3, 0), ValueError),
            (torch.ones(3, 5, dtype=torch.int64), TypeError),
            (torch.zeros(3, 5), ValueError),
            (torch.full((3, 5), float("nan")), ValueError),
            (torch.full((3, 5), float("inf")), ValueError),
        ):
            with self.subTest(embeddings=embeddings):
                with self.assertRaises(error):
                    CNNEmbed(embeddings, weights=None)

    def test_factory_loads_each_configured_embedding_type(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "src" / "configs" / "cnn_embed_config.yaml"
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.assertEqual(config["model"]["weights"], "IMAGENET1K_V1")
        config["model"].pop("embedding_archive")
        for embedding_type in Embedding:
            with self.assertRaisesRegex(ValueError, "class-embedding"):
                initialize_model(embedding_type, config)
        config["model"]["weights"] = None
        with tempfile.TemporaryDirectory() as directory:
            for embedding_type, dimension in ((Embedding.WORD2VEC, 300), (Embedding.BERT, 768)):
                path = Path(directory) / f"{embedding_type.value}.pt"
                torch.save(torch.randn(3, dimension), path)
                config["model"]["embedding_paths"][embedding_type.value] = str(path)
                model = initialize_model(embedding_type, config)
                self.assertEqual(model.projection.out_features, dimension)
                self.assertEqual(model(torch.randn(1, 3, 64, 64)).shape, (1, 3))

    def test_invalid_configuration(self) -> None:
        config = {
            "model": {
                "backbone": "resnet18",
                "weights": None,
                "embedding_paths": {"Word2Vec": ""},
            }
        }
        with self.assertRaises(ValueError):
            initialize_model("unknown", config)
        with self.assertRaises(TypeError):
            initialize_model(Embedding.BERT, {"model": None})
        config["model"]["backbone"] = "resnet50"
        with self.assertRaises(ValueError):
            initialize_model(Embedding.WORD2VEC, config)

    def test_cli_string_choices_select_both_losses(self) -> None:
        for embedding_choice, embedding_type in (("1", Embedding.WORD2VEC), ("2", Embedding.BERT)):
            for loss_choice, loss_type in (("1", nn.CrossEntropyLoss), ("2", nn.CosineEmbeddingLoss)):
                with patch("src.train.load_config", return_value={}), patch(
                    "src.train.initialize_model_embed", return_value=self.model
                ) as factory:
                    criterion, model, _, _, embedding = initializer(
                        Models.CNN_EMBED, embedding_choice, loss_choice
                    )
                self.assertIsInstance(criterion, loss_type)
                self.assertIs(model, self.model)
                self.assertEqual(embedding, embedding_type)
                factory.assert_called_once_with(embedding_type, {})


if __name__ == "__main__":
    unittest.main()
