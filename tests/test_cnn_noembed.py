import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
import yaml

from FrozenResNet import FrozenResNet18
from src.cnn_noembed import CNNNoEmbed, initialize_model


class CNNNoEmbedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.original_rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.model = CNNNoEmbed(num_classes=3, weights=None)

    def tearDown(self) -> None:
        torch.set_rng_state(self.original_rng_state)

    def test_feature_and_logit_shapes(self) -> None:
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                images = torch.randn(batch_size, 3, 224, 224)
                self.assertEqual(self.model.backbone(images).shape, (batch_size, 512))
                logits = self.model(images)
                self.assertEqual(logits.shape, (batch_size, 3))
                self.assertTrue(torch.isfinite(logits).all().item())
                self.assertTrue(logits.requires_grad)

    def test_training_updates_only_head_and_preserves_backbone_state(self) -> None:
        self.model.train()
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.classifier.training)
        self.assertTrue(all(not module.training for module in self.model.backbone.modules()))
        before_backbone = copy.deepcopy(self.model.backbone.state_dict())
        before_head = self.model.classifier.weight.detach().clone()
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=0.001,
        )

        logits = self.model(torch.randn(2, 3, 224, 224))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 2]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(self.model.classifier.weight.grad)
        self.assertGreater(self.model.classifier.weight.grad.abs().sum().item(), 0)
        for parameter in self.model.backbone.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        optimizer.step()

        self.assertFalse(torch.equal(before_head, self.model.classifier.weight))
        for name, value in self.model.backbone.state_dict().items():
            with self.subTest(state=name):
                self.assertTrue(torch.equal(before_backbone[name], value))

    def test_eval_train_transitions_keep_backbone_frozen(self) -> None:
        for mode in (False, True, False, True):
            self.assertIs(self.model.train(mode), self.model)
            self.assertEqual(self.model.classifier.training, mode)
            self.assertTrue(
                all(not module.training for module in self.model.backbone.modules())
            )

    def test_pretrained_weights_are_requested(self) -> None:
        with patch("src.backbone.resnet18", return_value=resnet18(weights=None)) as factory:
            backbone = FrozenResNet18()
        factory.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.assertEqual(backbone.output_dim, 512)
        self.assertIsInstance(backbone.encoder.fc, nn.Identity)

    def test_repository_configuration(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "configs"
            / "cnn_noembed_config.yaml"
        )
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.assertEqual(config["model"]["weights"], "IMAGENET1K_V1")
        # Exercise the actual factory without downloading pretrained weights.
        config["model"]["weights"] = None
        model = initialize_model(config)
        self.assertEqual(model.classifier.out_features, 1000)
        self.assertEqual(model.classifier.in_features, 512)

    def test_invalid_configuration_fails_explicitly(self) -> None:
        for num_classes in (0, -1, True, 2.5):
            with self.subTest(num_classes=num_classes):
                with self.assertRaises((TypeError, ValueError)):
                    CNNNoEmbed(num_classes=num_classes, weights=None)
        for model_config, error in (
            ({"backbone": "resnet50", "num_classes": 3, "weights": None}, ValueError),
            ({"backbone": "resnet18", "num_classes": "3", "weights": None}, TypeError),
            ({"backbone": "resnet18", "num_classes": 3, "weights": 1}, TypeError),
            ({"backbone": "resnet18", "num_classes": 3, "weights": "typo"}, ValueError),
        ):
            with self.subTest(model_config=model_config):
                with self.assertRaises(error):
                    initialize_model({"model": model_config})
        with self.assertRaises(TypeError):
            initialize_model({"model": None})
        with self.assertRaises(KeyError):
            initialize_model({})

    def test_state_dict_round_trip(self) -> None:
        self.model.eval()
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = self.model(images)
        restored = CNNNoEmbed(num_classes=3, weights=None)
        restored.load_state_dict(self.model.state_dict())
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
