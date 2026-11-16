from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.FrozenResNet import FrozenResNet18
from src.utils.enums.embedding_enums import Embedding


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Class embeddings must be a tensor.")
    if embeddings.ndim != 2 or min(embeddings.shape) < 1:
        raise ValueError("Class embeddings must have shape (num_classes, embedding_dim).")
    if not embeddings.is_floating_point():
        raise TypeError("Class embeddings must be floating point.")
    embeddings = embeddings.detach().to(dtype=torch.float32).clone()
    if not torch.isfinite(embeddings).all():
        raise ValueError("Class embeddings must contain only finite values.")
    norms = embeddings.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Class embeddings must have finite, nonzero row norms.")
    return embeddings / norms


class CNNEmbed(nn.Module):
    """Map frozen CNN features into a fixed class-label embedding space."""

    def __init__(
        self,
        class_embeddings: torch.Tensor,
        weights: str | None = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()
        if weights is not None and not isinstance(weights, str):
            raise TypeError("weights must be a ResNet-18 weight name or None.")

        self.register_buffer("class_embeddings", _normalize_embeddings(class_embeddings))
        self.backbone = FrozenResNet18(weights=weights)
        self.projection = nn.Linear(
            self.backbone.output_dim, self.class_embeddings.shape[1]
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return unit-length image embeddings for cosine-loss training."""
        return F.normalize(self.projection(self.backbone(images)), dim=1)

    def score_embeddings(
        self,
        image_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score the training vocabulary or a replacement zero-shot vocabulary."""
        candidates = self.class_embeddings
        if candidate_embeddings is not None:
            candidates = _normalize_embeddings(candidate_embeddings)
            if candidates.shape[1] != self.projection.out_features:
                raise ValueError("Candidate embeddings must match the projection dimension.")
        candidates = candidates.to(image_embeddings)
        return F.normalize(image_embeddings, dim=1) @ candidates.T

    def forward(
        self,
        images: torch.Tensor,
        candidate_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return cosine-similarity logits, not softmax probabilities."""
        return self.score_embeddings(self.encode_images(images), candidate_embeddings)


def initialize_model(
    embedding_type: Embedding | str, config: Mapping[str, object]
) -> CNNEmbed:
    """Load a fixed, ordered class-embedding tensor and build its projection."""
    embedding_type = Embedding(embedding_type)
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise TypeError("config['model'] must be a mapping.")
    if model_config["backbone"] != "resnet18":
        raise ValueError("CNNEmbed supports only resnet18 backbone.")

    embedding_paths = model_config["embedding_paths"]
    if not isinstance(embedding_paths, Mapping):
        raise TypeError("model.embedding_paths must be a mapping.")
    embedding_path = embedding_paths[embedding_type.value]
    if not isinstance(embedding_path, str) or not embedding_path.strip():
        raise ValueError(
            f"Set model.embedding_paths.{embedding_type.value} to a class-embedding .pt file."
        )
    class_embeddings = torch.load(
        Path(embedding_path), map_location="cpu", weights_only=True
    )
    return CNNEmbed(class_embeddings, weights=model_config["weights"])