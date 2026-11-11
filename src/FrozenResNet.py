import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class FrozenResNet18(nn.Module):

    def __init__(self, weights: str | None = "IMAGENET1K_V1") -> None:
        super().__init__()
        
        if weights is not None and weights not in ResNet18_Weights.__members__:
            raise ValueError(f"Unknown ResNet-18 weights: {weights!r}")

        pretrained_weights = ResNet18_Weights[weights] if weights is not None else None
        self.encoder = resnet18(weights=pretrained_weights)
        self.output_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenResNet18":
        # Frozen parameters alone do not freeze BatchNorm running statistics.
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(images)
