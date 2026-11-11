from collections.abc import Mapping
import torch
from torch import nn
from FrozenResNet import FrozenResNet18

class CNNNoEmbed(nn.Module):

    def __init__(
        self, num_classes: int, weights: str | None = "IMAGENET1K_V1"
    ) -> None:
        super().__init__()

        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer.")
        
        if num_classes < 1:
            raise ValueError("num_classes must be greater than 0.")
        
        if weights is not None and not isinstance(weights, str):
            raise TypeError("weights must be a ResNet-18 weight name or None.")

        self.backbone = FrozenResNet18(weights=weights)
        self.classifier = nn.Linear(self.backbone.output_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))


def initialize_model(config: Mapping[str, object]) -> CNNNoEmbed:
    model_config = config["model"]

    if not isinstance(model_config, Mapping):
        raise TypeError("config['model'] must be a mapping.")
    
    if model_config["backbone"] != "resnet18":
        raise ValueError("CNNNoEmbed supports only resnet18 backbone.")

    num_classes = model_config["num_classes"]
    weights = model_config["weights"]
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("model.num_classes must be an integer.")
    
    if weights is not None and not isinstance(weights, str):
        raise TypeError("model.weights must be a ResNet-18 weight name or None.")
    
    return CNNNoEmbed(num_classes=num_classes, weights=weights)