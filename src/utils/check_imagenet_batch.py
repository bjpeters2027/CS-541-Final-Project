import torch

from src.utils.load_config import load_config
from src.utils.load_data import load_data

config = load_config("cnn_noembed_config")
data = config["data"]
num_classes = config["model"]["num_classes"]

train, test, val = load_data(
    data["datafile"],
    batch_size=8,
    test_split=data["test_split"],
    validation_split=data["val_split"],
)

images, labels = next(iter(train))
assert tuple(images.shape) == (8, 3, 224, 224), images.shape
assert labels.dtype == torch.long, labels.dtype
assert ((labels >= 0) & (labels < num_classes)).all(), labels

print("ImageNet batch passed:", tuple(images.shape), labels.tolist())
print("Split sizes:", len(train.dataset), len(test.dataset), len(val.dataset))