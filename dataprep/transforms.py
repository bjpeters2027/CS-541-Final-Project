"""Standard ImageNet-style preprocessing, shared by every manifest-backed dataset
(ImageNet itself plus the three domain eval sets) so seen and unseen images are
preprocessed identically.
"""
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
