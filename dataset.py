"""
Dataset utilities for Dog-Cat Classification.
Supports:
  - ImageFolder-based loading from train/val/test directories
  - Data augmentation for training
  - Normalization with ImageNet stats
"""

import os
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image


# ── ImageNet normalization constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 128   # resize all images to 128×128


def get_transforms(augment: bool = True):
    """
    Build torchvision transform pipelines.

    Args:
        augment: If True, apply data augmentation (for training).
                 If False, apply only resize + normalize (for val/test).

    Returns:
        torchvision.transforms.Compose
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(data_dir: str,
                    batch_size: int = 32,
                    num_workers: int = 2,
                    val_split: float = 0.0):
    """
    Create DataLoaders for train, validation, and test sets.

    Expected directory structure:
        data_dir/
            train/
                dogs/   *.jpg
                cats/   *.jpg
            val/
                dogs/   *.jpg
                cats/   *.jpg
            test/       *.jpg   (optional, unlabelled)

    Args:
        data_dir  : Root data directory.
        batch_size: Images per mini-batch.
        num_workers: Parallel data-loading workers.
        val_split : If > 0, carve this fraction from train set as validation
                    (only used when `val/` directory is empty or missing).

    Returns:
        dict with keys 'train', 'val', (optionally 'test') → DataLoader
        dict class_to_idx mapping {'cats': 0, 'dogs': 1}
    """
    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=get_transforms(augment=True)
    )

    val_path = data_dir / "val"
    if val_path.exists() and any(val_path.iterdir()) and val_split == 0.0:
        val_dataset = datasets.ImageFolder(
            root=str(val_path),
            transform=get_transforms(augment=False)
        )
    elif val_split > 0.0:
        val_size   = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        # Override val transform (random_split keeps original transform)
        val_dataset.dataset.transform = get_transforms(augment=False)
    else:
        val_dataset = None

    loaders = {}

    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    if val_dataset is not None:
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    test_path = data_dir / "test"
    if test_path.exists():
        test_dataset = UnlabelledTestDataset(
            test_dir=str(test_path),
            transform=get_transforms(augment=False)
        )
        if len(test_dataset) > 0:
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

    class_to_idx = (
        train_dataset.class_to_idx
        if isinstance(train_dataset, datasets.ImageFolder)
        else train_dataset.dataset.class_to_idx
    )

    return loaders, class_to_idx


class UnlabelledTestDataset(Dataset):
    """Loads images from a flat directory (no sub-folders) for inference."""

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def __init__(self, test_dir: str, transform=None):
        self.test_dir  = Path(test_dir)
        self.transform = transform
        self.image_paths = sorted([
            p for p in self.test_dir.iterdir()
            if p.suffix.lower() in self.VALID_EXTS
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(img_path.name)
