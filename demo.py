"""
Demo / Quick-Start script.
Generates synthetic noise images to verify the full pipeline works
without needing real training data.

Usage:
    python demo.py
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import DogCatCNN
from utils import get_device, set_seed, accuracy, AverageMeter


def create_dummy_dataset(base_dir: str, n_per_class: int = 50, img_size: int = 128):
    """
    Create tiny synthetic dataset with random-colour images
    for pipeline smoke-testing.
    """
    base = Path(base_dir)
    for split in ("train", "val"):
        for cls in ("dogs", "cats"):
            folder = base / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                color = [np.random.randint(0, 255) for _ in range(3)]
                arr   = np.full((img_size, img_size, 3), color, dtype=np.uint8)
                # Add some noise
                arr   = np.clip(arr + np.random.randint(-30, 30,
                        arr.shape, dtype=np.int16), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(folder / f"{cls[:-1]}_{i:04d}.jpg")
    print(f"[OK] Synthetic dataset created in '{base_dir}'")


def run_demo():
    set_seed(42)
    device = get_device()

    demo_data = "demo_data"
    print("\n[-] Creating synthetic demo dataset ...")
    create_dummy_dataset(demo_data, n_per_class=30)

    from dataset import get_dataloaders
    loaders, class_to_idx = get_dataloaders(
        data_dir=demo_data,
        batch_size=8,
        num_workers=0,
        val_split=0.0
    )
    print(f"  class_to_idx = {class_to_idx}")
    print(f"  Train batches: {len(loaders['train'])}")

    # Instantiate model
    print("\n[-] Building model ...")
    model = DogCatCNN().to(device)
    model.summary()

    # One forward pass
    print("\n[-] Running one forward pass ...")
    images, labels = next(iter(loaders["train"]))
    images = images.to(device)
    labels_f = labels.float().unsqueeze(1).to(device)

    import torch.nn as nn
    criterion = nn.BCELoss()
    with torch.no_grad():
        outputs = model(images)
    loss = criterion(outputs, labels_f)
    acc  = accuracy(outputs, labels_f)

    print(f"  Batch size    : {images.size(0)}")
    print(f"  Output shape  : {outputs.shape}")
    print(f"  BCE Loss      : {loss.item():.4f}")
    print(f"  Accuracy      : {acc:.2f}%  (random init -> ~50% expected)")

    # Visualise model outputs
    Path("plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    idx_to_class = {v: k.capitalize() for k, v in class_to_idx.items()}

    for i in range(min(8, images.size(0))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        pred_prob = outputs[i].item()
        pred_cls  = "Dog" if pred_prob >= 0.5 else "Cat"
        true_cls  = idx_to_class[labels[i].item()]
        color     = "green" if pred_cls == true_cls else "red"
        axes[i].imshow(img)
        axes[i].set_title(f"True:{true_cls} | Pred:{pred_cls}\n({pred_prob:.2f})",
                          fontsize=8, color=color)
        axes[i].axis("off")

    plt.suptitle("Demo: Forward Pass Results (random weights)", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/demo_output.png", dpi=120)
    print("\n[OK] Demo output saved -> plots/demo_output.png")
    print("[OK] Pipeline smoke-test PASSED!\n")

    # Cleanup
    import shutil
    shutil.rmtree(demo_data)
    print(f"[OK] Cleaned up '{demo_data}'\n")


if __name__ == "__main__":
    run_demo()
