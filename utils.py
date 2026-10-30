"""
Utility functions for Dog-Cat CNN project.
  - Device selection
  - Reproducibility (seed)
  - Checkpoint save / load
  - Early stopping
  - Metrics (accuracy, AverageMeter)
  - Training history plots
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt


# -- Reproducibility -----------------------------------------------------------
def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -- Device selection ----------------------------------------------------------
def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] No GPU detected -- using CPU")
    return device


# -- AverageMeter --------------------------------------------------------------
class AverageMeter:
    """Track and compute running average of a scalar."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# -- Accuracy ------------------------------------------------------------------
def accuracy(outputs: torch.Tensor, labels: torch.Tensor,
             threshold: float = 0.5) -> float:
    """
    Binary classification accuracy (%).

    Args:
        outputs : model sigmoid outputs, shape (N, 1)
        labels  : ground-truth float labels, shape (N, 1)
        threshold: decision boundary

    Returns:
        accuracy in percent
    """
    preds   = (outputs >= threshold).float()
    correct = (preds == labels).sum().item()
    return correct / labels.size(0) * 100.0


# -- Checkpoint helpers --------------------------------------------------------
def save_checkpoint(model, optimizer, epoch: int, metric: float,
                    save_dir: str, tag: str = "best"):
    """
    Save model + optimizer state to a .pth file.

    Args:
        model    : DogCatCNN instance
        optimizer: optimizer instance
        epoch    : current epoch number
        metric   : validation accuracy (used in filename)
        save_dir : directory to save checkpoints
        tag      : 'best' or 'final'
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / f"{tag}.pth"
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric":               metric,
    }, path)
    print(f"[OK] Checkpoint saved -> {path}  (val_acc={metric:.2f}%)")


def load_checkpoint(model, path: str, device=None):
    """
    Load a saved checkpoint into the model.

    Returns:
        model, optimizer_state, epoch, metric
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[OK] Loaded checkpoint: {path}  "
          f"(epoch={ckpt['epoch']}, metric={ckpt['metric']:.2f}%)")
    return model, ckpt.get("optimizer_state_dict"), ckpt["epoch"], ckpt["metric"]


# -- Early Stopping ------------------------------------------------------------
class EarlyStopping:
    """
    Stop training when the monitored metric stops improving.

    Args:
        patience : epochs to wait after last improvement
        min_delta: minimum change to qualify as improvement
        save_dir : directory to save the best model
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 save_dir: str = "models"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_dir   = save_dir
        self.counter    = 0
        self.best_loss  = float("inf")
        self.best_state = None

    def __call__(self, val_loss: float, model, optimizer, epoch: int) -> bool:
        """
        Call at the end of each epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = model.state_dict()
            self.counter    = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                # Restore best weights
                if self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
        return False


# -- Training history plots ----------------------------------------------------
def plot_training_history(history: dict, save_path: str = "plots/training_history.png"):
    """
    Plot loss and accuracy curves for train and validation sets.

    Args:
        history  : dict with keys train_loss, train_acc, val_loss, val_acc
        save_path: where to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dog-Cat CNN -- Training History", fontsize=14, fontweight="bold")

    # -- Loss --------------------------------------------------------------
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    if history.get("val_loss"):
        ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -- Accuracy ----------------------------------------------------------
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    if history.get("val_acc"):
        ax2.plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[OK] Training history plot saved -> {save_path}")


# -- Visualise a batch ---------------------------------------------------------
def show_batch(images: torch.Tensor, labels: torch.Tensor,
               class_names=("Cat", "Dog"), n: int = 8,
               save_path: str = "plots/sample_batch.png"):
    """Display a grid of images from a DataLoader batch."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 6))
    axes = axes.flatten()
    for i in range(min(n, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = std * img + mean          # unnormalize
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i].item()], fontsize=10)
        axes[i].axis("off")
    plt.suptitle("Sample Training Images", fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[OK] Sample batch saved -> {save_path}")
