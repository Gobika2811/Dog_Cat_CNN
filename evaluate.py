"""
Evaluation & inference script for the trained Dog-Cat CNN.
Usage:
    # Evaluate on val set
    python evaluate.py --data_dir data --checkpoint models/best.pth

    # Predict a single image
    python evaluate.py --image path/to/image.jpg --checkpoint models/best.pth
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from model import DogCatCNN
from dataset import get_dataloaders, get_transforms
from utils import get_device, load_checkpoint


# -- CLI -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate or run inference with the Dog-Cat CNN"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--data_dir",   type=str, default="data",
                        help="Root data directory (for dataset evaluation)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--image",      type=str, default=None,
                        help="Single image path for inference")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Decision threshold (dog probability)")
    return parser.parse_args()


# -- Single-image inference ----------------------------------------------------
def predict_single(model, image_path: str, device, threshold: float = 0.5):
    """Predict dog/cat for one image and display the result."""
    transform = get_transforms(augment=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prob = model(tensor).item()

    label = "Dog 🐶" if prob >= threshold else "Cat 🐱"
    confidence = prob if prob >= threshold else 1 - prob

    print(f"\n{'=' * 40}")
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print(f"  Dog prob   : {prob * 100:.2f}%")
    print(f"{'=' * 40}\n")

    # Visualise
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{label}  (conf: {confidence * 100:.1f}%)", fontsize=14)
    plt.tight_layout()
    out = f"plots/prediction_{Path(image_path).stem}.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(out, dpi=100)
    plt.show()
    print(f"[OK] Prediction image saved -> {out}")


# -- Full dataset evaluation ---------------------------------------------------
@torch.no_grad()
def evaluate_dataset(model, loader, device, threshold: float = 0.5,
                      class_names=("Cat", "Dog")):
    model.eval()
    all_probs  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        probs  = model(images).squeeze(1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= threshold).astype(int)

    # -- Metrics ----------------------------------------------------------
    acc     = (all_preds == all_labels).mean() * 100
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm      = confusion_matrix(all_labels, all_preds)
    report  = classification_report(all_labels, all_preds,
                                     target_names=class_names)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"\n{report}")

    # -- Confusion matrix plot ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix
    im = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(class_names)
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(class_names)
    plt.colorbar(im, ax=axes[0])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=14, fontweight="bold")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC (AUC = {auc_roc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = "plots/evaluation_metrics.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.show()
    print(f"[OK] Evaluation plots saved -> {out}")

    return acc, auc_roc


# -- Entry point ---------------------------------------------------------------
def main():
    args   = parse_args()
    device = get_device()

    model = DogCatCNN()
    model, _, _, _ = load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    if args.image:
        predict_single(model, args.image, device, args.threshold)
    else:
        loaders, class_to_idx = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        idx_to_class = {v: k.capitalize() for k, v in class_to_idx.items()}
        class_names  = [idx_to_class[0], idx_to_class[1]]

        split = "val" if "val" in loaders else "train"
        print(f"\n[->] Evaluating on '{split}' split ...")
        evaluate_dataset(model, loaders[split], device,
                         args.threshold, class_names)


if __name__ == "__main__":
    main()
