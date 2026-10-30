"""
Training script for Dog-Cat CNN.
Usage:
    python train.py --data_dir data --epochs 30 --batch_size 32 --lr 0.001
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import DogCatCNN
from dataset import get_dataloaders
from utils import (
    set_seed, get_device, EarlyStopping,
    save_checkpoint, plot_training_history,
    AverageMeter, accuracy
)


# -- Argument parsing ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN for Dog vs Cat classification"
    )
    parser.add_argument("--data_dir",    type=str,   default="data",
                        help="Root data directory")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--dropout",     type=float, default=0.5)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=7,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--save_dir",    type=str,   default="models")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--val_split",   type=float, default=0.2,
                        help="Val fraction (used only if val/ folder is empty)")
    return parser.parse_args()


# -- One epoch of training -----------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc,         images.size(0))

    return loss_meter.avg, acc_meter.avg


# -- One epoch of evaluation ---------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        acc = accuracy(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc,         images.size(0))

    return loss_meter.avg, acc_meter.avg


# -- Main training loop --------------------------------------------------------
def main():
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("   Dog vs Cat -- CNN Training")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print("=" * 60 + "\n")

    # -- Data -------------------------------------------------------------
    loaders, class_to_idx = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )
    print(f"  Classes        : {class_to_idx}")
    print(f"  Train batches  : {len(loaders['train'])}")
    if "val" in loaders:
        print(f"  Val   batches  : {len(loaders['val'])}")

    # -- Model -------------------------------------------------------------
    model = DogCatCNN(dropout_rate=args.dropout).to(device)
    model.summary()

    # -- Loss / Optimizer / Scheduler --------------------------------------
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler     = ReduceLROnPlateau(optimizer, mode="min",
                                      factor=0.5, patience=3, verbose=True)
    early_stopper = EarlyStopping(patience=args.patience,
                                  save_dir=args.save_dir)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    # -- Training loop -----------------------------------------------------
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss = val_acc = None
        if "val" in loaders:
            val_loss, val_acc = evaluate(
                model, loaders["val"], criterion, device
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, epoch,
                                val_acc, args.save_dir, tag="best")

            stop = early_stopper(val_loss, model, optimizer, epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:>3}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
            + (f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% " if val_acc is not None else "")
            + f"| {elapsed:.1f}s"
        )

        if "val" in loaders and stop:
            print("\n[!] Early stopping triggered.")
            break

    # -- Save final model & history -----------------------------------------
    save_checkpoint(model, optimizer, epoch,
                    best_val_acc, args.save_dir, tag="final")

    history_path = Path(args.save_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved -> {history_path}")

    # -- Plot --------------------------------------------------------------
    plot_training_history(history, save_path="plots/training_history.png")

    print(f"\n[OK] Best validation accuracy: {best_val_acc:.2f}%")
    print("[OK] Training complete!\n")


if __name__ == "__main__":
    main()
