# 🐶🐱 Dog vs Cat Image Classifier — 3-Layer CNN

A clean PyTorch implementation of a **3-Convolutional-Layer CNN** for binary dog/cat image classification.

---

## Architecture

```
Input (3 × 128 × 128)
        │
┌───────▼────────────────────────────┐
│  Conv1: 3→32 filters, 3×3, pad=1  │
│  BatchNorm2d(32) → ReLU            │
│  MaxPool2d(2×2)  → 64×64          │
└───────────────────────────────────┘
        │
┌───────▼────────────────────────────┐
│  Conv2: 32→64 filters, 3×3, pad=1 │
│  BatchNorm2d(64) → ReLU            │
│  MaxPool2d(2×2)  → 32×32          │
└───────────────────────────────────┘
        │
┌───────▼─────────────────────────────┐
│  Conv3: 64→128 filters, 3×3, pad=1 │
│  BatchNorm2d(128) → ReLU            │
│  MaxPool2d(2×2)   → 16×16          │
└────────────────────────────────────┘
        │
   Flatten → 32768
        │
┌───────▼────────────────────┐
│  FC1: 32768→512, ReLU      │
│  Dropout(p=0.5)             │
└────────────────────────────┘
        │
┌───────▼───────────────────┐
│  FC2: 512→1, Sigmoid       │
└───────────────────────────┘
        │
  Output: P(Dog)  0.0–1.0
```

| Layer | Type | In | Out | Kernel | Activation |
|-------|------|----|-----|--------|------------|
| Conv1 | Conv2d + BN | 3 ch | 32 ch | 3×3 | ReLU + MaxPool |
| Conv2 | Conv2d + BN | 32 ch | 64 ch | 3×3 | ReLU + MaxPool |
| Conv3 | Conv2d + BN | 64 ch | 128 ch | 3×3 | ReLU + MaxPool |
| FC1   | Linear | 32768 | 512 | — | ReLU + Dropout |
| FC2   | Linear | 512 | 1 | — | Sigmoid |

> **Total parameters ≈ 16.8 M**

---

## Project Structure

```
dog_cat_cnn/
├── model.py          ← CNN architecture definition
├── dataset.py        ← DataLoader, transforms, augmentation
├── train.py          ← Training loop, checkpointing, early stopping
├── evaluate.py       ← Metrics, confusion matrix, ROC curve, single-image inference
├── utils.py          ← Helpers: seed, device, AverageMeter, plots
├── demo.py           ← Smoke-test with synthetic data (no real images needed)
├── requirements.txt  ← Python dependencies
├── data/
│   ├── train/
│   │   ├── dogs/     ← put training dog images here
│   │   └── cats/     ← put training cat images here
│   ├── val/
│   │   ├── dogs/     ← put validation dog images here
│   │   └── cats/     ← put validation cat images here
│   └── test/         ← optional unlabelled test images
├── models/           ← saved checkpoints (best.pth, final.pth)
└── plots/            ← training curves, confusion matrix, ROC, predictions
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Smoke-test (no real data required)
```bash
cd D:\dog_cat_cnn
python demo.py
```

### 3 — Prepare real data
Download the [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) and place images under `data/train/dogs/`, `data/train/cats/`, etc.

### 4 — Train
```bash
python train.py --data_dir data --epochs 30 --batch_size 32 --lr 0.001
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data` | Root data folder |
| `--epochs` | `30` | Max training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--dropout` | `0.5` | FC dropout rate |
| `--weight_decay` | `1e-4` | L2 regularisation |
| `--patience` | `7` | Early-stopping patience |
| `--val_split` | `0.2` | Val fraction (if val/ is empty) |
| `--save_dir` | `models` | Checkpoint directory |

### 5 — Evaluate
```bash
# Full validation-set evaluation
python evaluate.py --checkpoint models/best.pth --data_dir data

# Single image prediction
python evaluate.py --checkpoint models/best.pth --image path/to/photo.jpg
```

---

## Features
- **3 convolutional blocks** with BatchNorm + ReLU + MaxPooling
- **Data augmentation**: horizontal/vertical flip, rotation, colour jitter
- **Learning rate schedule**: ReduceLROnPlateau (halves LR when val loss stalls)
- **Early stopping** with best-weight restoration
- **Metrics**: Accuracy, BCE Loss, AUC-ROC, Confusion Matrix
- **GPU/CPU** auto-selection

---

## Expected Performance (Kaggle Dogs vs Cats)
| Metric | ~20 epochs |
|--------|-----------|
| Val Accuracy | 85-90% |
| AUC-ROC | 0.93-0.97 |

> For higher accuracy consider transfer learning with ResNet-50 or EfficientNet.
