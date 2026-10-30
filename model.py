"""
CNN Model Definition for Dog-Cat Image Classification
Architecture: 3 Convolutional Layers + Fully Connected Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DogCatCNN(nn.Module):
    """
    3-Layer Convolutional Neural Network for Dog vs Cat Binary Classification.

    Architecture:
        Conv Layer 1: 3 -> 32 filters, 3x3 kernel, ReLU, MaxPool 2x2
        Conv Layer 2: 32 -> 64 filters, 3x3 kernel, ReLU, MaxPool 2x2
        Conv Layer 3: 64 -> 128 filters, 3x3 kernel, ReLU, MaxPool 2x2
        FC Layer 1  : Flatten -> 512 neurons, ReLU, Dropout(0.5)
        FC Layer 2  : 512 -> 1 neuron (Sigmoid for binary output)

    Input : (batch, 3, 128, 128) -- RGB images resized to 128x128
    Output: (batch, 1)           -- probability of being a dog (1) or cat (0)
    """

    def __init__(self, dropout_rate: float = 0.5):
        super(DogCatCNN, self).__init__()

        # -- Convolutional Block 1 ------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,   # RGB input
            out_channels=32,
            kernel_size=3,
            padding=1        # 'same' padding keeps spatial dims
        )
        self.bn1 = nn.BatchNorm2d(32)

        # -- Convolutional Block 2 ------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # -- Convolutional Block 3 ------------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # MaxPooling (shared across blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -- Fully Connected Head -------------------------------------------
        # After 3x MaxPool on 128x128 input -> 128 x (128/8) x (128/8) = 128 x 16 x 16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    # ------------------------------------------------------------------
    def summary(self):
        """Print a human-readable architecture summary."""
        print("=" * 60)
        print("         Dog-Cat CNN  --  Architecture Summary")
        print("=" * 60)
        print(f"  Conv1 : 3  -> 32  filters | 3x3 | BN | ReLU | MaxPool")
        print(f"  Conv2 : 32 -> 64  filters | 3x3 | BN | ReLU | MaxPool")
        print(f"  Conv3 : 64 -> 128 filters | 3x3 | BN | ReLU | MaxPool")
        print(f"  FC1   : {128*16*16} -> 512  | ReLU | Dropout(0.5)")
        print(f"  FC2   : 512 -> 1           | Sigmoid")
        print("=" * 60)
        total_params = sum(p.numel() for p in self.parameters())
        trainable   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters    : {total_params:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print("=" * 60)
