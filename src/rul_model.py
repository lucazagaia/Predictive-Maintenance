"""
RUL model — Li et al. (2018) deep CNN for Remaining Useful Life estimation, in PyTorch.

The RUL counterpart of mvt_flow_model.py: the network definition (LiCNN) that both the
trainer (scripts/train_rul.py) and the inference interface (prediction.py) import. A
PyTorch state_dict stores weights only, so the class must live in importable code.

Architecture (Li, Ding, Sun 2018, FD001): four convolution layers with 10 filters of
length 10, then one convolution with a single filter of length 3, all tanh-activated;
the feature map is flattened, dropped out (0.5), and passed through a 100-unit
fully-connected tanh layer to a single linear RUL output.
"""

import torch
import torch.nn as nn


class LiCNN(nn.Module):
    """Li et al. (2018) 1-D CNN RUL regressor."""

    def __init__(self, n_features: int = 17, window: int = 30):
        super().__init__()
        # Convolution runs along the time axis. PyTorch Conv1d expects
        # (batch, channels, length) = (batch, features, window), so forward() transposes.
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 10, kernel_size=10, padding="same"), nn.Tanh(),
            nn.Conv1d(10, 10, kernel_size=10, padding="same"), nn.Tanh(),
            nn.Conv1d(10, 10, kernel_size=10, padding="same"), nn.Tanh(),
            nn.Conv1d(10, 10, kernel_size=10, padding="same"), nn.Tanh(),
            nn.Conv1d(10, 1, kernel_size=3, padding="same"), nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(window, 100), nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features) -> (batch, features, window)
        x = x.transpose(1, 2)
        return self.head(self.features(x)).squeeze(-1)   # (batch,)
