"""
MVT-Flow Model for voraus-AD Anomaly Detection

This module provides the MVT-Flow (Multivariate Time-series Flow) implementation
for anomaly detection on robot sensor data.

Based on: Brockmann et al. (2023) - The voraus-AD Dataset for Anomaly Detection
in Robot Applications (arXiv:2311.04765)

Architecture:
- Real-NVP normalizing flow with 4 coupling blocks
- 1D convolutional internal networks for temporal modeling
- Soft-clamping for training stability
- Maximum likelihood training on normal data

Author: Luca Zagaia
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import pickle
from pathlib import Path


class CouplingBlock(nn.Module):
    """Real-NVP coupling block for multivariate time series."""
    
    def __init__(
        self,
        n_signals: int,
        n_timesteps: int,
        hidden_channels: int = 2,
        kernel_sizes: List[int] = [13, 1, 1],
        dilations: List[int] = [2, 1, 1],
        alpha: float = 1.9
    ):
        super().__init__()
        
        self.n_signals = n_signals
        self.n_timesteps = n_timesteps
        self.alpha = alpha
        
        perm = torch.randperm(n_signals)
        self.register_buffer('permutation', perm)
        
        self.split_size = n_signals // 2
        
        self.g1 = self._build_conv_net(
            self.split_size, 
            self.split_size * hidden_channels,
            kernel_sizes, 
            dilations
        )
        
        self.g2 = self._build_conv_net(
            self.split_size,
            self.split_size * hidden_channels,
            kernel_sizes,
            dilations
        )
        
    def _build_conv_net(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_sizes: List[int],
        dilations: List[int]
    ) -> nn.Sequential:
        """Build 1D convolutional network."""
        layers = []
        
        layers.extend([
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_sizes[0],
                padding=(kernel_sizes[0] - 1) * dilations[0] // 2,
                dilation=dilations[0]
            ),
            nn.ReLU()
        ])
        
        for i in range(1, len(kernel_sizes) - 1):
            layers.extend([
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_sizes[i],
                    padding=(kernel_sizes[i] - 1) * dilations[i] // 2,
                    dilation=dilations[i]
                ),
                nn.ReLU()
            ])
        
        layers.append(
            nn.Conv1d(
                hidden_channels,
                in_channels * 2,
                kernel_size=kernel_sizes[-1],
                padding=(kernel_sizes[-1] - 1) * dilations[-1] // 2,
                dilation=dilations[-1]
            )
        )
        
        return nn.Sequential(*layers)
    
    def _soft_clamp(self, s: torch.Tensor) -> torch.Tensor:
        """Soft-clamping to prevent exploding gradients."""
        return (2 * self.alpha / np.pi) * torch.atan(s * np.pi / (2 * self.alpha))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x -> z"""
        x = x[:, self.permutation, :]
        x1, x2 = torch.split(x, self.split_size, dim=1)
        
        out1 = self.g1(x1)
        s1, t1 = torch.chunk(out1, 2, dim=1)
        s1 = self._soft_clamp(s1)
        y2 = x2 * torch.exp(s1) + t1
        
        out2 = self.g2(y2)
        s2, t2 = torch.chunk(out2, 2, dim=1)
        s2 = self._soft_clamp(s2)
        y1 = x1 * torch.exp(s2) + t2
        
        y = torch.cat([y1, y2], dim=1)
        log_det_jacobian = torch.sum(s1, dim=[1, 2]) + torch.sum(s2, dim=[1, 2])
        
        return y, log_det_jacobian
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z -> x"""
        y1, y2 = torch.split(y, self.split_size, dim=1)
        
        out2 = self.g2(y2)
        s2, t2 = torch.chunk(out2, 2, dim=1)
        s2 = self._soft_clamp(s2)
        x1 = (y1 - t2) * torch.exp(-s2)
        
        out1 = self.g1(x1)
        s1, t1 = torch.chunk(out1, 2, dim=1)
        s1 = self._soft_clamp(s1)
        x2 = (y2 - t1) * torch.exp(-s1)
        
        x = torch.cat([x1, x2], dim=1)
        
        inv_permutation = torch.argsort(self.permutation)
        x = x[:, inv_permutation, :]
        
        return x


class MVTFlow(nn.Module):
    """MVT-Flow: Multivariate Time-series Normalizing Flow"""
    
    def __init__(
        self,
        n_signals: int,
        n_timesteps: int,
        n_blocks: int = 4,
        hidden_channels: int = 2,
        kernel_sizes: List[int] = [13, 1, 1],
        dilations: List[int] = [2, 1, 1],
        alpha: float = 1.9
    ):
        super().__init__()
        
        self.n_signals = n_signals
        self.n_timesteps = n_timesteps
        self.n_blocks = n_blocks
        
        self.blocks = nn.ModuleList([
            CouplingBlock(
                n_signals,
                n_timesteps,
                hidden_channels,
                kernel_sizes,
                dilations,
                alpha
            ) for _ in range(n_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: data -> latent space"""
        z = x
        log_det_jacobian = 0
        
        for block in self.blocks:
            z, ldj = block(z)
            log_det_jacobian += ldj
        
        return z, log_det_jacobian
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: latent space -> data"""
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of data under the model."""
        z, log_det_jacobian = self.forward(x)
        log_pz = -0.5 * torch.sum(z ** 2, dim=[1, 2])
        log_px = log_pz + log_det_jacobian
        return log_px
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss."""
        log_prob = self.log_prob(x)
        return -torch.mean(log_prob)
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score (negative log probability)."""
        with torch.no_grad():
            log_prob = self.log_prob(x)
        return -log_prob


class MVTFlowDetector:
    """
    Production-ready wrapper for MVT-Flow anomaly detection.
    
    This class provides a simplified interface for using MVT-Flow in the
    detection pipeline, handling model loading, preprocessing, and inference.
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        n_signals: int = 130,
        n_timesteps: int = 1100,
        device: str = None
    ):
        """
        Initialize MVT-Flow detector.
        
        Args:
            model_path: Path to trained .pt model file
            scaler_path: Path to fitted StandardScaler .pkl file
            n_signals: Number of sensor signals (default: 130 for voraus-AD)
            n_timesteps: Length of time series windows (default: 1100)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.n_signals = n_signals
        self.n_timesteps = n_timesteps
        
        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = MVTFlow(
            n_signals=n_signals,
            n_timesteps=n_timesteps,
            n_blocks=4,
            hidden_channels=2,
            kernel_sizes=[13, 1, 1],
            dilations=[2, 1, 1],
            alpha=1.9
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"MVT-Flow detector loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Signals: {n_signals}, Timesteps: {n_timesteps}")
    
    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw sensor data for MVT-Flow.
        
        Args:
            X: Raw sensor data of shape (n_samples, n_signals, n_timesteps)
        
        Returns:
            Preprocessed tensor ready for model input
        """
        n_samples, n_signals, n_timesteps = X.shape
        
        # Reshape for standardization
        X_flat = X.transpose(0, 2, 1).reshape(-1, n_signals)
        
        # Standardize
        X_scaled = self.scaler.transform(X_flat)
        
        # Reshape back
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_signals).transpose(0, 2, 1)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        return X_tensor
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for sensor data.
        
        Args:
            X: Raw sensor data of shape (n_samples, n_signals, n_timesteps)
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        X_tensor = self.preprocess(X)
        
        with torch.no_grad():
            scores = self.model.anomaly_score(X_tensor)
        
        return scores.cpu().numpy()
    

