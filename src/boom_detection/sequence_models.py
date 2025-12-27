"""
Sequence models for boom detection using PyTorch.

These models process the full temporal sequence of features to predict
the boom frame, leveraging temporal patterns that frame-level models miss.

Usage:
    from boom_detection.sequence_models import CNNClassifier, LSTMClassifier

    model = CNNClassifier(n_features=183)
    trainer = SequenceTrainer(model, device='cuda')
    trainer.fit(train_ids, train_boom_frames, cache)
    predictions = trainer.predict(test_ids, cache)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .features import FeatureCache


# =============================================================================
# PyTorch Models
# =============================================================================

class CNNClassifier(nn.Module):
    """
    1D CNN for frame-level classification (before/after boom).

    Uses multi-scale convolutions to capture patterns at different temporal scales.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 7, 15),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_branches = len(kernel_sizes)
        self.branch_dim = hidden_dim // self.n_branches

        # Multi-scale conv branches
        self.conv_branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(n_features, self.branch_dim, ks, padding=ks // 2),
                nn.BatchNorm1d(self.branch_dim),
                nn.ReLU(),
                nn.Conv1d(self.branch_dim, self.branch_dim, ks, padding=ks // 2),
                nn.BatchNorm1d(self.branch_dim),
                nn.ReLU(),
            )
            self.conv_branches.append(branch)

        # Combined processing - actual combined dim is branch_dim * n_branches
        combined_dim = self.branch_dim * self.n_branches
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(combined_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, 1),  # Output: probability per frame
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, frames, features)

        Returns:
            (batch, frames) - logits for each frame
        """
        # (batch, frames, features) -> (batch, features, frames)
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        branch_outputs = [branch(x) for branch in self.conv_branches]
        x = torch.cat(branch_outputs, dim=1)  # (batch, hidden_dim, frames)

        # Per-frame classification
        x = self.fc(x)  # (batch, 1, frames)
        return x.squeeze(1)  # (batch, frames)


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for frame-level classification.

    Captures long-range temporal dependencies.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,  # Half for each direction
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, frames, features)

        Returns:
            (batch, frames) - logits for each frame
        """
        x = self.input_proj(x)  # (batch, frames, hidden_dim)
        x, _ = self.lstm(x)  # (batch, frames, hidden_dim)
        x = self.fc(x)  # (batch, frames, 1)
        return x.squeeze(-1)  # (batch, frames)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for frame-level classification.

    Uses self-attention to capture temporal dependencies.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        max_len: int = 2000,
    ):
        super().__init__()
        self.n_features = n_features

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, frames, features)

        Returns:
            (batch, frames) - logits for each frame
        """
        seq_len = x.size(1)
        x = self.input_proj(x)  # (batch, frames, hidden_dim)
        x = x + self.pos_encoding[:, :seq_len, :]  # Add positional encoding
        x = self.transformer(x)  # (batch, frames, hidden_dim)
        x = self.fc(x)  # (batch, frames, 1)
        return x.squeeze(-1)  # (batch, frames)


# =============================================================================
# Dataset
# =============================================================================

class BoomDataset(Dataset):
    """Dataset for sequence models."""

    def __init__(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
        augment: bool = False,
    ):
        self.sim_ids = sim_ids
        self.boom_frames = boom_frames
        self.cache = cache
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sim_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sim_id = self.sim_ids[idx]
        boom_frame = int(self.boom_frames[idx])
        features = self.cache[sim_id]  # (frames, n_features)

        # Data augmentation
        if self.augment:
            features = self._augment(features)

        # Create labels: 1 if frame >= boom_frame, else 0
        n_frames = len(features)
        labels = (np.arange(n_frames) >= boom_frame).astype(np.float32)

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(labels),
            boom_frame,
        )

    def _augment(self, features: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Add small noise
        if np.random.random() < 0.5:
            noise = np.random.randn(*features.shape) * 0.01
            features = features + noise

        # Random time scaling (stretch/compress slightly)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.95, 1.05)
            n_frames = len(features)
            new_n_frames = int(n_frames * scale)
            if new_n_frames > 0:
                indices = np.linspace(0, n_frames - 1, new_n_frames)
                features = np.array([features[int(i)] for i in indices])

        return features


def collate_variable_length(batch):
    """Collate function for variable-length sequences."""
    features_list, labels_list, boom_frames = zip(*batch)

    # Pad to max length in batch
    max_len = max(f.size(0) for f in features_list)
    n_features = features_list[0].size(1)

    padded_features = torch.zeros(len(batch), max_len, n_features)
    padded_labels = torch.zeros(len(batch), max_len)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (features, labels) in enumerate(zip(features_list, labels_list)):
        seq_len = features.size(0)
        padded_features[i, :seq_len] = features
        padded_labels[i, :seq_len] = labels
        masks[i, :seq_len] = True

    return padded_features, padded_labels, masks, torch.tensor(boom_frames)


# =============================================================================
# Trainer
# =============================================================================

class SequenceTrainer:
    """
    Trainer for sequence models with the same interface as frame-level models.

    Compatible with cross_validate_cached().
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 8,
        patience: int = 10,
        augment: bool = True,
        verbose: bool = False,
    ):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.augment = augment
        self.verbose = verbose

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Train the model."""
        # Reset model weights
        self._reset_weights()

        dataset = BoomDataset(sim_ids, boom_frames, cache, augment=self.augment)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_variable_length,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for features, labels, masks, _ in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                logits = self.model(features)  # (batch, frames)

                # Masked BCE loss
                loss = F.binary_cross_entropy_with_logits(
                    logits[masks], labels[masks]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}: loss={avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_best_weights()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        self._load_best_weights()

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict boom frames."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for sim_id in sim_ids:
                features = cache[sim_id]
                features_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
                features_t = features_t.to(self.device)

                logits = self.model(features_t)  # (1, frames)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                # Find first frame where prob >= 0.5
                crossings = np.where(probs >= 0.5)[0]
                if len(crossings) > 0:
                    boom_pred = crossings[0]
                else:
                    boom_pred = np.argmax(probs)

                predictions.append(int(boom_pred))

        return np.array(predictions)

    def _reset_weights(self):
        """Reset model weights for fresh training."""
        for module in self.model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _save_best_weights(self):
        """Save current weights as best."""
        self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def _load_best_weights(self):
        """Load best weights."""
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)


# =============================================================================
# Factory Functions
# =============================================================================

def get_sequence_models(n_features: int) -> dict[str, SequenceTrainer]:
    """Get all sequence model trainers for comparison."""
    return {
        'cnn_classifier': SequenceTrainer(
            CNNClassifier(n_features),
            epochs=50,
            patience=10,
        ),
        'lstm_classifier': SequenceTrainer(
            LSTMClassifier(n_features),
            epochs=50,
            patience=10,
        ),
        'transformer_classifier': SequenceTrainer(
            TransformerClassifier(n_features),
            epochs=50,
            patience=10,
        ),
    }
