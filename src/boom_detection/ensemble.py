"""
Ensemble methods for boom detection.

Combines predictions from multiple models to achieve better accuracy
than any single model.

Usage:
    from boom_detection.ensemble import AdaptiveEnsemble

    ensemble = AdaptiveEnsemble()
    ensemble.add_model('cnn', cnn_trainer)
    ensemble.add_model('histgbm', histgbm_model)
    ensemble.fit(train_ids, train_boom_frames, cache)
    predictions = ensemble.predict(test_ids, cache)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import FeatureCache


class AdaptiveEnsemble:
    """
    Adaptive ensemble that learns weights from validation performance.

    Uses cross-validation to determine optimal weights for each model
    based on their individual performance.
    """

    def __init__(
        self,
        models: list[tuple[str, Any]] | None = None,
        weight_method: str = 'inverse_mae',  # 'inverse_mae', 'rank', 'softmax'
        cv_folds: int = 3,
    ):
        """
        Args:
            models: List of (name, model) tuples
            weight_method: How to compute weights from CV performance:
                - 'inverse_mae': weight = 1 / (MAE + epsilon)
                - 'rank': weight based on rank
                - 'softmax': softmax of negative MAE
            cv_folds: Number of CV folds for weight estimation
        """
        self.models = models or []
        self.weight_method = weight_method
        self.cv_folds = cv_folds
        self.weights_: np.ndarray | None = None

    def add_model(self, name: str, model: Any) -> None:
        """Add a model."""
        self.models.append((name, model))

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit all models and learn optimal weights."""
        n_models = len(self.models)

        if n_models == 0:
            raise ValueError("No models")

        # Cross-validation to estimate model performance
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        model_maes = np.zeros(n_models)
        model_counts = np.zeros(n_models)

        for train_idx, val_idx in kf.split(sim_ids):
            train_ids = [sim_ids[i] for i in train_idx]
            train_booms = boom_frames[train_idx]
            val_ids = [sim_ids[i] for i in val_idx]
            val_booms = boom_frames[val_idx]

            for j, (_, model) in enumerate(self.models):
                model.fit(train_ids, train_booms, cache)
                preds = model.predict(val_ids, cache)
                mae_val = np.mean(np.abs(preds - val_booms))
                model_maes[j] += mae_val
                model_counts[j] += 1

        model_maes /= model_counts

        # Compute weights from MAEs
        if self.weight_method == 'inverse_mae':
            self.weights_ = 1.0 / (model_maes + 1.0)
        elif self.weight_method == 'rank':
            ranks = np.argsort(np.argsort(model_maes)) + 1  # 1 = best
            self.weights_ = 1.0 / ranks
        elif self.weight_method == 'softmax':
            # Softmax of negative MAE (lower MAE = higher weight)
            scaled = -model_maes / (model_maes.std() + 1e-8)
            exp_scaled = np.exp(scaled - scaled.max())
            self.weights_ = exp_scaled / exp_scaled.sum()
        else:
            raise ValueError(f"Unknown weight_method: {self.weight_method}")

        self.weights_ /= self.weights_.sum()

        # Re-fit all models on full training data
        for _, model in self.models:
            model.fit(sim_ids, boom_frames, cache)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict using learned weights."""
        if self.weights_ is None:
            raise ValueError("Ensemble not fitted")

        all_preds = []
        for _, model in self.models:
            preds = model.predict(sim_ids, cache)
            all_preds.append(preds)

        all_preds = np.array(all_preds)
        weighted_preds = np.average(all_preds, axis=0, weights=self.weights_)
        return np.round(weighted_preds).astype(int)

    def get_weights(self) -> dict[str, float]:
        """Get learned weights for each model."""
        if self.weights_ is None:
            return {}
        return {name: w for (name, _), w in zip(self.models, self.weights_)}
