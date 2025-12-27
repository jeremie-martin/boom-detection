"""
Ensemble methods for boom detection.

Combines predictions from multiple models to achieve better accuracy
than any single model.

Usage:
    from boom_detection.ensemble import WeightedEnsemble, StackingEnsemble

    ensemble = WeightedEnsemble([
        ('cnn', cnn_trainer, 0.5),
        ('histgbm', histgbm_model, 0.3),
        ('cusum', cusum_detector, 0.2),
    ])
    ensemble.fit(train_ids, train_boom_frames, cache)
    predictions = ensemble.predict(test_ids, cache)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import FeatureCache


class WeightedEnsemble:
    """
    Combine multiple models using weighted averaging.

    Each model predicts a boom frame, and the final prediction is
    a weighted average of all predictions.
    """

    def __init__(
        self,
        models: list[tuple[str, Any, float]] | None = None,
        combination: str = 'weighted_mean',  # 'weighted_mean', 'weighted_median', 'vote'
    ):
        """
        Args:
            models: List of (name, model, weight) tuples.
                   Models must implement fit(sim_ids, boom_frames, cache)
                   and predict(sim_ids, cache).
            combination: How to combine predictions:
                - 'weighted_mean': Weighted average
                - 'weighted_median': Weighted median
                - 'vote': Majority vote (rounded to nearest 10 frames)
        """
        self.models = models or []
        self.combination = combination

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models.append((name, model, weight))

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit all models in the ensemble."""
        for _, model, _ in self.models:
            model.fit(sim_ids, boom_frames, cache)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict by combining all model predictions."""
        if not self.models:
            raise ValueError("No models in ensemble")

        # Collect predictions from all models
        all_preds = []
        weights = []
        for _, model, weight in self.models:
            preds = model.predict(sim_ids, cache)
            all_preds.append(preds)
            weights.append(weight)

        all_preds = np.array(all_preds)  # (n_models, n_samples)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        if self.combination == 'weighted_mean':
            return np.round(np.average(all_preds, axis=0, weights=weights)).astype(int)

        elif self.combination == 'weighted_median':
            # Weighted median for each sample
            predictions = []
            for i in range(all_preds.shape[1]):
                preds = all_preds[:, i]
                # Sort by prediction value
                sorted_idx = np.argsort(preds)
                sorted_preds = preds[sorted_idx]
                sorted_weights = weights[sorted_idx]
                cumsum = np.cumsum(sorted_weights)
                median_idx = np.searchsorted(cumsum, 0.5)
                predictions.append(int(sorted_preds[min(median_idx, len(sorted_preds) - 1)]))
            return np.array(predictions)

        elif self.combination == 'vote':
            # Bin predictions and vote
            predictions = []
            for i in range(all_preds.shape[1]):
                preds = all_preds[:, i]
                # Round to nearest 10 frames
                binned = (preds / 10).astype(int) * 10
                # Weighted vote
                unique_bins = np.unique(binned)
                bin_weights = {b: 0.0 for b in unique_bins}
                for pred, w in zip(binned, weights):
                    bin_weights[pred] += w
                best_bin = max(bin_weights, key=bin_weights.get)
                predictions.append(int(best_bin))
            return np.array(predictions)

        else:
            raise ValueError(f"Unknown combination: {self.combination}")


class StackingEnsemble:
    """
    Stacking ensemble: use base model predictions as features for a meta-model.

    The meta-model learns the optimal way to combine base model predictions
    based on their errors on validation data.
    """

    def __init__(
        self,
        base_models: list[tuple[str, Any]] | None = None,
        meta_model: Any = None,
    ):
        """
        Args:
            base_models: List of (name, model) tuples for base models
            meta_model: Model that combines base predictions (default: Ridge)
        """
        self.base_models = base_models or []
        self.meta_model = meta_model
        self._meta_fitted = False

    def add_base_model(self, name: str, model: Any) -> None:
        """Add a base model."""
        self.base_models.append((name, model))

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """
        Fit base models and meta-model.

        Uses leave-one-out cross-validation on training data to generate
        meta-features for training the meta-model.
        """
        n_samples = len(sim_ids)
        n_models = len(self.base_models)

        if n_models == 0:
            raise ValueError("No base models")

        # Initialize meta-model if not provided
        if self.meta_model is None:
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)

        # Leave-one-out to generate meta-features
        meta_features = np.zeros((n_samples, n_models))

        for i in range(n_samples):
            # Train on all except i
            train_ids = [s for j, s in enumerate(sim_ids) if j != i]
            train_booms = np.array([b for j, b in enumerate(boom_frames) if j != i])
            test_id = [sim_ids[i]]

            # Get predictions from each base model
            for j, (_, model) in enumerate(self.base_models):
                model.fit(train_ids, train_booms, cache)
                pred = model.predict(test_id, cache)
                meta_features[i, j] = pred[0]

        # Fit meta-model
        self.meta_model.fit(meta_features, boom_frames)
        self._meta_fitted = True

        # Re-fit base models on full training data
        for _, model in self.base_models:
            model.fit(sim_ids, boom_frames, cache)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict using stacked ensemble."""
        if not self._meta_fitted:
            raise ValueError("Ensemble not fitted")

        # Get base model predictions
        n_models = len(self.base_models)
        meta_features = np.zeros((len(sim_ids), n_models))

        for j, (_, model) in enumerate(self.base_models):
            preds = model.predict(sim_ids, cache)
            meta_features[:, j] = preds

        # Combine with meta-model
        predictions = self.meta_model.predict(meta_features)
        return np.round(predictions).astype(int)


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
        n_samples = len(sim_ids)
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


def create_default_ensemble(
    n_features: int,
    include_cnn: bool = True,
    include_lstm: bool = False,
) -> AdaptiveEnsemble:
    """
    Create a default ensemble with the best-performing models.

    Args:
        n_features: Number of input features
        include_cnn: Include CNN model (requires PyTorch)
        include_lstm: Include LSTM model (requires PyTorch)

    Returns:
        AdaptiveEnsemble with pre-configured models
    """
    from .frame_models import FrameLevelClassifier
    from .changepoint import MultiFeatureCUSUM

    ensemble = AdaptiveEnsemble(weight_method='inverse_mae')

    # HistGBM classifier (best non-neural)
    ensemble.add_model('histgbm', FrameLevelClassifier(
        model='hist_gbm',
        n_estimators=200,
        max_depth=7,
    ))

    # Multi-feature CUSUM
    ensemble.add_model('cusum', MultiFeatureCUSUM(
        n_features=5,
        combination='median',
    ))

    # Neural models if requested
    if include_cnn:
        try:
            from .sequence_models import CNNClassifier, SequenceTrainer
            cnn = SequenceTrainer(
                CNNClassifier(n_features),
                epochs=50,
                patience=10,
            )
            ensemble.add_model('cnn', cnn)
        except ImportError:
            pass

    if include_lstm:
        try:
            from .sequence_models import LSTMClassifier, SequenceTrainer
            lstm = SequenceTrainer(
                LSTMClassifier(n_features),
                epochs=50,
                patience=10,
            )
            ensemble.add_model('lstm', lstm)
        except ImportError:
            pass

    return ensemble
