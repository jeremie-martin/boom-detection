"""
Models for predicting boom quality.

Boom quality (0-1) indicates how clear and well-defined the boom moment is:
- High quality (0.8-1.0): Clear separation, sharp convergence, clean caustics
- Low quality (0.0-0.3): Multiple groups, fuzzy convergence, weak caustics

Usage:
    from boom_detection.quality_models import QualityRegressor, QualityClassifier

    model = QualityRegressor(model='hist_gbm')
    model.fit(train_ids, train_qualities, cache)
    predictions = model.predict(test_ids, cache)
"""

from __future__ import annotations

import numpy as np

from .features import FeatureCache


class MeanQualityPredictor:
    """Predict mean quality from training set."""

    def __init__(self):
        self.mean_value = 0.5

    def fit(self, sim_ids: list[str], qualities: np.ndarray, cache: FeatureCache) -> None:
        self.mean_value = float(np.mean(qualities))

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        return np.full(len(sim_ids), self.mean_value)


class MedianQualityPredictor:
    """Predict median quality from training set."""

    def __init__(self):
        self.median_value = 0.5

    def fit(self, sim_ids: list[str], qualities: np.ndarray, cache: FeatureCache) -> None:
        self.median_value = float(np.median(qualities))

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        return np.full(len(sim_ids), self.median_value)


class QualityRegressor:
    """
    Predict boom quality using simulation-level features.

    Aggregates frame-level features to simulation-level, then uses regression.
    """

    def __init__(
        self,
        model: str = 'ridge',
        alpha: float = 1.0,
        n_estimators: int = 100,
        max_depth: int = 5,
        seed: int | None = None,
    ):
        self.model_type = model
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self._model = None
        self._feature_mean = None
        self._feature_std = None

    def _extract_sim_features(self, features: np.ndarray) -> np.ndarray:
        """Extract simulation-level features from frame-level features."""
        # Aggregate statistics over time
        return np.concatenate([
            features.mean(axis=0),      # Mean of each feature over time
            features.std(axis=0),       # Std (temporal variability)
            features.max(axis=0),       # Peak values
            np.percentile(features, 90, axis=0),  # 90th percentile
            np.percentile(features, 10, axis=0),  # 10th percentile
        ])

    def _extract_spike_features(self, features: np.ndarray) -> np.ndarray:
        """Extract features about the variance spike (boom signature)."""
        # Focus on tip variance (indices 2, 3 typically)
        tip_var = features[:, 2] + features[:, 3] if features.shape[1] > 3 else features[:, 0]

        # Spike characteristics
        deriv = np.gradient(tip_var)
        deriv2 = np.gradient(deriv)

        return np.array([
            np.max(tip_var),                    # Max variance
            np.max(deriv),                      # Max rate of increase
            np.max(np.abs(deriv2)),             # Max curvature (sharpness)
            np.argmax(deriv) / len(tip_var),    # Relative position of spike
            np.std(deriv),                      # Variability of rate
        ])

    def fit(self, sim_ids: list[str], qualities: np.ndarray, cache: FeatureCache) -> None:
        """Fit regressor on simulation-level features."""
        # Extract features for all simulations
        X = []
        for sim_id in sim_ids:
            features = cache[sim_id]
            sim_features = self._extract_sim_features(features)
            spike_features = self._extract_spike_features(features)
            X.append(np.concatenate([sim_features, spike_features]))

        X = np.array(X)

        # Normalize
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X = (X - self._feature_mean) / self._feature_std

        # Create model
        if self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=self.alpha)
        elif self.model_type == 'hist_gbm':
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed if self.seed is not None else 42,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

        self._model.fit(X, qualities)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict quality for simulations."""
        X = []
        for sim_id in sim_ids:
            features = cache[sim_id]
            sim_features = self._extract_sim_features(features)
            spike_features = self._extract_spike_features(features)
            X.append(np.concatenate([sim_features, spike_features]))

        X = np.array(X)
        X = (X - self._feature_mean) / self._feature_std

        predictions = self._model.predict(X)
        return np.clip(predictions, 0, 1)  # Ensure valid range


class QualityClassifier:
    """
    Binary classifier for high/low quality booms.

    Threshold of 0.5 splits quality into high (>=0.5) and low (<0.5).
    Outputs probability of high quality.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model: str = 'hist_gbm',
        n_estimators: int = 100,
        max_depth: int = 5,
        seed: int | None = None,
    ):
        self.threshold = threshold
        self.model_type = model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self._model = None
        self._feature_mean = None
        self._feature_std = None

    def _extract_features(self, features: np.ndarray) -> np.ndarray:
        """Extract simulation-level features."""
        # Same as QualityRegressor
        sim_features = np.concatenate([
            features.mean(axis=0),
            features.std(axis=0),
            features.max(axis=0),
        ])

        # Spike features
        tip_var = features[:, 2] + features[:, 3] if features.shape[1] > 3 else features[:, 0]
        deriv = np.gradient(tip_var)
        deriv2 = np.gradient(deriv)

        spike_features = np.array([
            np.max(tip_var),
            np.max(deriv),
            np.max(np.abs(deriv2)),
            np.argmax(deriv) / len(tip_var),
            np.std(deriv),
        ])

        return np.concatenate([sim_features, spike_features])

    def fit(self, sim_ids: list[str], qualities: np.ndarray, cache: FeatureCache) -> None:
        """Fit classifier."""
        X = np.array([self._extract_features(cache[sid]) for sid in sim_ids])

        # Normalize
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X = (X - self._feature_mean) / self._feature_std

        # Convert to binary labels
        y = (qualities >= self.threshold).astype(int)

        # Create model
        if self.model_type == 'hist_gbm':
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed if self.seed is not None else 42,
            )
        elif self.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(
                max_iter=1000,
                random_state=self.seed if self.seed is not None else 42,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

        self._model.fit(X, y)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict quality class (0 or 1)."""
        X = np.array([self._extract_features(cache[sid]) for sid in sim_ids])
        X = (X - self._feature_mean) / self._feature_std
        return self._model.predict(X)

    def predict_proba(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict probability of high quality."""
        X = np.array([self._extract_features(cache[sid]) for sid in sim_ids])
        X = (X - self._feature_mean) / self._feature_std
        return self._model.predict_proba(X)[:, 1]


class BoomAwareQualityPredictor:
    """
    Predict quality using features extracted around the estimated boom frame.

    Pipeline:
    1. Estimate boom frame using a frame detector
    2. Extract features from a window around the estimated boom
    3. Predict quality from those features
    """

    def __init__(
        self,
        frame_model=None,
        window_size: int = 50,
        model: str = 'hist_gbm',
        seed: int | None = None,
    ):
        self.frame_model = frame_model
        self.window_size = window_size
        self.model_type = model
        self.seed = seed
        self._quality_model = None
        self._feature_mean = None
        self._feature_std = None

    def _extract_boom_features(
        self,
        features: np.ndarray,
        boom_frame: int,
    ) -> np.ndarray:
        """Extract features from window around boom frame."""
        n_frames = len(features)
        start = max(0, boom_frame - self.window_size)
        end = min(n_frames, boom_frame + self.window_size)

        window = features[start:end]

        # Features about the boom window
        return np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.max(axis=0) - window.min(axis=0),  # Range in window
            features[boom_frame] if boom_frame < n_frames else features[-1],  # Features at boom
        ])

    def fit(
        self,
        sim_ids: list[str],
        qualities: np.ndarray,
        cache: FeatureCache,
        boom_frames: np.ndarray | None = None,
    ) -> None:
        """
        Fit quality predictor.

        Args:
            sim_ids: Simulation IDs
            qualities: Quality scores (0-1)
            cache: Feature cache
            boom_frames: Ground truth boom frames (for training).
                        If None, uses frame_model to estimate.
        """
        # If no boom frames provided, need to estimate them
        if boom_frames is None and self.frame_model is not None:
            # Use LOOCV to get boom frame estimates
            from sklearn.model_selection import LeaveOneOut
            boom_frames = np.zeros(len(sim_ids))
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(sim_ids):
                train_ids = [sim_ids[i] for i in train_idx]
                test_ids = [sim_ids[test_idx[0]]]
                # Need boom frames for training the frame model
                # This is a chicken-and-egg problem - use ground truth for training
                # This method is best used when ground truth boom_frames are available
                pass
            # For simplicity, if boom_frames not provided, use middle of simulation
            boom_frames = np.array([len(cache[sid]) // 2 for sid in sim_ids])

        # Extract features around boom
        X = []
        for sim_id, boom in zip(sim_ids, boom_frames.astype(int)):
            features = cache[sim_id]
            boom_features = self._extract_boom_features(features, int(boom))
            X.append(boom_features)

        X = np.array(X)

        # Normalize
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X = (X - self._feature_mean) / self._feature_std

        # Train model
        if self.model_type == 'hist_gbm':
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._quality_model = HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=5,
                random_state=self.seed if self.seed is not None else 42,
            )
        else:
            from sklearn.linear_model import Ridge
            self._quality_model = Ridge(alpha=1.0)

        self._quality_model.fit(X, qualities)

    def predict(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
        boom_frames: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict quality.

        Args:
            sim_ids: Simulation IDs
            cache: Feature cache
            boom_frames: Estimated boom frames. If None, uses middle of simulation.
        """
        if boom_frames is None:
            boom_frames = np.array([len(cache[sid]) // 2 for sid in sim_ids])

        X = []
        for sim_id, boom in zip(sim_ids, boom_frames.astype(int)):
            features = cache[sim_id]
            boom_features = self._extract_boom_features(features, int(boom))
            X.append(boom_features)

        X = np.array(X)
        X = (X - self._feature_mean) / self._feature_std

        predictions = self._quality_model.predict(X)
        return np.clip(predictions, 0, 1)


def get_quality_predictors() -> dict[str, object]:
    """Get all quality prediction models."""
    return {
        'quality_mean': MeanQualityPredictor(),
        'quality_median': MedianQualityPredictor(),
        'quality_ridge': QualityRegressor(model='ridge'),
        'quality_histgbm': QualityRegressor(model='hist_gbm'),
        'quality_classifier': QualityClassifier(model='hist_gbm'),
    }
