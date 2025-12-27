"""
Baseline predictors for boom detection.

These establish lower and upper bounds for performance and help validate
the evaluation framework.

Usage:
    from boom_detection.baselines import MeanPredictor, VarianceThresholdPredictor
    from boom_detection.evaluation import Evaluator

    evaluator = Evaluator(dataset)
    results = evaluator.cross_validate(MeanPredictor(), k=5)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .loader import Simulation
from .features import extract_features, FeatureExtractor


# =============================================================================
# Trivial Baselines
# =============================================================================

class MeanPredictor:
    """
    Predicts the mean of training targets for all samples.

    This is the simplest possible baseline - any useful model must beat this.
    """

    def __init__(self):
        self.mean_value: float = 0.0

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        self.mean_value = float(np.mean(y))

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        return np.full(len(X), self.mean_value)


class MedianPredictor:
    """
    Predicts the median of training targets for all samples.

    More robust to outliers than mean.
    """

    def __init__(self):
        self.median_value: float = 0.0

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        self.median_value = float(np.median(y))

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        return np.full(len(X), self.median_value)


class RandomPredictor:
    """
    Predicts random values within the training range.

    Useful for sanity checking - any model should beat this.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.min_val: float = 0.0
        self.max_val: float = 1.0

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        self.min_val = float(np.min(y))
        self.max_val = float(np.max(y))

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.uniform(self.min_val, self.max_val, size=len(X))


# =============================================================================
# Classical Signal-Based Methods
# =============================================================================

class VarianceThresholdPredictor:
    """
    Predicts boom frame based on when tip position variance crosses a threshold.

    This represents a classical approach: find when the "spread" metric
    exceeds some learned threshold.
    """

    def __init__(self, threshold_percentile: float = 50.0):
        """
        Args:
            threshold_percentile: What percentile of max variance to use as threshold.
                                  Lower = earlier detection, higher = later detection.
        """
        self.threshold_percentile = threshold_percentile
        self.threshold_fraction: float = 0.5

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        """Learn the optimal threshold fraction from training data."""
        # For each training sample, find what fraction of max variance
        # occurs at the boom frame
        fractions = []
        for sim, boom_frame in zip(X, y):
            features = extract_features(sim)
            # Use tip position variance (indices 2,3 for x2,y2 in variance features)
            tip_var = features[:, 2] + features[:, 3]  # var_x2 + var_y2
            max_var = np.max(tip_var)
            if max_var > 0:
                boom_frame = int(boom_frame)
                if boom_frame < len(tip_var):
                    fractions.append(tip_var[boom_frame] / max_var)

        if fractions:
            self.threshold_fraction = float(np.percentile(fractions, self.threshold_percentile))

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        """Predict boom frame as first frame exceeding threshold."""
        predictions = []
        for sim in X:
            features = extract_features(sim)
            tip_var = features[:, 2] + features[:, 3]
            max_var = np.max(tip_var)
            threshold = self.threshold_fraction * max_var

            # Find first frame exceeding threshold
            exceeds = np.where(tip_var >= threshold)[0]
            if len(exceeds) > 0:
                predictions.append(exceeds[0])
            else:
                predictions.append(len(tip_var) // 2)  # fallback to middle

        return np.array(predictions)


class DerivativeThresholdPredictor:
    """
    Predicts boom frame based on maximum rate of change of spread.

    The boom is often characterized by rapid increase in divergence,
    so the derivative peak may be more informative than absolute threshold.
    """

    def __init__(self, smoothing_window: int = 5):
        """
        Args:
            smoothing_window: Window size for smoothing derivatives
        """
        self.smoothing_window = smoothing_window
        self.peak_offset: int = 0  # learned offset from derivative peak to actual boom

    def _compute_spread_derivative(self, sim: Simulation) -> np.ndarray:
        """Compute smoothed derivative of tip spread."""
        features = extract_features(sim)
        tip_var = features[:, 2] + features[:, 3]

        # Take derivative
        deriv = np.diff(tip_var)

        # Smooth with moving average
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            deriv = np.convolve(deriv, kernel, mode='same')

        return deriv

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        """Learn offset from derivative peak to actual boom."""
        offsets = []
        for sim, boom_frame in zip(X, y):
            deriv = self._compute_spread_derivative(sim)
            peak_frame = int(np.argmax(deriv))
            offsets.append(int(boom_frame) - peak_frame)

        self.peak_offset = int(np.median(offsets))

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        """Predict boom as derivative peak + learned offset."""
        predictions = []
        for sim in X:
            deriv = self._compute_spread_derivative(sim)
            peak_frame = int(np.argmax(deriv))
            pred = peak_frame + self.peak_offset
            pred = max(0, min(pred, sim.frame_count - 1))
            predictions.append(pred)

        return np.array(predictions)


class SecondDerivativePredictor:
    """
    Predicts boom based on the inflection point (peak of second derivative).

    The boom might correspond to the point of maximum acceleration
    in the divergence process.
    """

    def __init__(self, smoothing_window: int = 10):
        self.smoothing_window = smoothing_window
        self.peak_offset: int = 0

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        offsets = []
        for sim, boom_frame in zip(X, y):
            features = extract_features(sim)
            tip_var = features[:, 2] + features[:, 3]

            # Second derivative
            d2 = np.diff(tip_var, n=2)

            # Smooth
            if self.smoothing_window > 1:
                kernel = np.ones(self.smoothing_window) / self.smoothing_window
                d2 = np.convolve(d2, kernel, mode='same')

            peak_frame = int(np.argmax(d2))
            offsets.append(int(boom_frame) - peak_frame)

        self.peak_offset = int(np.median(offsets))

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        predictions = []
        for sim in X:
            features = extract_features(sim)
            tip_var = features[:, 2] + features[:, 3]
            d2 = np.diff(tip_var, n=2)

            if self.smoothing_window > 1:
                kernel = np.ones(self.smoothing_window) / self.smoothing_window
                d2 = np.convolve(d2, kernel, mode='same')

            peak_frame = int(np.argmax(d2))
            pred = peak_frame + self.peak_offset
            pred = max(0, min(pred, sim.frame_count - 1))
            predictions.append(pred)

        return np.array(predictions)


# =============================================================================
# Feature-Based ML Baselines
# =============================================================================

class LinearRegressionPredictor:
    """
    Linear regression on summary statistics of the time series.

    Extracts simple summary stats (mean, max of various metrics) and
    fits a linear model to predict boom frame.
    """

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_extractor = FeatureExtractor()

    def _extract_summary(self, sim: Simulation) -> np.ndarray:
        """Extract summary statistics from simulation."""
        features = self.feature_extractor.transform(sim)

        # Summary stats across time
        summary = np.concatenate([
            np.mean(features, axis=0),
            np.max(features, axis=0),
            np.std(features, axis=0),
        ])

        return summary

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        # Extract features for all training samples
        features = np.array([self._extract_summary(sim) for sim in X])
        y = np.asarray(y)

        # Simple least squares (with regularization for stability)
        # (X^T X + λI)^-1 X^T y
        lambda_reg = 1.0
        XtX = features.T @ features + lambda_reg * np.eye(features.shape[1])
        Xty = features.T @ y
        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = 0.0  # included in features via intercept

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        features = np.array([self._extract_summary(sim) for sim in X])
        return features @ self.weights + self.bias


# =============================================================================
# Convenience Functions
# =============================================================================

def get_all_baselines() -> dict[str, object]:
    """Get a dictionary of all baseline predictors with default settings."""
    return {
        'mean': MeanPredictor(),
        'median': MedianPredictor(),
        'random': RandomPredictor(),
        'variance_threshold': VarianceThresholdPredictor(),
        'derivative_peak': DerivativeThresholdPredictor(),
        'second_derivative': SecondDerivativePredictor(),
        'linear_regression': LinearRegressionPredictor(),
    }
