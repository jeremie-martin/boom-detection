"""
Frame-level models for boom detection.

These models leverage all ~50k frame observations instead of just 49 simulation-level
samples, dramatically increasing effective training data.

Usage:
    from boom_detection.frame_models import FrameLevelRegressor

    model = FrameLevelRegressor(model='gbm')
    model.fit(train_ids, train_boom_frames, cache)
    predictions = model.predict(test_ids, cache)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .features import FeatureCache


class FrameLevelRegressor:
    """
    Predict boom frame by regressing distance-to-boom for each frame.

    For each frame t, predicts: distance = boom_frame - t
    At inference, finds where predicted distance ≈ 0.

    This leverages all ~50k frame observations instead of 49 simulation samples.
    """

    def __init__(
        self,
        model: Literal['ridge', 'gbm', 'hist_gbm'] = 'ridge',
        alpha: float = 1.0,  # Ridge regularization
        n_estimators: int = 100,  # GBM trees
        max_depth: int = 5,  # GBM depth
        aggregation: Literal['median', 'zero_crossing', 'weighted'] = 'median',
    ):
        self.model_type = model
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.aggregation = aggregation
        self._model = None

    def fit(self, sim_ids: list[str], boom_frames: np.ndarray, cache: FeatureCache) -> None:
        """
        Fit on all frames from training simulations.

        Args:
            sim_ids: List of simulation IDs
            boom_frames: Array of boom frame numbers
            cache: FeatureCache with extracted features
        """
        # Collect all frame-level training data
        X_frames = []
        y_distances = []

        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]  # (frames, n_features)
            n_frames = len(features)
            # Distance: positive before boom, negative after
            distances = int(boom) - np.arange(n_frames)
            X_frames.append(features)
            y_distances.append(distances)

        X = np.vstack(X_frames)
        y = np.concatenate(y_distances)

        # Create and fit model
        if self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=self.alpha)
        elif self.model_type == 'gbm':
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
        elif self.model_type == 'hist_gbm':
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._model.fit(X, y)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """
        Predict boom frame for each simulation.

        For each simulation:
        1. Predict distance-to-boom for all frames
        2. Aggregate predictions based on chosen strategy
        """
        predictions = []

        for sim_id in sim_ids:
            features = cache[sim_id]
            n_frames = len(features)

            # Predict distance for each frame
            # Positive = before boom, negative = after boom
            distances = self._model.predict(features)

            if self.aggregation == 'median':
                # Each frame's prediction of where the boom is
                predicted_booms = np.arange(n_frames) + distances
                boom_pred = np.median(predicted_booms)

            elif self.aggregation == 'zero_crossing':
                # Find where distance crosses zero (like classifier threshold)
                # Distance goes from positive (before boom) to negative (after boom)
                crossings = np.where(distances[:-1] * distances[1:] <= 0)[0]
                if len(crossings) > 0:
                    # Use first crossing point
                    idx = crossings[0]
                    # Linear interpolation to find exact crossing
                    d0, d1 = distances[idx], distances[idx + 1]
                    if d1 != d0:
                        t = -d0 / (d1 - d0)
                        boom_pred = idx + t
                    else:
                        boom_pred = idx
                else:
                    # No crossing found - use frame with smallest absolute distance
                    boom_pred = np.argmin(np.abs(distances))

            elif self.aggregation == 'weighted':
                # Weight each frame's prediction by inverse of predicted distance magnitude
                # Frames predicting distance ≈ 0 get highest weight
                predicted_booms = np.arange(n_frames) + distances
                weights = 1.0 / (np.abs(distances) + 1.0)  # +1 to avoid div by zero
                boom_pred = np.average(predicted_booms, weights=weights)

            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            # Clip to valid range
            boom_pred = int(np.clip(boom_pred, 0, n_frames - 1))
            predictions.append(boom_pred)

        return np.array(predictions)


class FrameLevelClassifier:
    """
    Predict boom frame via binary classification: before vs after boom.

    For each frame, predicts P(after_boom | features).
    At inference, finds where probability crosses 0.5.

    This frames boom detection as finding a decision boundary in time.
    """

    def __init__(
        self,
        model: Literal['logistic', 'gbm', 'hist_gbm'] = 'hist_gbm',
        C: float = 1.0,  # Logistic regularization (inverse)
        n_estimators: int = 100,  # GBM trees
        max_depth: int = 5,  # GBM depth
    ):
        self.model_type = model
        self.C = C
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None

    def fit(self, sim_ids: list[str], boom_frames: np.ndarray, cache: FeatureCache) -> None:
        """
        Fit on all frames from training simulations.

        Labels: 0 = before boom, 1 = after boom (inclusive).
        """
        X_frames = []
        y_labels = []

        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]
            n_frames = len(features)
            # Label: 1 if frame >= boom_frame, else 0
            labels = (np.arange(n_frames) >= int(boom)).astype(int)
            X_frames.append(features)
            y_labels.append(labels)

        X = np.vstack(X_frames)
        y = np.concatenate(y_labels)

        # Create and fit model
        if self.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        elif self.model_type == 'gbm':
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
        elif self.model_type == 'hist_gbm':
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._model.fit(X, y)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """
        Predict boom frame for each simulation.

        Find where P(after_boom) crosses 0.5.
        """
        predictions = []

        for sim_id in sim_ids:
            features = cache[sim_id]
            n_frames = len(features)

            # Get probability of being after boom
            probs = self._model.predict_proba(features)[:, 1]

            # Find first frame where prob >= 0.5
            after_boom = probs >= 0.5
            crossings = np.where(after_boom)[0]

            if len(crossings) > 0:
                boom_pred = crossings[0]
            else:
                # No crossing found, use frame with prob closest to 0.5
                boom_pred = np.argmin(np.abs(probs - 0.5))

            predictions.append(int(boom_pred))

        return np.array(predictions)


def get_frame_level_predictors() -> dict[str, object]:
    """Get all frame-level predictors for comparison."""
    return {
        'frame_ridge': FrameLevelRegressor(model='ridge'),
        'frame_gbm': FrameLevelRegressor(model='gbm'),
        'frame_logistic': FrameLevelClassifier(model='logistic'),
        'frame_gbm_clf': FrameLevelClassifier(model='gbm'),
    }
