"""
Run all baseline predictors and compare results.

Usage:
    uv run python -m boom_detection.run_baselines data
    uv run python -m boom_detection.run_baselines data -n 10  # quick test

This validates the evaluation framework and establishes baseline performance.
"""

from __future__ import annotations

import time

import numpy as np

from .loader import load_dataset, Dataset
from .evaluation import compute_all_metrics
from .features import FeatureCache, FeatureConfig


# =============================================================================
# Cache-aware baseline predictors
# =============================================================================

class MeanPredictor:
    """Predicts the mean of training targets."""

    def __init__(self, cache: FeatureCache | None = None):
        self.mean_value = 0.0

    def fit(self, ids: list[str], y: np.ndarray, cache: FeatureCache) -> None:
        self.mean_value = float(np.mean(y))

    def predict(self, ids: list[str], cache: FeatureCache) -> np.ndarray:
        return np.full(len(ids), self.mean_value)


class MedianPredictor:
    """Predicts the median of training targets."""

    def __init__(self):
        self.median_value = 0.0

    def fit(self, ids: list[str], y: np.ndarray, cache: FeatureCache) -> None:
        self.median_value = float(np.median(y))

    def predict(self, ids: list[str], cache: FeatureCache) -> np.ndarray:
        return np.full(len(ids), self.median_value)


class VarianceThresholdPredictor:
    """
    Predicts boom frame based on when tip variance crosses a threshold.

    Uses cached features for speed.
    """

    def __init__(self, threshold_percentile: float = 50.0):
        self.threshold_percentile = threshold_percentile
        self.threshold_fraction = 0.5

    def fit(self, ids: list[str], y: np.ndarray, cache: FeatureCache) -> None:
        fractions = []
        for sim_id, boom_frame in zip(ids, y):
            features = cache[sim_id]
            # var_x2 + var_y2 (indices 2,3 in variance features)
            tip_var = features[:, 2] + features[:, 3]
            max_var = np.max(tip_var)
            if max_var > 0:
                boom_frame = int(boom_frame)
                if boom_frame < len(tip_var):
                    fractions.append(tip_var[boom_frame] / max_var)

        if fractions:
            self.threshold_fraction = float(np.percentile(fractions, self.threshold_percentile))

    def predict(self, ids: list[str], cache: FeatureCache) -> np.ndarray:
        predictions = []
        for sim_id in ids:
            features = cache[sim_id]
            tip_var = features[:, 2] + features[:, 3]
            max_var = np.max(tip_var)
            threshold = self.threshold_fraction * max_var

            exceeds = np.where(tip_var >= threshold)[0]
            if len(exceeds) > 0:
                predictions.append(exceeds[0])
            else:
                predictions.append(len(tip_var) // 2)

        return np.array(predictions)


class DerivativeThresholdPredictor:
    """Predicts boom based on maximum rate of change of spread."""

    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self.peak_offset = 0

    def _get_spread_derivative(self, features: np.ndarray) -> np.ndarray:
        tip_var = features[:, 2] + features[:, 3]
        deriv = np.diff(tip_var)
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            deriv = np.convolve(deriv, kernel, mode='same')
        return deriv

    def fit(self, ids: list[str], y: np.ndarray, cache: FeatureCache) -> None:
        offsets = []
        for sim_id, boom_frame in zip(ids, y):
            deriv = self._get_spread_derivative(cache[sim_id])
            peak_frame = int(np.argmax(deriv))
            offsets.append(int(boom_frame) - peak_frame)
        self.peak_offset = int(np.median(offsets))

    def predict(self, ids: list[str], cache: FeatureCache) -> np.ndarray:
        predictions = []
        for sim_id in ids:
            features = cache[sim_id]
            deriv = self._get_spread_derivative(features)
            peak_frame = int(np.argmax(deriv))
            pred = max(0, min(peak_frame + self.peak_offset, len(features) - 1))
            predictions.append(pred)
        return np.array(predictions)


class SecondDerivativePredictor:
    """Predicts boom based on inflection point (peak of second derivative)."""

    def __init__(self, smoothing_window: int = 10):
        self.smoothing_window = smoothing_window
        self.peak_offset = 0

    def _get_second_derivative(self, features: np.ndarray) -> np.ndarray:
        tip_var = features[:, 2] + features[:, 3]
        d2 = np.diff(tip_var, n=2)
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            d2 = np.convolve(d2, kernel, mode='same')
        return d2

    def fit(self, ids: list[str], y: np.ndarray, cache: FeatureCache) -> None:
        offsets = []
        for sim_id, boom_frame in zip(ids, y):
            d2 = self._get_second_derivative(cache[sim_id])
            peak_frame = int(np.argmax(d2))
            offsets.append(int(boom_frame) - peak_frame)
        self.peak_offset = int(np.median(offsets))

    def predict(self, ids: list[str], cache: FeatureCache) -> np.ndarray:
        predictions = []
        for sim_id in ids:
            features = cache[sim_id]
            d2 = self._get_second_derivative(features)
            peak_frame = int(np.argmax(d2))
            pred = max(0, min(peak_frame + self.peak_offset, len(features) - 1))
            predictions.append(pred)
        return np.array(predictions)


class LinearRegressionPredictor:
    """Linear regression on summary statistics of features."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def _summarize(self, features: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.mean(features, axis=0),
            np.max(features, axis=0),
            np.std(features, axis=0),
        ])

    def fit(self, ids: list[str], y: np.ndarray, cache: FeatureCache) -> None:
        X = np.array([self._summarize(cache[sim_id]) for sim_id in ids])
        y = np.asarray(y)

        # Ridge regression
        lambda_reg = 1.0
        XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
        Xty = X.T @ y
        self.weights = np.linalg.solve(XtX, Xty)

    def predict(self, ids: list[str], cache: FeatureCache) -> np.ndarray:
        X = np.array([self._summarize(cache[sim_id]) for sim_id in ids])
        return X @ self.weights


# =============================================================================
# Evaluation with caching
# =============================================================================

def cross_validate_cached(
    dataset: Dataset,
    cache: FeatureCache,
    predictor,
    k: int = 5,
    seed: int = 42,
    task: str = "frame",
) -> dict:
    """Run k-fold CV using cached features."""
    from sklearn.model_selection import KFold

    ids = [a.id for a in dataset.annotations]
    if task == "frame":
        targets = np.array([a.boom_frame for a in dataset.annotations])
    else:
        targets = np.array([a.boom_quality for a in dataset.annotations])

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    all_preds = np.zeros(len(ids))
    all_gt = targets.copy()

    for train_idx, test_idx in kf.split(ids):
        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]
        train_y = targets[train_idx]

        predictor.fit(train_ids, train_y, cache)
        preds = predictor.predict(test_ids, cache)
        all_preds[test_idx] = preds

    return {
        'predictions': all_preds,
        'ground_truth': all_gt,
        'metrics': compute_all_metrics(all_gt, all_preds, task=task),
    }


def get_baselines() -> dict[str, object]:
    """Get all baseline predictors."""
    return {
        'mean': MeanPredictor(),
        'median': MedianPredictor(),
        'variance_threshold': VarianceThresholdPredictor(),
        'derivative_peak': DerivativeThresholdPredictor(),
        'second_derivative': SecondDerivativePredictor(),
        'linear_regression': LinearRegressionPredictor(),
    }


def run_baselines(
    data_path: str,
    k: int = 5,
    seed: int = 42,
    max_samples: int | None = None,
    max_pendulums: int | None = 2000,  # Default to 2000 for speed
):
    """Run all baselines and print comparison."""
    print(f"Loading dataset from: {data_path}")
    if max_samples:
        print(f"Limiting to {max_samples} samples")
    if max_pendulums:
        print(f"Subsampling to {max_pendulums} pendulums per simulation")
    print("=" * 60)

    t0 = time.time()
    dataset = load_dataset(data_path, verbose=True, max_samples=max_samples)
    print(f"Loaded {len(dataset)} simulations in {time.time() - t0:.1f}s")
    print()

    # Extract features once (with optional subsampling for speed)
    print("Extracting features (one-time cost)...")
    t0 = time.time()
    config = FeatureConfig(max_pendulums=max_pendulums) if max_pendulums else None
    cache = FeatureCache(config=config)
    cache.extract_all(dataset, verbose=True)
    print(f"Feature extraction: {time.time() - t0:.1f}s")
    print()

    # Run baselines
    baselines = get_baselines()
    results = {}

    print(f"Running {k}-fold cross-validation (seed={seed})")
    print("=" * 60)

    for name, predictor in baselines.items():
        t0 = time.time()
        try:
            result = cross_validate_cached(dataset, cache, predictor, k=k, seed=seed)
            results[name] = result
            m = result['metrics']
            elapsed = time.time() - t0
            print(f"{name:<25} MAE={m['mae']:>6.1f}  within_10={m['within_10']:>5.0%}  ({elapsed:.2f}s)")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print()
    print("=" * 60)
    print("SUMMARY - Boom Frame Prediction")
    print("=" * 60)
    print(f"{'Method':<25} {'MAE':>8} {'MedAE':>8} {'W/in 5':>8} {'W/in 10':>8} {'Corr':>8}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['mae'])
    for name, result in sorted_results:
        m = result['metrics']
        print(f"{name:<25} {m['mae']:>8.1f} {m['median_ae']:>8.1f} {m['within_5']:>7.0%} {m['within_10']:>8.0%} {m['correlation']:>8.2f}")

    if sorted_results:
        best_name, best_result = sorted_results[0]
        print("-" * 70)
        print(f"Best: {best_name} with MAE = {best_result['metrics']['mae']:.1f} frames")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline predictors")
    parser.add_argument("data_path", nargs="?", default="data", help="Path to data directory")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Limit samples (for quick testing)")
    parser.add_argument("-p", "--max-pendulums", type=int, default=2000, help="Subsample pendulums (default: 2000, 0=all)")
    parser.add_argument("-k", "--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    max_pendulums = args.max_pendulums if args.max_pendulums > 0 else None
    run_baselines(args.data_path, k=args.folds, seed=args.seed, max_samples=args.max_samples, max_pendulums=max_pendulums)
