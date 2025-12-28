"""
Run baseline predictors and compare results.

Usage:
    uv run python -m boom_detection.run_baselines data
    uv run python -m boom_detection.run_baselines data --quick  # single seed
    uv run python -m boom_detection.run_baselines data -n 10    # limit samples

This validates the evaluation framework and establishes baseline performance.
Results are reported with proper uncertainty estimates (mean ± std).

Uses the unified CachedEvaluator for consistent evaluation across the codebase.
"""

from __future__ import annotations

import time

import numpy as np

from .loader import load_dataset
from .evaluation import CachedEvaluator
from .features import FeatureCache, FeatureConfig
from .frame_models import get_frame_level_predictors
from .logging_config import logger


# =============================================================================
# Cache-aware baseline predictors
# =============================================================================

class MeanPredictor:
    """Predicts the mean of training targets."""

    def __init__(self):
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
# Baseline Registry
# =============================================================================

def get_baselines(include_frame_level: bool = True) -> dict[str, callable]:
    """
    Get all baseline predictor factories.

    Returns factory functions (not instances) so each seed gets a fresh model.
    """
    baselines = {
        'mean': lambda: MeanPredictor(),
        'median': lambda: MedianPredictor(),
        'variance_threshold': lambda: VarianceThresholdPredictor(),
        'derivative_peak': lambda: DerivativeThresholdPredictor(),
        'second_derivative': lambda: SecondDerivativePredictor(),
        'linear_regression': lambda: LinearRegressionPredictor(),
    }

    if include_frame_level:
        # get_frame_level_predictors() now returns factories directly
        frame_predictors = get_frame_level_predictors()
        baselines.update(frame_predictors)

    return baselines


# =============================================================================
# Main Evaluation
# =============================================================================

def run_baselines(
    data_path: str,
    k: int = 5,
    seeds: list[int] | None = None,
    max_samples: int | None = None,
    max_pendulums: int | None = 2000,
):
    """
    Run all baselines using the unified CachedEvaluator framework.

    Args:
        data_path: Path to data directory
        k: Number of CV folds
        seeds: Random seeds for robust evaluation
        max_samples: Limit number of samples (for quick testing)
        max_pendulums: Subsample pendulums per simulation
    """
    if seeds is None:
        seeds = [42, 43, 44]  # 3 seeds for reasonable speed/robustness tradeoff

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

    # Extract features once
    print("Extracting features (one-time cost)...")
    t0 = time.time()
    config = FeatureConfig(max_pendulums=max_pendulums) if max_pendulums else FeatureConfig()
    cache = FeatureCache(config=config, cache_dir='.feature_cache')
    cache.extract_all(dataset, verbose=True)
    n_features = cache[dataset.annotations[0].id].shape[1]
    print(f"Feature extraction: {time.time() - t0:.1f}s ({n_features} features)")
    print()

    # Create unified evaluator
    evaluator = CachedEvaluator(dataset, cache)

    # Run baselines with multi-seed CV
    baselines = get_baselines()
    results = {}

    print(f"Running {k}-fold CV × {len(seeds)} seeds (robust evaluation)")
    print("=" * 60)

    for name, predictor_factory in baselines.items():
        logger.info("Evaluating baseline: {}", name)
        t0 = time.time()
        try:
            # Use unified CachedEvaluator
            result = evaluator.cross_validate(
                predictor_factory,
                k=k,
                seeds=seeds,
                verbose=False,
            )

            results[name] = {
                'mean_metrics': result.mean_metrics,
                'std_metrics': result.std_metrics,
            }

            elapsed = time.time() - t0
            mae_mean = result.mean_metrics['mae']
            mae_std = result.std_metrics['mae']
            w10_mean = result.mean_metrics['within_10']
            print(f"{name:<25} MAE={mae_mean:>5.1f}±{mae_std:>4.1f}  within_10={w10_mean:>5.0%}  ({elapsed:.1f}s)")

        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print()
    print("=" * 60)
    print(f"SUMMARY - Boom Frame Prediction ({len(seeds)} seeds)")
    print("=" * 60)
    print(f"{'Method':<25} {'MAE':>12} {'MedAE':>12} {'W/in 5':>12} {'W/in 10':>12}")
    print("-" * 75)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_metrics']['mae'])
    for name, result in sorted_results:
        m = result['mean_metrics']
        s = result['std_metrics']
        print(f"{name:<25} {m['mae']:>5.1f}±{s['mae']:<5.1f} {m['median_ae']:>5.1f}±{s['median_ae']:<5.1f} "
              f"{m['within_5']*100:>4.0f}±{s['within_5']*100:<4.0f}% {m['within_10']*100:>4.0f}±{s['within_10']*100:<4.0f}%")

    if sorted_results:
        best_name, best_result = sorted_results[0]
        m = best_result['mean_metrics']
        s = best_result['std_metrics']
        print("-" * 75)
        print(f"Best: {best_name} with MAE = {m['mae']:.1f} ± {s['mae']:.1f} frames")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline predictors with robust multi-seed evaluation")
    parser.add_argument("data_path", nargs="?", default="data", help="Path to data directory")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Limit samples (for quick testing)")
    parser.add_argument("-p", "--max-pendulums", type=int, default=2000, help="Subsample pendulums (default: 2000, 0=all)")
    parser.add_argument("-k", "--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--quick", action="store_true", help="Quick single-seed evaluation")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    args = parser.parse_args()

    max_pendulums = args.max_pendulums if args.max_pendulums > 0 else None

    # Determine seeds
    if args.quick:
        seeds = [42]
    else:
        seeds = list(range(42, 42 + args.seeds))

    run_baselines(
        args.data_path,
        k=args.folds,
        seeds=seeds,
        max_samples=args.max_samples,
        max_pendulums=max_pendulums,
    )
