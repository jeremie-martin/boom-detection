"""
Run baseline predictors and compare results.

Usage:
    uv run python -m boom_detection.run_baselines data
    uv run python -m boom_detection.run_baselines data --quick  # single seed
    uv run python -m boom_detection.run_baselines data -n 10    # limit samples

This validates the evaluation framework and establishes baseline performance.
Results are reported with proper uncertainty estimates (mean ± std).
"""

from __future__ import annotations

import time

import numpy as np

from .loader import load_dataset, Dataset
from .evaluation import compute_all_metrics
from .features import FeatureCache, FeatureConfig
from .frame_models import get_frame_level_predictors


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
# Fast Evaluation (no dataset loading needed)
# =============================================================================

def quick_cv(
    predictor_fn,
    cache: FeatureCache,
    data_path: str = 'data',
    k: int = 5,
    seeds: list[int] | None = None,
    task: str = 'frame',
    quality_threshold: float | None = None,
) -> dict:
    """
    Run robust multi-seed CV using only cached features.

    This is the fastest way to iterate: ~0.2s to load features vs ~30s for full dataset.
    Uses multiple seeds by default for proper uncertainty estimates.

    Args:
        predictor_fn: Factory function returning a fresh predictor
                     Example: lambda: FrameLevelClassifier(max_depth=7)
        cache: FeatureCache with disk caching enabled
        data_path: Path to data directory (for annotations.json)
        k: Number of CV folds
        seeds: Random seeds (default: [42, 43, 44] for quick, or [42] for --quick)
        task: 'frame' for boom frame prediction, 'quality' for boom quality prediction
        quality_threshold: If set, filter to simulations with quality >= threshold

    Returns:
        Dict with mean_metrics, std_metrics, and per-seed results
    """
    from sklearn.model_selection import KFold
    from .loader import load_annotations
    import os

    if seeds is None:
        seeds = [42, 43, 44]  # 3 seeds by default for quick iteration

    # Load just the annotations (tiny file, instant)
    if os.path.isdir(data_path):
        ann_path = os.path.join(data_path, 'annotations.json')
    else:
        ann_path = data_path
    annotations = load_annotations(ann_path)

    # Load features from disk cache
    sim_ids = [a.id for a in annotations]
    cache.load_from_disk(sim_ids, verbose=False)

    # Filter to only simulations we have cached
    available = [a for a in annotations if a.id in cache]
    if len(available) < len(annotations):
        print(f"Warning: Only {len(available)}/{len(annotations)} simulations cached")
    annotations = available

    # Filter by quality if requested
    if quality_threshold is not None:
        annotations = [a for a in annotations if a.boom_quality >= quality_threshold]
        print(f"Filtered to {len(annotations)} simulations with quality >= {quality_threshold}")

    ids = [a.id for a in annotations]

    # Select targets based on task
    if task == 'quality':
        targets = np.array([a.boom_quality for a in annotations])
    else:
        targets = np.array([a.boom_frame for a in annotations])

    # Run CV with multiple seeds
    all_seed_metrics = []

    for seed in seeds:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        all_preds = np.zeros(len(ids))

        for train_idx, test_idx in kf.split(ids):
            train_ids = [ids[i] for i in train_idx]
            test_ids = [ids[i] for i in test_idx]
            train_y = targets[train_idx]

            predictor = predictor_fn()  # Fresh predictor each fold
            predictor.fit(train_ids, train_y, cache)
            preds = predictor.predict(test_ids, cache)
            all_preds[test_idx] = preds

        metrics = compute_all_metrics(targets, all_preds, task=task)
        all_seed_metrics.append(metrics)

    # Aggregate across seeds
    metric_names = list(all_seed_metrics[0].keys())
    mean_metrics = {}
    std_metrics = {}

    for name in metric_names:
        values = [m[name] for m in all_seed_metrics]
        mean_metrics[name] = float(np.mean(values))
        std_metrics[name] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'seed_metrics': all_seed_metrics,
        'n_seeds': len(seeds),
    }


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
    """
    Run k-fold CV using cached features.

    Args:
        dataset: Dataset with annotations
        cache: FeatureCache with extracted features
        predictor: Predictor instance
        k: Number of CV folds
        seed: Random seed for reproducibility
        task: 'frame' or 'quality'
    """
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
        # Convert frame_level predictors to factories
        frame_predictors = get_frame_level_predictors()
        for name, predictor_class in frame_predictors.items():
            # frame_level predictors are already instances, wrap them
            baselines[name] = lambda p=predictor_class: p.__class__() if hasattr(p, '__class__') else p

    return baselines


def run_baselines(
    data_path: str,
    k: int = 5,
    seeds: list[int] | None = None,
    max_samples: int | None = None,
    max_pendulums: int | None = 2000,
):
    """Run all baselines and print comparison with proper uncertainty."""
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

    # Run baselines with multi-seed CV
    baselines = get_baselines()
    results = {}

    print(f"Running {k}-fold CV × {len(seeds)} seeds (robust evaluation)")
    print("=" * 60)

    for name, predictor_factory in baselines.items():
        t0 = time.time()
        try:
            # Run multi-seed CV
            all_metrics = []
            for seed in seeds:
                result = cross_validate_cached(dataset, cache, predictor_factory(), k=k, seed=seed)
                all_metrics.append(result['metrics'])

            # Aggregate
            mean_metrics = {}
            std_metrics = {}
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                mean_metrics[metric_name] = np.mean(values)
                std_metrics[metric_name] = np.std(values, ddof=1) if len(values) > 1 else 0.0

            results[name] = {
                'mean_metrics': mean_metrics,
                'std_metrics': std_metrics,
            }

            elapsed = time.time() - t0
            mae_mean = mean_metrics['mae']
            mae_std = std_metrics['mae']
            w10_mean = mean_metrics['within_10']
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
