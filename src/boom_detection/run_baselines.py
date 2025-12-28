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
from .frame_models import get_frame_level_predictors, FrameLevelClassifier
from .changepoint import CUSUMDetector, BOCPDDetector, MultiFeatureCUSUM
from .ensemble import AdaptiveEnsemble, create_default_ensemble


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
    predictor,
    cache: FeatureCache,
    data_path: str = 'data',
    k: int = 5,
    seed: int = 42,
    task: str = 'frame',
    quality_threshold: float | None = None,
) -> dict:
    """
    Run CV using only cached features - no dataset loading required.

    This is the fastest way to iterate: ~0.2s to load features vs ~30s for full dataset.

    Args:
        predictor: Model with fit() and predict() methods
        cache: FeatureCache with disk caching enabled
        data_path: Path to data directory (for annotations.json)
        k: Number of CV folds
        seed: Random seed
        task: 'frame' for boom frame prediction, 'quality' for boom quality prediction
        quality_threshold: If set, filter to simulations with quality >= threshold

    Returns:
        Dict with predictions, ground_truth, and metrics
    """
    from sklearn.model_selection import KFold
    from .loader import load_annotations
    import os

    # Load just the annotations (tiny file, instant)
    if os.path.isdir(data_path):
        ann_path = os.path.join(data_path, 'annotations.json')
    else:
        ann_path = data_path
    annotations = load_annotations(ann_path)

    # Load features from disk cache
    sim_ids = [a.id for a in annotations]
    cache.load_from_disk(sim_ids, verbose=True)

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

    # Run CV
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    all_preds = np.zeros(len(ids))

    for train_idx, test_idx in kf.split(ids):
        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]
        train_y = targets[train_idx]

        predictor.fit(train_ids, train_y, cache)
        preds = predictor.predict(test_ids, cache)
        all_preds[test_idx] = preds

    return {
        'predictions': all_preds,
        'ground_truth': targets,
        'metrics': compute_all_metrics(targets, all_preds, task=task),
    }


# =============================================================================
# Evaluation with caching
# =============================================================================

def _run_fold(fold_data: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single CV fold. Used by parallel CV."""
    (train_idx, test_idx, ids, targets, features_dict, predictor_factory) = fold_data

    # Reconstruct a simple cache-like object for the predictor
    class SimpleCache:
        def __init__(self, data: dict):
            self._data = data
        def __getitem__(self, key: str) -> np.ndarray:
            return self._data[key]

    cache = SimpleCache(features_dict)
    predictor = predictor_factory()

    train_ids = [ids[i] for i in train_idx]
    test_ids = [ids[i] for i in test_idx]
    train_y = targets[train_idx]

    predictor.fit(train_ids, train_y, cache)
    preds = predictor.predict(test_ids, cache)

    return test_idx, preds, targets[test_idx]


def cross_validate_cached(
    dataset: Dataset,
    cache: FeatureCache,
    predictor,
    k: int = 5,
    seed: int = 42,
    task: str = "frame",
    n_jobs: int = 1,
) -> dict:
    """
    Run k-fold CV using cached features.

    Args:
        dataset: Dataset with annotations
        cache: FeatureCache with extracted features
        predictor: Predictor instance (will be cloned for parallel execution)
        k: Number of CV folds
        seed: Random seed for reproducibility
        task: 'frame' or 'quality'
        n_jobs: Number of parallel jobs. 1=sequential, -1=all cores.
                Note: If predictor uses all cores internally (e.g., HistGBM),
                setting n_jobs>1 may cause resource contention.
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

    if n_jobs == 1:
        # Sequential execution (default, most efficient for HistGBM)
        for train_idx, test_idx in kf.split(ids):
            train_ids = [ids[i] for i in train_idx]
            test_ids = [ids[i] for i in test_idx]
            train_y = targets[train_idx]

            predictor.fit(train_ids, train_y, cache)
            preds = predictor.predict(test_ids, cache)
            all_preds[test_idx] = preds
    else:
        # Parallel execution using joblib
        from joblib import Parallel, delayed
        import copy

        # Extract features to dict (small, serializable)
        features_dict = {sim_id: cache[sim_id] for sim_id in ids}

        # Create a factory function that returns fresh predictor instances
        def predictor_factory():
            return copy.deepcopy(predictor)

        # Prepare fold data
        fold_data = [
            (train_idx, test_idx, ids, targets, features_dict, predictor_factory)
            for train_idx, test_idx in kf.split(ids)
        ]

        # Run folds in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_fold)(data) for data in fold_data
        )

        # Collect results
        for test_idx, preds, _ in results:
            all_preds[test_idx] = preds

    return {
        'predictions': all_preds,
        'ground_truth': all_gt,
        'metrics': compute_all_metrics(all_gt, all_preds, task=task),
    }


def get_baselines(include_frame_level: bool = True) -> dict[str, object]:
    """Get all baseline predictors."""
    baselines = {
        'mean': MeanPredictor(),
        'median': MedianPredictor(),
        'variance_threshold': VarianceThresholdPredictor(),
        'derivative_peak': DerivativeThresholdPredictor(),
        'second_derivative': SecondDerivativePredictor(),
        'linear_regression': LinearRegressionPredictor(),
    }

    if include_frame_level:
        baselines.update(get_frame_level_predictors())

    return baselines


def get_changepoint_detectors() -> dict[str, object]:
    """Get change point detection predictors."""
    return {
        'cusum': CUSUMDetector(feature_idx=2, use_derivative=True),  # var_x2
        'cusum_no_deriv': CUSUMDetector(feature_idx=2, use_derivative=False),
        'bocpd': BOCPDDetector(feature_idx=2, hazard_rate=1/200),
        'multi_cusum': MultiFeatureCUSUM(n_features=5, combination='median'),
    }


def get_ensemble_predictors(n_features: int) -> dict[str, object]:
    """Get ensemble predictors."""
    # Simple ensemble with best models
    histgbm = FrameLevelClassifier(model='hist_gbm', n_estimators=200, max_depth=7)
    cusum = MultiFeatureCUSUM(n_features=5, combination='median')

    simple_ensemble = AdaptiveEnsemble(weight_method='inverse_mae')
    simple_ensemble.add_model('histgbm', histgbm)
    simple_ensemble.add_model('cusum', cusum)

    return {
        'ensemble_histgbm_cusum': simple_ensemble,
    }


def run_baselines(
    data_path: str,
    k: int = 5,
    seed: int = 42,
    max_samples: int | None = None,
    max_pendulums: int | None = 2000,  # Default to 2000 for speed
    include_changepoint: bool = False,
    include_ensemble: bool = False,
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
    n_features = cache[dataset.annotations[0].id].shape[1]
    print(f"Feature extraction: {time.time() - t0:.1f}s ({n_features} features)")
    print()

    # Run baselines
    baselines = get_baselines()

    # Add changepoint detectors if requested
    if include_changepoint:
        baselines.update(get_changepoint_detectors())

    # Add ensemble predictors if requested
    if include_ensemble:
        baselines.update(get_ensemble_predictors(n_features))

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
    parser.add_argument("--changepoint", action="store_true", help="Include change point detectors")
    parser.add_argument("--ensemble", action="store_true", help="Include ensemble predictors")
    parser.add_argument("--phase4", action="store_true", help="Run Phase 4 models (changepoint + ensemble)")
    args = parser.parse_args()

    max_pendulums = args.max_pendulums if args.max_pendulums > 0 else None
    include_changepoint = args.changepoint or args.phase4
    include_ensemble = args.ensemble or args.phase4
    run_baselines(
        args.data_path,
        k=args.folds,
        seed=args.seed,
        max_samples=args.max_samples,
        max_pendulums=max_pendulums,
        include_changepoint=include_changepoint,
        include_ensemble=include_ensemble,
    )
