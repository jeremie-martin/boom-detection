"""
Evaluation framework for boom detection experiments.

Provides k-fold cross-validation, multiple metrics, and utilities for
systematic experimentation.

Usage:
    from boom_detection.evaluation import Evaluator, cross_validate
    from boom_detection import load_dataset

    dataset = load_dataset('data')
    evaluator = Evaluator(dataset)

    # Evaluate any predictor with sklearn-style fit/predict
    results = evaluator.cross_validate(my_model, k=5)
    print(results.summary())

    # Or use a simple function
    results = evaluator.cross_validate_fn(
        lambda train, test: [mean(train_targets) for _ in test],
        k=5
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from sklearn.model_selection import KFold

from .loader import Dataset, Simulation


# =============================================================================
# Metrics
# =============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error - robust to outliers."""
    return float(np.median(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error - penalizes large errors."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def max_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum Absolute Error - worst case."""
    return float(np.max(np.abs(y_true - y_pred)))


def within_n_frames(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    """Fraction of predictions within n frames of ground truth."""
    return float(np.mean(np.abs(y_true - y_pred) <= n))


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation - does it rank correctly?"""
    if np.std(y_pred) < 1e-9:  # constant predictions
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "frame"
) -> dict[str, float]:
    """
    Compute all relevant metrics for a prediction task.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        task: "frame" for boom frame, "quality" for boom quality

    Returns:
        Dictionary of metric_name -> value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        'mae': mae(y_true, y_pred),
        'median_ae': median_ae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'max_ae': max_ae(y_true, y_pred),
        'correlation': correlation(y_true, y_pred),
    }

    if task == "frame":
        # Frame-specific: fraction within N frames
        for n in [5, 10, 15, 30]:
            metrics[f'within_{n}'] = within_n_frames(y_true, y_pred, n)
    elif task == "quality":
        # Quality-specific: within certain thresholds on 0-1 scale
        for thresh in [0.05, 0.10, 0.15, 0.20]:
            metrics[f'within_{thresh:.2f}'] = within_n_frames(y_true, y_pred, thresh)

    return metrics


# =============================================================================
# Predictor Protocol
# =============================================================================

class Predictor(Protocol):
    """Protocol for predictors with sklearn-style fit/predict interface."""

    def fit(self, X: Sequence[Simulation], y: np.ndarray) -> None:
        """Fit the predictor on training data."""
        ...

    def predict(self, X: Sequence[Simulation]) -> np.ndarray:
        """Predict on new data."""
        ...


# Type for functional predictors
PredictorFn = Callable[
    [list[tuple[Simulation, int, float]], list[Simulation]],  # (train_data, test_sims)
    np.ndarray  # predictions
]


# =============================================================================
# Results Container
# =============================================================================

@dataclass
class FoldResult:
    """Results from a single fold."""
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    predictions: np.ndarray
    ground_truth: np.ndarray
    metrics: dict[str, float]


@dataclass
class EvaluationResult:
    """Complete results from cross-validation."""
    task: str  # "frame" or "quality"
    k: int
    seed: int
    fold_results: list[FoldResult]

    # Aggregated predictions (all test predictions across folds)
    all_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    all_ground_truth: np.ndarray = field(default_factory=lambda: np.array([]))
    all_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    # Aggregated metrics
    aggregate_metrics: dict[str, float] = field(default_factory=dict)

    # Per-fold metric statistics
    metric_means: dict[str, float] = field(default_factory=dict)
    metric_stds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.fold_results) > 0 and len(self.aggregate_metrics) == 0:
            self._compute_aggregates()

    def _compute_aggregates(self):
        """Compute aggregate metrics from fold results."""
        # Collect all predictions in original order
        n_samples = sum(len(fr.test_indices) for fr in self.fold_results)
        self.all_predictions = np.zeros(n_samples)
        self.all_ground_truth = np.zeros(n_samples)
        self.all_indices = np.zeros(n_samples, dtype=int)

        for fr in self.fold_results:
            self.all_predictions[fr.test_indices] = fr.predictions
            self.all_ground_truth[fr.test_indices] = fr.ground_truth
            self.all_indices[fr.test_indices] = fr.test_indices

        # Compute aggregate metrics on all predictions
        self.aggregate_metrics = compute_all_metrics(
            self.all_ground_truth,
            self.all_predictions,
            task=self.task
        )

        # Compute mean and std of per-fold metrics
        metric_names = list(self.fold_results[0].metrics.keys())
        for name in metric_names:
            values = [fr.metrics[name] for fr in self.fold_results]
            self.metric_means[name] = float(np.mean(values))
            self.metric_stds[name] = float(np.std(values))

    def summary(self, primary_metric: str = 'mae') -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Cross-Validation Results ({self.k}-fold, seed={self.seed})",
            f"Task: {self.task}",
            "=" * 50,
            "",
            f"Primary metric ({primary_metric}): {self.aggregate_metrics[primary_metric]:.3f}",
            "",
            "All metrics (aggregate over all folds):",
        ]

        for name, value in sorted(self.aggregate_metrics.items()):
            if name.startswith('within'):
                lines.append(f"  {name}: {value:.1%}")
            else:
                lines.append(f"  {name}: {value:.3f}")

        lines.extend([
            "",
            "Per-fold stability (mean ± std):",
        ])

        for name in sorted(self.metric_means.keys()):
            mean = self.metric_means[name]
            std = self.metric_stds[name]
            if name.startswith('within'):
                lines.append(f"  {name}: {mean:.1%} ± {std:.1%}")
            else:
                lines.append(f"  {name}: {mean:.3f} ± {std:.3f}")

        return "\n".join(lines)

    def per_sample_errors(self) -> list[tuple[int, float, float, float]]:
        """
        Get per-sample errors for analysis.

        Returns:
            List of (index, ground_truth, prediction, error) tuples,
            sorted by error descending.
        """
        errors = np.abs(self.all_ground_truth - self.all_predictions)
        order = np.argsort(-errors)  # descending

        return [
            (int(self.all_indices[i]),
             float(self.all_ground_truth[i]),
             float(self.all_predictions[i]),
             float(errors[i]))
            for i in order
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'task': self.task,
            'k': self.k,
            'seed': self.seed,
            'aggregate_metrics': self.aggregate_metrics,
            'metric_means': self.metric_means,
            'metric_stds': self.metric_stds,
            'predictions': self.all_predictions.tolist(),
            'ground_truth': self.all_ground_truth.tolist(),
            'indices': self.all_indices.tolist(),
        }

    def save(self, path: str | Path):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Evaluator Class
# =============================================================================

class Evaluator:
    """
    Main evaluation harness for boom detection experiments.

    Usage:
        evaluator = Evaluator(dataset)
        results = evaluator.cross_validate(model, k=5)
    """

    def __init__(self, dataset: Dataset):
        """
        Initialize evaluator with a dataset.

        Args:
            dataset: Loaded Dataset object with simulations
        """
        self.dataset = dataset
        self.annotations = dataset.annotations
        self.n_samples = len(dataset)

        # Extract targets
        self.boom_frames = np.array([a.boom_frame for a in self.annotations])
        self.boom_qualities = np.array([a.boom_quality for a in self.annotations])

    def get_simulation(self, idx: int) -> Simulation:
        """Get simulation by index."""
        ann = self.annotations[idx]
        return self.dataset.simulations[ann.id]

    def cross_validate(
        self,
        predictor: Predictor,
        k: int = 5,
        seed: int = 42,
        task: str = "frame",
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Run k-fold cross-validation with a sklearn-style predictor.

        Args:
            predictor: Object with fit(X, y) and predict(X) methods
            k: Number of folds
            seed: Random seed for reproducibility
            task: "frame" for boom frame, "quality" for boom quality
            verbose: Print progress

        Returns:
            EvaluationResult with all metrics and predictions
        """
        targets = self.boom_frames if task == "frame" else self.boom_qualities
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.annotations)):
            if verbose:
                print(f"Fold {fold_idx + 1}/{k}...")

            # Get train/test simulations
            train_sims = [self.get_simulation(i) for i in train_idx]
            test_sims = [self.get_simulation(i) for i in test_idx]
            train_y = targets[train_idx]
            test_y = targets[test_idx]

            # Fit and predict
            predictor.fit(train_sims, train_y)
            predictions = predictor.predict(test_sims)
            predictions = np.asarray(predictions)

            # Compute metrics
            metrics = compute_all_metrics(test_y, predictions, task=task)

            fold_results.append(FoldResult(
                fold=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                predictions=predictions,
                ground_truth=test_y,
                metrics=metrics,
            ))

            if verbose:
                print(f"  MAE: {metrics['mae']:.2f}")

        result = EvaluationResult(
            task=task,
            k=k,
            seed=seed,
            fold_results=fold_results,
        )

        if verbose:
            print(f"\nAggregate MAE: {result.aggregate_metrics['mae']:.2f}")

        return result

    def cross_validate_fn(
        self,
        predict_fn: PredictorFn,
        k: int = 5,
        seed: int = 42,
        task: str = "frame",
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Run k-fold cross-validation with a function-based predictor.

        This is useful for simple methods that don't need a class.

        Args:
            predict_fn: Function(train_data, test_sims) -> predictions
                where train_data is list of (Simulation, boom_frame, boom_quality)
                and test_sims is list of Simulation
            k: Number of folds
            seed: Random seed
            task: "frame" or "quality"
            verbose: Print progress

        Returns:
            EvaluationResult
        """
        targets = self.boom_frames if task == "frame" else self.boom_qualities
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.annotations)):
            if verbose:
                print(f"Fold {fold_idx + 1}/{k}...")

            # Build train data as (sim, frame, quality) tuples
            train_data = [
                (self.get_simulation(i),
                 int(self.boom_frames[i]),
                 float(self.boom_qualities[i]))
                for i in train_idx
            ]
            test_sims = [self.get_simulation(i) for i in test_idx]
            test_y = targets[test_idx]

            # Get predictions
            predictions = predict_fn(train_data, test_sims)
            predictions = np.asarray(predictions)

            # Compute metrics
            metrics = compute_all_metrics(test_y, predictions, task=task)

            fold_results.append(FoldResult(
                fold=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                predictions=predictions,
                ground_truth=test_y,
                metrics=metrics,
            ))

            if verbose:
                print(f"  MAE: {metrics['mae']:.2f}")

        result = EvaluationResult(
            task=task,
            k=k,
            seed=seed,
            fold_results=fold_results,
        )

        if verbose:
            print(f"\nAggregate MAE: {result.aggregate_metrics['mae']:.2f}")

        return result

    def evaluate_single(
        self,
        predictions: np.ndarray,
        task: str = "frame",
    ) -> dict[str, float]:
        """
        Evaluate predictions directly (no cross-validation).

        Useful when you have pre-computed predictions for all samples.

        Args:
            predictions: Array of predictions (same length as dataset)
            task: "frame" or "quality"

        Returns:
            Dictionary of metrics
        """
        targets = self.boom_frames if task == "frame" else self.boom_qualities
        return compute_all_metrics(targets, predictions, task=task)


# =============================================================================
# Convenience Functions
# =============================================================================

def cross_validate(
    dataset: Dataset,
    predictor: Predictor,
    k: int = 5,
    seed: int = 42,
    task: str = "frame",
    verbose: bool = True,
) -> EvaluationResult:
    """
    Convenience function for cross-validation.

    Equivalent to Evaluator(dataset).cross_validate(...)
    """
    return Evaluator(dataset).cross_validate(
        predictor, k=k, seed=seed, task=task, verbose=verbose
    )


# =============================================================================
# Robust Multi-Seed Evaluation (NEW)
# =============================================================================

@dataclass
class MultiSeedResult:
    """
    Results from multi-seed cross-validation.

    Provides proper uncertainty estimates (mean ± std with confidence intervals).
    This is the recommended result format for all experiments.
    """
    task: str
    k: int
    seeds: list[int]
    seed_results: list[EvaluationResult]

    # Aggregated statistics (computed in __post_init__)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    ci_lower: dict[str, float] = field(default_factory=dict)
    ci_upper: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.seed_results) > 0 and len(self.mean_metrics) == 0:
            self._compute_statistics()

    def _compute_statistics(self):
        """Compute mean, std, and 95% CI across seeds."""
        from scipy import stats

        metric_names = list(self.seed_results[0].aggregate_metrics.keys())

        for name in metric_names:
            values = [r.aggregate_metrics[name] for r in self.seed_results]
            n = len(values)

            self.mean_metrics[name] = float(np.mean(values))
            self.std_metrics[name] = float(np.std(values, ddof=1)) if n > 1 else 0.0

            # 95% confidence interval (t-distribution for small n)
            if n > 1 and self.std_metrics[name] > 0:
                se = self.std_metrics[name] / np.sqrt(n)
                t_val = stats.t.ppf(0.975, n - 1)
                self.ci_lower[name] = self.mean_metrics[name] - t_val * se
                self.ci_upper[name] = self.mean_metrics[name] + t_val * se
            else:
                self.ci_lower[name] = self.mean_metrics[name]
                self.ci_upper[name] = self.mean_metrics[name]

    def summary(self, primary_metric: str = 'mae') -> str:
        """Generate summary with confidence intervals."""
        lines = [
            f"Robust Cross-Validation Results",
            f"  {self.k}-fold CV × {len(self.seeds)} seeds = {self.k * len(self.seeds)} evaluations",
            f"  Seeds: {self.seeds}",
            "=" * 60,
            "",
        ]

        # Primary metric with full details
        pm = primary_metric
        lines.append(f"Primary metric ({pm}):")
        lines.append(f"  {self.mean_metrics[pm]:.2f} ± {self.std_metrics[pm]:.2f}")
        lines.append(f"  95% CI: [{self.ci_lower[pm]:.2f}, {self.ci_upper[pm]:.2f}]")
        lines.append("")

        # All metrics
        lines.append("All metrics (mean ± std):")
        for name in sorted(self.mean_metrics.keys()):
            mean = self.mean_metrics[name]
            std = self.std_metrics[name]

            if name.startswith('within'):
                lines.append(f"  {name}: {mean:.1%} ± {std:.1%}")
            else:
                lines.append(f"  {name}: {mean:.2f} ± {std:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'task': self.task,
            'k': self.k,
            'seeds': self.seeds,
            'mean_metrics': self.mean_metrics,
            'std_metrics': self.std_metrics,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
        }


# =============================================================================
# Cache-Aware Evaluator (NEW)
# =============================================================================

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .features import FeatureCache


class CachedPredictor(Protocol):
    """
    Protocol for predictors that work with FeatureCache.

    This is the ACTUAL interface used by all models in this codebase:
    - FrameLevelClassifier, FrameLevelRegressor
    - SequenceTrainer (CNN, LSTM, Transformer)
    - BoomDetectionPipeline
    - All models in run_baselines.py
    """

    def fit(
        self,
        sim_ids: list[str],
        targets: np.ndarray,
        cache: 'FeatureCache'
    ) -> None:
        """Fit the model on training data."""
        ...

    def predict(
        self,
        sim_ids: list[str],
        cache: 'FeatureCache'
    ) -> np.ndarray:
        """Predict on new simulations."""
        ...


class CachedEvaluator:
    """
    Evaluator that works with FeatureCache and the actual model interface.

    This is the recommended evaluator for all experiments. It:
    - Uses the (sim_ids, targets, cache) interface that all models implement
    - Runs multiple seeds by default for robust estimates
    - Returns proper uncertainty bounds

    Usage:
        from boom_detection.features import FeatureCache, FeatureConfig
        from boom_detection.evaluation import CachedEvaluator

        cache = FeatureCache(FeatureConfig(), cache_dir='.feature_cache')
        cache.extract_all(dataset)

        evaluator = CachedEvaluator(dataset, cache)

        # Quick single-seed evaluation (for development)
        result = evaluator.cross_validate(my_model, seeds=[42])

        # Robust multi-seed evaluation (for reporting)
        result = evaluator.cross_validate(my_model)  # Uses 5 seeds by default
        print(result.summary())
    """

    def __init__(self, dataset: Dataset, cache: 'FeatureCache'):
        """
        Args:
            dataset: Dataset with annotations
            cache: FeatureCache with extracted features
        """
        self.dataset = dataset
        self.cache = cache
        self.annotations = dataset.annotations

        # Extract data
        self.sim_ids = [a.id for a in self.annotations]
        self.boom_frames = np.array([a.boom_frame for a in self.annotations])
        self.boom_qualities = np.array([a.boom_quality for a in self.annotations])

    def cross_validate(
        self,
        predictor_fn: Callable[[], CachedPredictor],
        k: int = 5,
        seeds: list[int] | None = None,
        task: str = "frame",
        verbose: bool = True,
    ) -> MultiSeedResult:
        """
        Run robust multi-seed cross-validation.

        IMPORTANT: predictor_fn must be a factory function that returns a NEW
        predictor instance each time. This ensures each seed gets a fresh model.

        Args:
            predictor_fn: Factory function returning a fresh predictor
                         Example: lambda: FrameLevelClassifier(max_depth=7)
            k: Number of folds (default: 5)
            seeds: List of random seeds (default: [42, 43, 44, 45, 46])
            task: "frame" or "quality"
            verbose: Print progress

        Returns:
            MultiSeedResult with uncertainty estimates

        Example:
            # For a class-based model:
            result = evaluator.cross_validate(
                lambda: FrameLevelClassifier(max_depth=7),
                seeds=[42, 43, 44]
            )

            # For quick development (single seed):
            result = evaluator.cross_validate(
                lambda: MyModel(),
                seeds=[42]
            )
        """
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]  # 5 seeds by default

        targets = self.boom_frames if task == "frame" else self.boom_qualities
        seed_results = []

        for seed_idx, seed in enumerate(seeds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Seed {seed} ({seed_idx + 1}/{len(seeds)})")
                print('='*50)

            # Run single-seed CV
            result = self._cross_validate_single_seed(
                predictor_fn, k=k, seed=seed, targets=targets, task=task, verbose=verbose
            )
            seed_results.append(result)

        multi_result = MultiSeedResult(
            task=task,
            k=k,
            seeds=seeds,
            seed_results=seed_results,
        )

        if verbose:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print('='*60)
            print(f"MAE: {multi_result.mean_metrics['mae']:.2f} ± {multi_result.std_metrics['mae']:.2f}")
            print(f"95% CI: [{multi_result.ci_lower['mae']:.2f}, {multi_result.ci_upper['mae']:.2f}]")

        return multi_result

    def _cross_validate_single_seed(
        self,
        predictor_fn: Callable[[], CachedPredictor],
        k: int,
        seed: int,
        targets: np.ndarray,
        task: str,
        verbose: bool,
    ) -> EvaluationResult:
        """Run CV with a single seed."""
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sim_ids)):
            # Get train/test data
            train_ids = [self.sim_ids[i] for i in train_idx]
            test_ids = [self.sim_ids[i] for i in test_idx]
            train_y = targets[train_idx]
            test_y = targets[test_idx]

            # Create fresh predictor and train
            predictor = predictor_fn()
            predictor.fit(train_ids, train_y, self.cache)

            # Predict
            predictions = predictor.predict(test_ids, self.cache)
            predictions = np.asarray(predictions)

            # Compute metrics
            metrics = compute_all_metrics(test_y, predictions, task=task)

            fold_results.append(FoldResult(
                fold=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                predictions=predictions,
                ground_truth=test_y,
                metrics=metrics,
            ))

            if verbose:
                print(f"  Fold {fold_idx + 1}/{k}: MAE={metrics['mae']:.2f}")

        return EvaluationResult(
            task=task,
            k=k,
            seed=seed,
            fold_results=fold_results,
        )

    def quick_evaluate(
        self,
        predictor_fn: Callable[[], CachedPredictor],
        k: int = 5,
        seed: int = 42,
        task: str = "frame",
        verbose: bool = False,
    ) -> dict[str, float]:
        """
        Quick single-seed evaluation for development iteration.

        Returns just the metrics dict, not the full result object.
        Use this during development when you need fast feedback.

        Args:
            predictor_fn: Factory function returning predictor
            k: Number of folds
            seed: Random seed
            task: "frame" or "quality"
            verbose: Print progress

        Returns:
            Dict of metric_name -> value
        """
        targets = self.boom_frames if task == "frame" else self.boom_qualities
        result = self._cross_validate_single_seed(
            predictor_fn, k=k, seed=seed, targets=targets, task=task, verbose=verbose
        )
        return result.aggregate_metrics


def robust_evaluate(
    cache: 'FeatureCache',
    predictor_fn: Callable[[], CachedPredictor],
    data_path: str = 'data',
    k: int = 5,
    seeds: list[int] | None = None,
    task: str = "frame",
    verbose: bool = True,
) -> MultiSeedResult:
    """
    One-line robust evaluation for any model.

    This is the recommended way to evaluate models. It:
    - Loads annotations automatically
    - Runs multi-seed CV
    - Returns proper uncertainty estimates

    Args:
        cache: FeatureCache with extracted features
        predictor_fn: Factory function returning a predictor
        data_path: Path to data directory
        k: Number of folds
        seeds: Random seeds (default: 5 seeds)
        task: "frame" or "quality"
        verbose: Print progress

    Returns:
        MultiSeedResult with mean ± std ± CI

    Example:
        from boom_detection.features import FeatureCache, FeatureConfig
        from boom_detection.evaluation import robust_evaluate
        from boom_detection.frame_models import FrameLevelClassifier

        cache = FeatureCache(FeatureConfig(), cache_dir='.feature_cache')
        cache.load_from_disk()  # Load pre-extracted features

        result = robust_evaluate(
            cache,
            lambda: FrameLevelClassifier(max_depth=7),
        )
        print(result.summary())
    """
    from .loader import load_annotations
    import os

    # Load annotations
    if os.path.isdir(data_path):
        ann_path = os.path.join(data_path, 'annotations.json')
    else:
        ann_path = data_path
    annotations = load_annotations(ann_path)

    # Filter to available simulations
    available_ids = set()
    for a in annotations:
        if a.id in cache:
            available_ids.add(a.id)

    annotations = [a for a in annotations if a.id in available_ids]

    if verbose:
        print(f"Evaluating on {len(annotations)} simulations")

    # Create minimal dataset structure
    class MinimalDataset:
        def __init__(self, anns):
            self.annotations = anns

    dataset = MinimalDataset(annotations)
    evaluator = CachedEvaluator(dataset, cache)

    return evaluator.cross_validate(
        predictor_fn, k=k, seeds=seeds, task=task, verbose=verbose
    )
