"""
Evaluation framework for boom detection experiments.

Provides cross-validation with proper uncertainty estimates.
This module requires sklearn and scipy.

For core metrics without ML dependencies, see boom_detection.metrics.

Usage:
    from boom_detection.evaluation import CachedEvaluator
    from boom_detection import load_dataset

    dataset = load_dataset('data')
    cache = FeatureCache(FeatureConfig(), cache_dir='.feature_cache')
    cache.extract_all(dataset)

    evaluator = CachedEvaluator(dataset, cache)
    result = evaluator.cross_validate(lambda: MyModel())
    print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TYPE_CHECKING

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from .logging_config import logger

# Import core metrics and types (no sklearn dependency)
from .metrics import (
    SelectivePrediction,
    compute_all_metrics,
    compute_selective_metrics,
    RunArtifact,
    mae,
    median_ae,
    rmse,
)

# Re-export for convenience
__all__ = [
    # From metrics (re-exported)
    'SelectivePrediction',
    'compute_all_metrics',
    'compute_selective_metrics',
    'RunArtifact',
    'mae',
    'median_ae',
    'rmse',
    # CV-specific
    'FoldResult',
    'EvaluationResult',
    'MultiSeedResult',
    'MultiSeedSelectiveResult',
    'CachedPredictor',
    'CachedSelectivePredictor',
    'CachedEvaluator',
    'robust_evaluate',
    'cross_validate',
    # Formula experiment
    'CachedRawPrediction',
    'FormulaExperiment',
]

if TYPE_CHECKING:
    from .features import FeatureCache
    from .loader import Dataset


# =============================================================================
# Results Containers
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

        return "\n".join(lines)

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


# =============================================================================
# Multi-Seed Results
# =============================================================================

@dataclass
class MultiSeedResult:
    """
    Results from multi-seed cross-validation.

    Provides proper uncertainty estimates (mean +/- std with confidence intervals).
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
            "Robust Cross-Validation Results",
            f"  {self.k}-fold CV x {len(self.seeds)} seeds = {self.k * len(self.seeds)} evaluations",
            f"  Seeds: {self.seeds}",
            "=" * 60,
            "",
        ]

        # Primary metric with full details
        pm = primary_metric
        lines.append(f"Primary metric ({pm}):")
        lines.append(f"  {self.mean_metrics[pm]:.2f} +/- {self.std_metrics[pm]:.2f}")
        lines.append(f"  95% CI: [{self.ci_lower[pm]:.2f}, {self.ci_upper[pm]:.2f}]")
        lines.append("")

        # All metrics
        lines.append("All metrics (mean +/- std):")
        for name in sorted(self.mean_metrics.keys()):
            mean = self.mean_metrics[name]
            std = self.std_metrics[name]

            if name.startswith('within'):
                lines.append(f"  {name}: {mean:.1%} +/- {std:.1%}")
            else:
                lines.append(f"  {name}: {mean:.2f} +/- {std:.2f}")

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


@dataclass
class MultiSeedSelectiveResult:
    """
    Results from multi-seed cross-validation for selective (abstaining) predictors.

    Provides proper uncertainty estimates for selective metrics like coverage and
    selective_mae separately.
    """
    k: int
    seeds: list[int]
    seed_metrics: list[dict[str, float]]

    # Aggregated statistics (computed in __post_init__)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    ci_lower: dict[str, float] = field(default_factory=dict)
    ci_upper: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.seed_metrics) > 0 and len(self.mean_metrics) == 0:
            self._compute_statistics()

    def _compute_statistics(self):
        """Compute mean, std, and 95% CI across seeds."""
        # Get all metric names from first result
        metric_names = list(self.seed_metrics[0].keys())

        for name in metric_names:
            values = [m[name] for m in self.seed_metrics
                     if not np.isnan(m.get(name, float('nan')))]
            n = len(values)

            if n > 0:
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
            else:
                self.mean_metrics[name] = float('nan')
                self.std_metrics[name] = 0.0
                self.ci_lower[name] = float('nan')
                self.ci_upper[name] = float('nan')

    def summary(self) -> str:
        """Generate summary with confidence intervals."""
        lines = [
            "Selective Cross-Validation Results",
            f"  {self.k}-fold CV x {len(self.seeds)} seeds",
            f"  Seeds: {self.seeds}",
            "=" * 60,
            "",
        ]

        # Primary metrics
        if 'selective_mae' in self.mean_metrics:
            mae_val = self.mean_metrics['selective_mae']
            mae_std = self.std_metrics.get('selective_mae', 0)
            lines.append(f"Selective MAE: {mae_val:.2f} +/- {mae_std:.2f}")
            lines.append(f"  95% CI: [{self.ci_lower.get('selective_mae', mae_val):.2f}, "
                        f"{self.ci_upper.get('selective_mae', mae_val):.2f}]")

        if 'coverage' in self.mean_metrics:
            cov = self.mean_metrics['coverage']
            cov_std = self.std_metrics.get('coverage', 0)
            lines.append(f"Coverage: {cov:.1%} +/- {cov_std:.1%}")

        lines.append("")
        lines.append("All metrics (mean +/- std):")

        for name in sorted(self.mean_metrics.keys()):
            mean = self.mean_metrics[name]
            std = self.std_metrics.get(name, 0)

            if np.isnan(mean):
                lines.append(f"  {name}: N/A")
            elif name.startswith('selective_within') or name == 'coverage':
                lines.append(f"  {name}: {mean:.1%} +/- {std:.1%}")
            elif name in ('n_total', 'n_accepted'):
                lines.append(f"  {name}: {mean:.0f}")
            else:
                lines.append(f"  {name}: {mean:.2f} +/- {std:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'k': self.k,
            'seeds': self.seeds,
            'mean_metrics': self.mean_metrics,
            'std_metrics': self.std_metrics,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'seed_metrics': self.seed_metrics,
        }


# =============================================================================
# Formula Experiment Framework
# =============================================================================

@dataclass
class CachedRawPrediction:
    """
    Cached raw prediction from a pipeline, before acceptance decision.

    This stores all information needed to recompute acceptance with different
    formula parameters, without retraining or re-running inference.
    """
    sim_id: str
    model_predictions: dict[str, int]  # e.g., {'cnn': 546, 'hgb': 542}
    predicted_quality: float
    true_boom: int
    true_quality: float

    @property
    def primary_prediction(self) -> int:
        """Return first model's prediction (typically the primary model)."""
        return next(iter(self.model_predictions.values()))


class FormulaExperiment:
    """
    Experiment framework for iterating on acceptance formulas.

    Trains models once, caches raw predictions, then allows fast iteration
    on different acceptance formulas WITHOUT retraining.

    This is the canonical way to experiment with formula parameters.
    Results are guaranteed to be IDENTICAL to running the full pipeline.

    Usage:
        evaluator = CachedEvaluator(dataset, cache)

        # Create experiment (trains models once, caches predictions)
        experiment = evaluator.create_formula_experiment(
            pipeline_factory=lambda: BoomDetectionPipeline(),
            seeds=[42, 43, 44],
        )

        # Now iterate on formulas instantly:
        for scale in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
            result = experiment.evaluate(
                agreement_formula='sqrt',
                agreement_scale=scale,
                accept_threshold=0.60,
            )
            print(f"scale={scale}: MAE {result.mean_metrics['selective_mae']:.2f}")
    """

    def __init__(
        self,
        cached_predictions: list[list[CachedRawPrediction]],
        seeds: list[int],
        k: int,
        primary_model: str = 'cnn',
    ):
        """
        Args:
            cached_predictions: List of lists, one per seed. Each inner list
                               contains CachedRawPrediction for all test samples
                               across all folds for that seed.
            seeds: The random seeds used for cross-validation.
            k: Number of folds used.
            primary_model: Which model's prediction to use when accepted.
        """
        self.cached_predictions = cached_predictions
        self.seeds = seeds
        self.k = k
        self.primary_model = primary_model

    def evaluate(
        self,
        agreement_formula: str = 'sqrt',
        agreement_scale: float = 15.0,
        agreement_weight: float = 0.4,
        quality_weight: float = 0.6,
        accept_threshold: float = 0.60,
        verbose: bool = False,
    ) -> MultiSeedSelectiveResult:
        """
        Evaluate a formula configuration on cached predictions.

        This is FAST - no model training or inference, just formula evaluation.

        Args:
            agreement_formula: 'sqrt' or 'linear'
            agreement_scale: Scale for agreement (higher = more permissive)
            agreement_weight: Weight for agreement in accept_score
            quality_weight: Weight for quality in accept_score
            accept_threshold: Threshold on accept_score for acceptance

        Returns:
            MultiSeedSelectiveResult with metrics for this configuration
        """
        all_seed_metrics = []

        for seed_idx, seed_preds in enumerate(self.cached_predictions):
            # Apply formula to all predictions for this seed
            selective_predictions = []
            true_booms = []
            true_qualities = []

            for pred in seed_preds:
                # Compute acceptance using EXACT same formula as deploy_pipeline.py
                result = self._apply_formula(
                    pred,
                    agreement_formula=agreement_formula,
                    agreement_scale=agreement_scale,
                    agreement_weight=agreement_weight,
                    quality_weight=quality_weight,
                    accept_threshold=accept_threshold,
                )
                selective_predictions.append(result)
                true_booms.append(pred.true_boom)
                true_qualities.append(pred.true_quality)

            # Compute metrics for this seed
            metrics = compute_selective_metrics(
                selective_predictions,
                np.array(true_booms),
                np.array(true_qualities),
            )
            all_seed_metrics.append(metrics)

            if verbose:
                n_accepted = sum(1 for p in selective_predictions if p.accepted)
                print(f"  Seed {self.seeds[seed_idx]}: {n_accepted}/{len(selective_predictions)} accepted, "
                      f"MAE={metrics.get('selective_mae', float('nan')):.2f}")

        return MultiSeedSelectiveResult(
            k=self.k,
            seeds=self.seeds,
            seed_metrics=all_seed_metrics,
        )

    def _apply_formula(
        self,
        pred: CachedRawPrediction,
        agreement_formula: str,
        agreement_scale: float,
        agreement_weight: float,
        quality_weight: float,
        accept_threshold: float,
    ) -> SelectivePrediction:
        """
        Apply acceptance formula to a cached prediction.

        IMPORTANT: This MUST match deploy_pipeline.py exactly!
        """
        # Compute disagreement as range of predictions (0 = perfect agreement)
        # Using max-min is consistent with original 2-model abs(a-b) formula
        pred_values = list(pred.model_predictions.values())
        if len(pred_values) > 1:
            disagreement = float(max(pred_values) - min(pred_values))
        else:
            disagreement = 0.0

        # Compute agreement score (1.0 = perfect, 0.0 = max disagreement)
        if agreement_formula == 'sqrt':
            agreement_score = 1.0 - np.sqrt(min(disagreement / agreement_scale, 1.0))
        else:
            agreement_score = 1.0 - min(disagreement / agreement_scale, 1.0)

        # Combine agreement and quality into accept_score
        accept_score = (
            agreement_weight * agreement_score +
            quality_weight * pred.predicted_quality
        )

        # Accept if score exceeds threshold
        accepted = accept_score >= accept_threshold

        # Use primary model's prediction for boom_frame
        primary_pred = pred.model_predictions.get(
            self.primary_model,
            pred.primary_prediction  # fallback to first model
        )

        return SelectivePrediction(
            boom_frame=primary_pred if accepted else None,
            accepted=accepted,
            accept_score=float(accept_score),
            predicted_quality=pred.predicted_quality,
            model_predictions=pred.model_predictions,
            disagreement=disagreement,
        )

    def sweep(
        self,
        scales: list[float] | None = None,
        formulas: list[str] | None = None,
        accept_thresholds: list[float] | None = None,
        verbose: bool = True,
    ) -> dict[tuple, MultiSeedSelectiveResult]:
        """
        Run a sweep over multiple formula configurations.

        Args:
            scales: List of agreement_scale values to try
            formulas: List of agreement_formula values ('sqrt', 'linear')
            accept_thresholds: List of accept_threshold values to try
            verbose: Print progress

        Returns:
            Dict mapping (formula, scale, threshold) -> MultiSeedSelectiveResult
        """
        if scales is None:
            scales = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
        if formulas is None:
            formulas = ['sqrt']
        if accept_thresholds is None:
            accept_thresholds = [0.60]

        results = {}
        total = len(scales) * len(formulas) * len(accept_thresholds)
        count = 0

        if verbose:
            print(f"\nRunning sweep: {len(scales)} scales x {len(formulas)} formulas x {len(accept_thresholds)} thresholds = {total} configs")
            print("-" * 70)

        for formula in formulas:
            for scale in scales:
                for threshold in accept_thresholds:
                    count += 1
                    result = self.evaluate(
                        agreement_formula=formula,
                        agreement_scale=scale,
                        accept_threshold=threshold,
                        verbose=False,
                    )
                    key = (formula, scale, threshold)
                    results[key] = result

                    if verbose:
                        mae = result.mean_metrics.get('selective_mae', float('nan'))
                        mae_std = result.std_metrics.get('selective_mae', 0)
                        cov = result.mean_metrics.get('coverage', 0)
                        print(f"  [{count}/{total}] {formula}/scale={scale:.0f}/thresh={threshold:.2f}: "
                              f"MAE {mae:.2f} +/- {mae_std:.2f} at {cov:.1%} coverage")

        return results


# =============================================================================
# Predictor Protocols
# =============================================================================

class CachedPredictor(Protocol):
    """
    Protocol for predictors that work with FeatureCache.

    This is the interface used by all models in this codebase:
    - FrameLevelClassifier, FrameLevelRegressor
    - SequenceTrainer (CNN, LSTM, Transformer)
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


class CachedSelectivePredictor(Protocol):
    """
    Protocol for selective (abstaining) predictors that work with FeatureCache.

    These predictors can reject simulations they're uncertain about.
    """

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        qualities: np.ndarray,
        cache: 'FeatureCache'
    ) -> None:
        """Fit the model on training data (requires quality scores)."""
        ...

    def predict(
        self,
        sim_ids: list[str],
        cache: 'FeatureCache'
    ) -> list[SelectivePrediction]:
        """Predict with abstention for simulations."""
        ...


# =============================================================================
# CachedEvaluator - The One Blessed Evaluator
# =============================================================================

class CachedEvaluator:
    """
    The primary evaluator for all experiments.

    Uses the (sim_ids, targets, cache) interface that all models implement.
    Runs multiple seeds by default for robust estimates.

    Usage:
        from boom_detection.features import FeatureCache, FeatureConfig
        from boom_detection.evaluation import CachedEvaluator

        cache = FeatureCache(FeatureConfig(), cache_dir='.feature_cache')
        cache.extract_all(dataset)

        evaluator = CachedEvaluator(dataset, cache)

        # Quick single-seed evaluation (for development)
        result = evaluator.cross_validate(lambda: MyModel(), seeds=[42])

        # Robust multi-seed evaluation (for reporting)
        result = evaluator.cross_validate(lambda: MyModel())  # Uses 5 seeds
        print(result.summary())
    """

    def __init__(self, dataset: 'Dataset', cache: 'FeatureCache'):
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
        """
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]  # 5 seeds by default

        targets = self.boom_frames if task == "frame" else self.boom_qualities
        seed_results = []

        logger.info("Starting {}-fold CV with {} seeds on {} samples",
                   k, len(seeds), len(self.sim_ids))
        total_start = time.time()

        for seed_idx, seed in enumerate(seeds):
            logger.info("Seed {} ({}/{})", seed, seed_idx + 1, len(seeds))
            seed_start = time.time()

            # Run single-seed CV
            result = self._cross_validate_single_seed(
                predictor_fn, k=k, seed=seed, targets=targets, task=task, verbose=verbose
            )
            seed_results.append(result)

            seed_elapsed = time.time() - seed_start
            logger.info("Seed {} complete in {:.1f}s (MAE: {:.2f})",
                       seed, seed_elapsed, result.aggregate_metrics['mae'])

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
            print(f"MAE: {multi_result.mean_metrics['mae']:.2f} +/- {multi_result.std_metrics['mae']:.2f}")
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
            fold_start = time.time()

            # Derive fold-specific seed for model training
            fold_seed = seed * 1000 + fold_idx

            # Get train/test data
            train_ids = [self.sim_ids[i] for i in train_idx]
            test_ids = [self.sim_ids[i] for i in test_idx]
            train_y = targets[train_idx]
            test_y = targets[test_idx]

            # Create fresh predictor and train
            logger.debug("  Fold {}/{}: training on {} samples...", fold_idx + 1, k, len(train_ids))
            predictor = predictor_fn()

            # Pass fold seed if the predictor supports it
            if hasattr(predictor, 'set_seed'):
                predictor.set_seed(fold_seed)

            train_start = time.time()
            predictor.fit(train_ids, train_y, self.cache)
            train_time = time.time() - train_start

            # Predict
            pred_start = time.time()
            predictions = predictor.predict(test_ids, self.cache)
            predictions = np.asarray(predictions)
            pred_time = time.time() - pred_start

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

            fold_elapsed = time.time() - fold_start
            logger.debug("  Fold {}/{}: MAE={:.2f} (train: {:.1f}s, pred: {:.1f}s, total: {:.1f}s)",
                        fold_idx + 1, k, metrics['mae'], train_time, pred_time, fold_elapsed)

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

    # =========================================================================
    # Selective (Abstaining) Predictor Support
    # =========================================================================

    def cross_validate_selective(
        self,
        predictor_fn: Callable[[], CachedSelectivePredictor],
        k: int = 5,
        seeds: list[int] | None = None,
        verbose: bool = True,
    ) -> MultiSeedSelectiveResult:
        """
        Run robust multi-seed cross-validation for selective (abstaining) predictors.

        IMPORTANT: predictor_fn must be a factory function that returns a NEW
        predictor instance each time. This ensures each seed gets a fresh model.

        Args:
            predictor_fn: Factory function returning a selective predictor
                         The predictor must implement:
                         - fit(sim_ids, boom_frames, qualities, cache)
                         - predict(sim_ids, cache) -> list[SelectivePrediction]
            k: Number of folds (default: 5)
            seeds: List of random seeds (default: [42, 43, 44, 45, 46])
            verbose: Print progress

        Returns:
            MultiSeedSelectiveResult with uncertainty estimates
        """
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]  # 5 seeds by default

        all_seed_metrics = []

        for seed_idx, seed in enumerate(seeds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Seed {seed} ({seed_idx + 1}/{len(seeds)})")
                print('='*50)

            # Run single-seed CV
            metrics = self._cross_validate_selective_single_seed(
                predictor_fn, k=k, seed=seed, verbose=verbose
            )
            all_seed_metrics.append(metrics)

        result = MultiSeedSelectiveResult(
            k=k,
            seeds=seeds,
            seed_metrics=all_seed_metrics,
        )

        if verbose:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print('='*60)
            mae_val = result.mean_metrics.get('selective_mae', float('nan'))
            mae_std = result.std_metrics.get('selective_mae', 0)
            cov = result.mean_metrics.get('coverage', 0)
            cov_std = result.std_metrics.get('coverage', 0)
            print(f"Selective MAE: {mae_val:.2f} +/- {mae_std:.2f}")
            print(f"Coverage: {cov:.1%} +/- {cov_std:.1%}")

        return result

    def _cross_validate_selective_single_seed(
        self,
        predictor_fn: Callable[[], CachedSelectivePredictor],
        k: int,
        seed: int,
        verbose: bool,
    ) -> dict[str, float]:
        """Run selective CV with a single seed."""
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        all_predictions: list[SelectivePrediction] = []
        all_true_booms: list[int] = []
        all_true_quals: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sim_ids)):
            # Derive fold-specific seed for model training
            fold_seed = seed * 1000 + fold_idx

            # Get train/test data
            train_ids = [self.sim_ids[i] for i in train_idx]
            test_ids = [self.sim_ids[i] for i in test_idx]
            train_booms = self.boom_frames[train_idx]
            train_quals = self.boom_qualities[train_idx]

            # Create fresh predictor and train
            predictor = predictor_fn()

            # Pass fold seed if the predictor supports it
            if hasattr(predictor, 'set_seed'):
                predictor.set_seed(fold_seed)

            predictor.fit(train_ids, train_booms, train_quals, self.cache)

            # Predict - get list of SelectivePrediction
            predictions = predictor.predict(test_ids, self.cache)

            # Handle both SelectivePrediction and dict outputs
            for pred in predictions:
                if isinstance(pred, SelectivePrediction):
                    all_predictions.append(pred)
                else:
                    all_predictions.append(SelectivePrediction.from_dict(pred))

            all_true_booms.extend(self.boom_frames[test_idx].tolist())
            all_true_quals.extend(self.boom_qualities[test_idx].tolist())

            if verbose:
                n_accepted = sum(1 for p in predictions
                               if (isinstance(p, SelectivePrediction) and p.accepted)
                               or (isinstance(p, dict) and p.get('accepted', False)))
                print(f"  Fold {fold_idx + 1}/{k}: {n_accepted}/{len(predictions)} accepted")

        # Compute metrics using unified framework
        metrics = compute_selective_metrics(
            all_predictions,
            np.array(all_true_booms),
            np.array(all_true_quals),
        )

        return metrics

    def quick_evaluate_selective(
        self,
        predictor_fn: Callable[[], CachedSelectivePredictor],
        k: int = 5,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, float]:
        """
        Quick single-seed evaluation for selective predictors.

        Returns just the metrics dict for fast development iteration.
        """
        return self._cross_validate_selective_single_seed(
            predictor_fn, k=k, seed=seed, verbose=verbose
        )

    # =========================================================================
    # Formula Experiment Factory
    # =========================================================================

    def create_formula_experiment(
        self,
        predictor_fn: Callable[[], CachedSelectivePredictor],
        k: int = 5,
        seeds: list[int] | None = None,
        verbose: bool = True,
    ) -> FormulaExperiment:
        """
        Create a FormulaExperiment by training models once and caching predictions.

        This is the canonical way to experiment with acceptance formulas.
        Models are trained once, then you can iterate on formula parameters
        WITHOUT retraining.

        GUARANTEE: Results from FormulaExperiment.evaluate() are IDENTICAL
        to running cross_validate_selective() with the same formula parameters.

        Args:
            predictor_fn: Factory function returning a selective predictor
                         (must have fit() and predict() methods)
            k: Number of folds (default: 5)
            seeds: List of random seeds (default: [42, 43, 44])
            verbose: Print progress

        Returns:
            FormulaExperiment for iterating on formulas

        Example:
            # Train once (slow)
            experiment = evaluator.create_formula_experiment(
                lambda: BoomDetectionPipeline(),
                seeds=[42, 43, 44],
            )

            # Iterate on formulas (fast)
            for scale in [5, 10, 15, 20]:
                result = experiment.evaluate(agreement_scale=scale)
                print(f"scale={scale}: MAE {result.mean_metrics['selective_mae']:.2f}")
        """
        if seeds is None:
            seeds = [42, 43, 44]  # 3 seeds by default for faster iteration

        all_cached_predictions: list[list[CachedRawPrediction]] = []
        primary_model = 'cnn'  # Will be updated from predictor

        total_start = time.time()

        for seed_idx, seed in enumerate(seeds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training seed {seed} ({seed_idx + 1}/{len(seeds)})")
                print('='*50)

            seed_predictions: list[CachedRawPrediction] = []
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sim_ids)):
                fold_start = time.time()
                fold_seed = seed * 1000 + fold_idx

                # Get train/test data
                train_ids = [self.sim_ids[i] for i in train_idx]
                test_ids = [self.sim_ids[i] for i in test_idx]
                train_booms = self.boom_frames[train_idx]
                train_quals = self.boom_qualities[train_idx]

                # Create fresh predictor and train
                predictor = predictor_fn()

                # Capture primary_model if available
                if hasattr(predictor, 'primary_model'):
                    primary_model = predictor.primary_model

                # Pass fold seed if the predictor supports it
                if hasattr(predictor, 'set_seed'):
                    predictor.set_seed(fold_seed)

                predictor.fit(train_ids, train_booms, train_quals, self.cache)

                # Get predictions from pipeline
                predictions = predictor.predict(test_ids, self.cache)

                # Extract raw predictions (model_predictions + predicted_quality)
                for pred, test_i in zip(predictions, test_idx):
                    if isinstance(pred, SelectivePrediction):
                        sp = pred
                    else:
                        sp = SelectivePrediction.from_dict(pred)

                    seed_predictions.append(CachedRawPrediction(
                        sim_id=self.sim_ids[test_i],
                        model_predictions=sp.model_predictions,
                        predicted_quality=sp.predicted_quality,
                        true_boom=int(self.boom_frames[test_i]),
                        true_quality=float(self.boom_qualities[test_i]),
                    ))

                fold_elapsed = time.time() - fold_start
                if verbose:
                    print(f"  Fold {fold_idx + 1}/{k}: {len(predictions)} predictions ({fold_elapsed:.1f}s)")

            all_cached_predictions.append(seed_predictions)

        total_elapsed = time.time() - total_start
        if verbose:
            total_preds = sum(len(sp) for sp in all_cached_predictions)
            print(f"\n{'='*50}")
            print(f"Training complete in {total_elapsed:.1f}s")
            print(f"Cached {total_preds} raw predictions ({len(seeds)} seeds x ~{len(self.sim_ids)} sims)")
            print("Now you can iterate on formulas WITHOUT retraining!")
            print('='*50)

        return FormulaExperiment(
            cached_predictions=all_cached_predictions,
            seeds=seeds,
            k=k,
            primary_model=primary_model,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def cross_validate(
    dataset: 'Dataset',
    cache: 'FeatureCache',
    predictor_fn: Callable[[], CachedPredictor],
    k: int = 5,
    seeds: list[int] | None = None,
    task: str = "frame",
    verbose: bool = True,
) -> MultiSeedResult:
    """
    Convenience function for cross-validation.

    Equivalent to CachedEvaluator(dataset, cache).cross_validate(...)
    """
    return CachedEvaluator(dataset, cache).cross_validate(
        predictor_fn, k=k, seeds=seeds, task=task, verbose=verbose
    )


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
        MultiSeedResult with mean +/- std +/- CI
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
