"""
Core metrics and types for boom detection evaluation.

This module has no ML dependencies (no sklearn, scipy, torch).
It can be imported on any Python installation with just numpy.

For cross-validation and evaluators, see boom_detection.evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Selective Prediction Types (for abstaining/rejection pipelines)
# =============================================================================

@dataclass
class SelectivePrediction:
    """
    A single prediction that may be accepted or rejected.

    This is the canonical format for selective (abstaining) predictors.
    Use this to ensure consistent handling across the codebase.

    Attributes:
        boom_frame: Predicted boom frame (None if rejected)
        accepted: Whether the prediction was accepted
        cnn_pred: CNN model prediction
        hgb_pred: HistGBM model prediction
        disagreement: Absolute difference between CNN and HGB predictions
        predicted_quality: Predicted quality score (0-1)
        accept_score: Single scalar combining all confidence signals (higher = more confident)
        confidence: Deprecated, use accept_score instead
    """
    boom_frame: int | None  # None if rejected
    accepted: bool
    cnn_pred: int
    hgb_pred: int
    disagreement: int
    predicted_quality: float
    accept_score: float | None = None  # Single scalar combining confidence signals
    confidence: float | None = None  # Deprecated - use accept_score

    @classmethod
    def from_dict(cls, d: dict) -> 'SelectivePrediction':
        """Create from a dictionary (e.g., from pipeline output)."""
        return cls(
            boom_frame=d.get('boom_frame'),
            accepted=d['accepted'],
            cnn_pred=d['cnn_pred'],
            hgb_pred=d['hgb_pred'],
            disagreement=d['disagreement'],
            predicted_quality=d['predicted_quality'],
            accept_score=d.get('accept_score'),
            confidence=d.get('confidence'),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            'boom_frame': self.boom_frame,
            'accepted': self.accepted,
            'cnn_pred': self.cnn_pred,
            'hgb_pred': self.hgb_pred,
            'disagreement': self.disagreement,
            'predicted_quality': self.predicted_quality,
        }
        if self.accept_score is not None:
            d['accept_score'] = self.accept_score
        if self.confidence is not None:
            d['confidence'] = self.confidence
        return d


# =============================================================================
# Basic Metrics (pure numpy)
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
# Selective (Abstaining) Metrics
# =============================================================================

def compute_selective_metrics(
    predictions: list[SelectivePrediction],
    true_booms: np.ndarray,
    true_qualities: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute metrics for selective (abstaining) predictions.

    These metrics separate coverage (acceptance rate) from accuracy.

    Args:
        predictions: List of SelectivePrediction objects
        true_booms: Ground truth boom frames
        true_qualities: Optional ground truth qualities

    Returns:
        Dictionary with selective metrics including:
        - n_total, n_accepted, coverage: Basic counts
        - selective_mae, selective_median_ae, selective_max_ae: Error metrics
        - selective_within_5, selective_within_10: Accuracy at thresholds
        - aurc: Area Under Risk-Coverage curve (lower is better)
        - optimal_coverage: Coverage at minimum risk-coverage product
    """
    n_total = len(predictions)
    accepted_indices = [i for i, p in enumerate(predictions) if p.accepted]
    n_accepted = len(accepted_indices)

    metrics: dict[str, float] = {
        'n_total': float(n_total),
        'n_accepted': float(n_accepted),
        'coverage': n_accepted / n_total if n_total > 0 else 0.0,
    }

    if n_accepted > 0:
        # Compute errors only for accepted predictions
        accepted_preds = [predictions[i] for i in accepted_indices]
        accepted_booms = true_booms[accepted_indices]

        errors = np.array([
            abs(p.boom_frame - t) for p, t in zip(accepted_preds, accepted_booms)
        ])

        metrics['selective_mae'] = float(np.mean(errors))
        metrics['selective_median_ae'] = float(np.median(errors))
        metrics['selective_max_ae'] = float(np.max(errors))
        metrics['selective_within_5'] = float(np.mean(errors <= 5))
        metrics['selective_within_10'] = float(np.mean(errors <= 10))

        # Quality of accepted simulations (if available)
        if true_qualities is not None:
            accepted_quals = true_qualities[accepted_indices]
            metrics['mean_accepted_quality'] = float(np.mean(accepted_quals))
            metrics['quality_precision'] = float(np.mean(accepted_quals >= 0.5))
    else:
        # No accepted predictions
        metrics['selective_mae'] = float('nan')
        metrics['selective_median_ae'] = float('nan')
        metrics['selective_max_ae'] = float('nan')
        metrics['selective_within_5'] = float('nan')
        metrics['selective_within_10'] = float('nan')

    # Rejection analysis
    rejected_indices = [i for i, p in enumerate(predictions) if not p.accepted]
    if rejected_indices and true_qualities is not None:
        rejected_quals = true_qualities[rejected_indices]
        metrics['rejected_high_quality_rate'] = float(np.mean(rejected_quals >= 0.5))

    # Add risk-coverage metrics (AURC)
    if n_total > 0:
        rc = compute_risk_coverage_curve(predictions, true_booms)
        metrics['aurc'] = rc['aurc']
        metrics['optimal_coverage'] = rc['optimal_coverage']

    return metrics


def compute_risk_coverage_curve(
    predictions: list[SelectivePrediction],
    true_booms: np.ndarray,
    score_key: str = 'accept_score',
) -> dict[str, Any]:
    """
    Compute risk-coverage curve for selective predictions.

    The risk-coverage curve shows how error (risk) changes as we accept more
    predictions (coverage). Lower area under the curve (AURC) is better.

    Args:
        predictions: List of SelectivePrediction objects with confidence scores
        true_booms: Ground truth boom frames
        score_key: Which field to use for ranking ('accept_score', 'confidence', or 'predicted_quality')

    Returns:
        Dictionary with:
            - coverages: Array of coverage values [0, 1]
            - risks: Array of risk (MAE) values at each coverage
            - aurc: Area Under Risk-Coverage curve
            - optimal_coverage: Coverage at minimum risk-coverage product
    """
    n = len(predictions)
    if n == 0:
        return {
            'coverages': np.array([]),
            'risks': np.array([]),
            'aurc': float('nan'),
            'optimal_coverage': 0.0,
        }

    # Get confidence scores and errors
    confidences = []
    errors = []
    for i, pred in enumerate(predictions):
        # Get confidence score (prefer accept_score, fall back to others)
        if score_key == 'accept_score' and pred.accept_score is not None:
            conf = pred.accept_score
        elif score_key == 'confidence' and pred.confidence is not None:
            conf = pred.confidence
        else:
            conf = pred.predicted_quality

        # Compute error (use CNN prediction as it's more accurate)
        error = abs(pred.cnn_pred - true_booms[i])

        confidences.append(conf)
        errors.append(error)

    confidences = np.array(confidences)
    errors = np.array(errors)

    # Sort by confidence (highest first)
    order = np.argsort(-confidences)
    sorted_errors = errors[order]

    # Compute cumulative risk at each coverage level
    coverages = []
    risks = []

    for k in range(1, n + 1):
        coverage = k / n
        risk = np.mean(sorted_errors[:k])  # MAE of top-k most confident
        coverages.append(coverage)
        risks.append(risk)

    coverages = np.array(coverages)
    risks = np.array(risks)

    # Compute AURC (Area Under Risk-Coverage curve) using trapezoidal rule
    aurc = float(np.trapezoid(risks, coverages))

    # Find optimal coverage (minimizes risk * (1 - coverage) or similar)
    risk_coverage_product = risks * (1 - coverages + 0.1)
    optimal_idx = np.argmin(risk_coverage_product)
    optimal_coverage = float(coverages[optimal_idx])

    return {
        'coverages': coverages,
        'risks': risks,
        'aurc': aurc,
        'optimal_coverage': optimal_coverage,
        'risk_at_optimal': float(risks[optimal_idx]),
    }


def compute_selective_metrics_with_rc(
    predictions: list[SelectivePrediction],
    true_booms: np.ndarray,
    true_qualities: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute selective metrics including risk-coverage analysis.

    This combines compute_selective_metrics with compute_risk_coverage_curve.
    """
    # Get basic selective metrics
    metrics = compute_selective_metrics(predictions, true_booms, true_qualities)

    # Add risk-coverage analysis
    rc = compute_risk_coverage_curve(predictions, true_booms)
    metrics['aurc'] = rc['aurc']
    metrics['optimal_coverage'] = rc['optimal_coverage']
    metrics['risk_at_optimal'] = rc.get('risk_at_optimal', float('nan'))

    # Store the full curve for plotting (not JSON-serializable)
    metrics['_risk_coverage_curve'] = rc

    return metrics


# =============================================================================
# Decision-Centric Metrics (for threshold tuning)
# =============================================================================

def coverage_at_max_mae(
    predictions: list[SelectivePrediction],
    true_booms: np.ndarray,
    max_mae: float,
    score_key: str = 'accept_score',
) -> float:
    """
    Find maximum achievable coverage while keeping MAE <= max_mae.

    This answers: "What fraction can we accept if we require MAE ≤ X frames?"

    Args:
        predictions: Selective predictions with accept scores
        true_booms: Ground truth boom frames
        max_mae: Maximum acceptable MAE
        score_key: Field to use for ranking ('accept_score' or 'predicted_quality')

    Returns:
        Maximum coverage (0-1) that achieves the MAE target
    """
    n = len(predictions)
    if n == 0:
        return 0.0

    # Get scores and errors
    scores = []
    errors = []
    for i, pred in enumerate(predictions):
        if score_key == 'accept_score' and pred.accept_score is not None:
            score = pred.accept_score
        else:
            score = pred.predicted_quality
        error = abs(pred.cnn_pred - true_booms[i])
        scores.append(score)
        errors.append(error)

    scores = np.array(scores)
    errors = np.array(errors)

    # Sort by score (highest first)
    order = np.argsort(-scores)
    sorted_errors = errors[order]

    # Find maximum k where MAE of top-k <= max_mae
    best_k = 0
    for k in range(1, n + 1):
        if np.mean(sorted_errors[:k]) <= max_mae:
            best_k = k
        else:
            break  # Once we exceed, we can't recover

    return best_k / n


def min_mae_at_coverage(
    predictions: list[SelectivePrediction],
    true_booms: np.ndarray,
    target_coverage: float,
    score_key: str = 'accept_score',
) -> float:
    """
    Find minimum achievable MAE while accepting at least target_coverage.

    This answers: "What's the best accuracy if we must accept at least X%?"

    Args:
        predictions: Selective predictions with accept scores
        true_booms: Ground truth boom frames
        target_coverage: Minimum required coverage (0-1)
        score_key: Field to use for ranking ('accept_score' or 'predicted_quality')

    Returns:
        Minimum MAE achievable at the target coverage
    """
    n = len(predictions)
    if n == 0:
        return float('nan')

    # Get scores and errors
    scores = []
    errors = []
    for i, pred in enumerate(predictions):
        if score_key == 'accept_score' and pred.accept_score is not None:
            score = pred.accept_score
        else:
            score = pred.predicted_quality
        error = abs(pred.cnn_pred - true_booms[i])
        scores.append(score)
        errors.append(error)

    scores = np.array(scores)
    errors = np.array(errors)

    # Sort by score (highest first)
    order = np.argsort(-scores)
    sorted_errors = errors[order]

    # How many samples must we accept to reach target_coverage?
    k = max(1, int(np.ceil(target_coverage * n)))
    k = min(k, n)  # Don't exceed total samples

    return float(np.mean(sorted_errors[:k]))


def find_optimal_threshold(
    predictions: list[SelectivePrediction],
    true_booms: np.ndarray,
    target_mae: float | None = None,
    target_coverage: float | None = None,
    score_key: str = 'accept_score',
) -> dict[str, float]:
    """
    Find optimal accept_score threshold for a given target.

    Either target_mae or target_coverage should be specified, not both.

    Args:
        predictions: Selective predictions with accept scores
        true_booms: Ground truth boom frames
        target_mae: Target maximum MAE (find threshold to achieve this)
        target_coverage: Target minimum coverage (find threshold to achieve this)
        score_key: Field to use ('accept_score' or 'predicted_quality')

    Returns:
        Dict with 'threshold', 'coverage', 'mae', and 'n_accepted'
    """
    n = len(predictions)
    if n == 0:
        return {'threshold': 0.0, 'coverage': 0.0, 'mae': float('nan'), 'n_accepted': 0}

    # Get scores and errors
    scores = []
    errors = []
    for i, pred in enumerate(predictions):
        if score_key == 'accept_score' and pred.accept_score is not None:
            score = pred.accept_score
        else:
            score = pred.predicted_quality
        error = abs(pred.cnn_pred - true_booms[i])
        scores.append(score)
        errors.append(error)

    scores = np.array(scores)
    errors = np.array(errors)

    # Sort by score (highest first)
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_errors = errors[order]

    if target_mae is not None:
        # Find threshold that gives MAE <= target_mae
        best_k = 0
        for k in range(1, n + 1):
            if np.mean(sorted_errors[:k]) <= target_mae:
                best_k = k

        if best_k == 0:
            # Can't achieve target even at lowest coverage
            return {
                'threshold': float(sorted_scores[0]) + 0.01,
                'coverage': 0.0,
                'mae': float('nan'),
                'n_accepted': 0,
            }

        threshold = float(sorted_scores[best_k - 1]) - 0.001
        return {
            'threshold': threshold,
            'coverage': best_k / n,
            'mae': float(np.mean(sorted_errors[:best_k])),
            'n_accepted': best_k,
        }

    elif target_coverage is not None:
        # Find threshold that gives at least target_coverage
        k = max(1, int(np.ceil(target_coverage * n)))
        k = min(k, n)

        threshold = float(sorted_scores[k - 1]) - 0.001
        return {
            'threshold': threshold,
            'coverage': k / n,
            'mae': float(np.mean(sorted_errors[:k])),
            'n_accepted': k,
        }

    else:
        raise ValueError("Either target_mae or target_coverage must be specified")


# =============================================================================
# Run Artifacts (for reproducibility and results tracking)
# =============================================================================

@dataclass
class RunArtifact:
    """
    Captures a complete evaluation run for reproducibility.

    Stores config, metrics, per-simulation predictions, and environment info.
    Can be saved to disk and loaded later for analysis.

    Usage:
        artifact = RunArtifact.create(
            config={...},
            predictions=[...],
            true_booms=[...],
            true_qualities=[...],
        )
        artifact.save(Path('runs/2025-01-01_exp1'))

        # Later
        artifact = RunArtifact.load(Path('runs/2025-01-01_exp1'))
    """
    config: dict
    metrics: dict
    predictions: list[dict]  # Per-simulation predictions
    environment: dict
    timestamp: str = ""

    @classmethod
    def create(
        cls,
        config: dict,
        predictions: list[SelectivePrediction] | list[dict],
        true_booms: np.ndarray,
        true_qualities: np.ndarray | None = None,
        sim_ids: list[str] | None = None,
    ) -> 'RunArtifact':
        """
        Create a run artifact from predictions.

        Args:
            config: Pipeline/model configuration
            predictions: List of SelectivePrediction or dicts
            true_booms: Ground truth boom frames
            true_qualities: Ground truth qualities
            sim_ids: Simulation IDs (optional)

        Returns:
            RunArtifact ready for saving
        """
        import datetime
        import sys
        import subprocess

        # Convert predictions to dicts if needed
        if predictions and isinstance(predictions[0], SelectivePrediction):
            pred_dicts = [p.to_dict() for p in predictions]
            selective_preds = list(predictions)
        else:
            pred_dicts = list(predictions)
            selective_preds = [SelectivePrediction.from_dict(p) for p in predictions]

        # Add ground truth to predictions
        for i, pd in enumerate(pred_dicts):
            pd['true_boom'] = int(true_booms[i])
            if true_qualities is not None:
                pd['true_quality'] = float(true_qualities[i])
            if sim_ids is not None:
                pd['sim_id'] = sim_ids[i]

        # Compute metrics
        metrics = compute_selective_metrics(
            selective_preds, true_booms, true_qualities
        )

        # Capture environment
        try:
            git_commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()[:8]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            git_commit = "unknown"

        environment = {
            'python_version': sys.version.split()[0],
            'git_commit': git_commit,
        }

        timestamp = datetime.datetime.now().isoformat()

        return cls(
            config=config,
            metrics=metrics,
            predictions=pred_dicts,
            environment=environment,
            timestamp=timestamp,
        )

    def save(self, path: Path) -> None:
        """
        Save artifact to directory.

        Creates:
            path/config.json
            path/metrics.json
            path/predictions.jsonl
            path/environment.json
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        with open(path / 'metrics.json', 'w') as f:
            # Filter out non-serializable items
            serializable_metrics = {
                k: v for k, v in self.metrics.items()
                if not k.startswith('_')
            }
            json.dump(serializable_metrics, f, indent=2)

        with open(path / 'predictions.jsonl', 'w') as f:
            for pred in self.predictions:
                f.write(json.dumps(pred) + '\n')

        env_with_meta = {
            **self.environment,
            'timestamp': self.timestamp,
        }
        with open(path / 'environment.json', 'w') as f:
            json.dump(env_with_meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'RunArtifact':
        """Load artifact from directory."""
        path = Path(path)

        with open(path / 'config.json') as f:
            config = json.load(f)

        with open(path / 'metrics.json') as f:
            metrics = json.load(f)

        predictions = []
        with open(path / 'predictions.jsonl') as f:
            for line in f:
                predictions.append(json.loads(line))

        with open(path / 'environment.json') as f:
            env_data = json.load(f)
            timestamp = env_data.pop('timestamp', '')
            environment = env_data

        return cls(
            config=config,
            metrics=metrics,
            predictions=predictions,
            environment=environment,
            timestamp=timestamp,
        )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Run Artifact ({self.timestamp})",
            "=" * 60,
            "",
            "Metrics:",
        ]

        for name, value in sorted(self.metrics.items()):
            if name.startswith('_'):
                continue
            if isinstance(value, float):
                if name.startswith('selective_within') or name.endswith('rate') or name == 'coverage':
                    lines.append(f"  {name}: {value:.1%}")
                elif not np.isnan(value):
                    lines.append(f"  {name}: {value:.2f}")
                else:
                    lines.append(f"  {name}: N/A")
            else:
                lines.append(f"  {name}: {value}")

        lines.extend([
            "",
            "Config:",
        ])
        for key, value in sorted(self.config.items()):
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            f"Environment: Python {self.environment.get('python_version', '?')}, "
            f"git {self.environment.get('git_commit', '?')}",
        ])

        return "\n".join(lines)
