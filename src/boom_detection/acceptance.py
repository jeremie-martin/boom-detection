"""
Acceptance functions for selective boom detection.

This module provides a flexible system for acceptance/rejection decisions.
Each acceptance function takes model predictions and quality, returning
(accepted: bool, accept_score: float).

Usage:
    from boom_detection.acceptance import get_acceptance_fn, ACCEPTANCE_REGISTRY

    # Get a registered formula
    accept_fn = get_acceptance_fn('sqrt', {'scale': 5.0, 'threshold': 0.60})
    accepted, score = accept_fn({'cnn': 50, 'hgb': 48}, predicted_quality=0.75)

    # Pass custom function
    def my_accept_fn(model_predictions, predicted_quality):
        # Custom logic
        return True, 0.8

    pipeline = BoomDetectionPipeline(custom_acceptance_fn=my_accept_fn)
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class AcceptanceFunction(Protocol):
    """
    Protocol for acceptance functions.

    Takes model predictions and predicted quality, returns (accepted, accept_score).
    The accept_score is useful for metrics, calibration, and debugging.
    """

    def __call__(
        self,
        model_predictions: dict[str, int],
        predicted_quality: float,
    ) -> tuple[bool, float]:
        """
        Evaluate acceptance for a simulation.

        Args:
            model_predictions: Dict mapping model name to predicted boom frame.
                              e.g., {'cnn': 50, 'hgb': 48}
            predicted_quality: Predicted quality score in [0, 1].

        Returns:
            Tuple of (accepted, accept_score) where:
            - accepted: True to accept prediction, False to reject
            - accept_score: Score used for the decision (useful for metrics)
        """
        ...


def _compute_disagreement(model_predictions: dict[str, int]) -> float:
    """Compute disagreement as range (max - min) of predictions."""
    preds = list(model_predictions.values())
    if len(preds) > 1:
        return float(max(preds) - min(preds))
    return 0.0


def make_sqrt_acceptance(params: dict | None = None) -> AcceptanceFunction:
    """
    Create sqrt-based acceptance function.

    agreement_score = 1 - sqrt(min(disagreement / scale, 1.0))

    This formula penalizes disagreement more strongly near zero,
    giving a "soft" transition as models diverge.

    Params:
        scale: float = 15.0 - Disagreement at which agreement becomes 0
        agreement_weight: float = 0.4 - Weight for agreement in accept_score
        quality_weight: float = 0.6 - Weight for quality in accept_score
        threshold: float = 0.60 - Accept if accept_score >= threshold

    Returns:
        AcceptanceFunction
    """
    params = params or {}
    scale = params.get('scale', 15.0)
    agreement_weight = params.get('agreement_weight', 0.4)
    quality_weight = params.get('quality_weight', 0.6)
    threshold = params.get('threshold', 0.60)

    def accept(
        model_predictions: dict[str, int],
        predicted_quality: float,
    ) -> tuple[bool, float]:
        disagreement = _compute_disagreement(model_predictions)
        agreement_score = 1.0 - np.sqrt(min(disagreement / scale, 1.0))
        accept_score = agreement_weight * agreement_score + quality_weight * predicted_quality
        return bool(accept_score >= threshold), float(accept_score)

    return accept


def make_linear_acceptance(params: dict | None = None) -> AcceptanceFunction:
    """
    Create linear acceptance function.

    agreement_score = 1 - min(disagreement / scale, 1.0)

    This formula provides a linear transition from full agreement (1.0)
    to no agreement (0.0) as disagreement increases.

    Params:
        scale: float = 10.0 - Disagreement at which agreement becomes 0
        agreement_weight: float = 0.4 - Weight for agreement in accept_score
        quality_weight: float = 0.6 - Weight for quality in accept_score
        threshold: float = 0.60 - Accept if accept_score >= threshold

    Returns:
        AcceptanceFunction
    """
    params = params or {}
    scale = params.get('scale', 10.0)
    agreement_weight = params.get('agreement_weight', 0.4)
    quality_weight = params.get('quality_weight', 0.6)
    threshold = params.get('threshold', 0.60)

    def accept(
        model_predictions: dict[str, int],
        predicted_quality: float,
    ) -> tuple[bool, float]:
        disagreement = _compute_disagreement(model_predictions)
        agreement_score = 1.0 - min(disagreement / scale, 1.0)
        accept_score = agreement_weight * agreement_score + quality_weight * predicted_quality
        return bool(accept_score >= threshold), float(accept_score)

    return accept


def make_sigmoid_acceptance(params: dict | None = None) -> AcceptanceFunction:
    """
    Create sigmoid-based acceptance function.

    agreement_score = 1 / (1 + exp(k * (disagreement/scale - 0.5)))

    This formula provides a smooth S-curve transition, with the steepest
    drop around disagreement = scale/2.

    Params:
        scale: float = 10.0 - Center of sigmoid transition
        steepness: float = 4.0 - Controls how sharp the transition is
        agreement_weight: float = 0.4 - Weight for agreement in accept_score
        quality_weight: float = 0.6 - Weight for quality in accept_score
        threshold: float = 0.60 - Accept if accept_score >= threshold

    Returns:
        AcceptanceFunction
    """
    params = params or {}
    scale = params.get('scale', 10.0)
    steepness = params.get('steepness', 4.0)
    agreement_weight = params.get('agreement_weight', 0.4)
    quality_weight = params.get('quality_weight', 0.6)
    threshold = params.get('threshold', 0.60)

    def accept(
        model_predictions: dict[str, int],
        predicted_quality: float,
    ) -> tuple[bool, float]:
        disagreement = _compute_disagreement(model_predictions)
        x = disagreement / scale
        agreement_score = 1.0 / (1.0 + np.exp(steepness * (x - 0.5)))
        accept_score = agreement_weight * agreement_score + quality_weight * predicted_quality
        return bool(accept_score >= threshold), float(accept_score)

    return accept


def make_quadratic_acceptance(params: dict | None = None) -> AcceptanceFunction:
    """
    Create quadratic acceptance function.

    agreement_score = 1 - min(disagreement / scale, 1.0)^2

    This formula is more permissive than linear for small disagreements,
    but drops off faster as disagreement approaches scale.

    Params:
        scale: float = 10.0 - Disagreement at which agreement becomes 0
        agreement_weight: float = 0.4 - Weight for agreement in accept_score
        quality_weight: float = 0.6 - Weight for quality in accept_score
        threshold: float = 0.60 - Accept if accept_score >= threshold

    Returns:
        AcceptanceFunction
    """
    params = params or {}
    scale = params.get('scale', 10.0)
    agreement_weight = params.get('agreement_weight', 0.4)
    quality_weight = params.get('quality_weight', 0.6)
    threshold = params.get('threshold', 0.60)

    def accept(
        model_predictions: dict[str, int],
        predicted_quality: float,
    ) -> tuple[bool, float]:
        disagreement = _compute_disagreement(model_predictions)
        x = min(disagreement / scale, 1.0)
        agreement_score = 1.0 - x * x
        accept_score = agreement_weight * agreement_score + quality_weight * predicted_quality
        return bool(accept_score >= threshold), float(accept_score)

    return accept


# Registry mapping formula names to factory functions
ACCEPTANCE_REGISTRY: dict[str, Callable[[dict | None], AcceptanceFunction]] = {
    'sqrt': make_sqrt_acceptance,
    'linear': make_linear_acceptance,
    'sigmoid': make_sigmoid_acceptance,
    'quadratic': make_quadratic_acceptance,
}


def get_acceptance_fn(formula: str, params: dict | None = None) -> AcceptanceFunction:
    """
    Get an acceptance function by name with parameters.

    Args:
        formula: Name of the formula ('sqrt', 'linear', 'sigmoid', 'quadratic')
        params: Dict of parameters for the formula. See each make_*_acceptance
                function for available parameters.

    Returns:
        AcceptanceFunction that can be called with (model_predictions, predicted_quality)

    Raises:
        ValueError: If formula is not in the registry

    Example:
        accept_fn = get_acceptance_fn('sqrt', {'scale': 5.0, 'threshold': 0.60})
        accepted, score = accept_fn({'cnn': 50, 'hgb': 48}, 0.75)
    """
    if formula not in ACCEPTANCE_REGISTRY:
        available = list(ACCEPTANCE_REGISTRY.keys())
        raise ValueError(f"Unknown formula: {formula}. Available: {available}")
    return ACCEPTANCE_REGISTRY[formula](params)


def list_formulas() -> list[str]:
    """Return list of available formula names."""
    return list(ACCEPTANCE_REGISTRY.keys())
