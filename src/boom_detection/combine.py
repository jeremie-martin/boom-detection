"""
Combiner abstraction for boom detection ensemble pipeline.

Combiners take predictions from multiple models and produce a final prediction
(or None to reject). This unifies the "should we accept?" and "what's the final
prediction?" decisions into a single abstraction.

Usage:
    from boom_detection.combine import ThresholdCombiner, ModelPrediction

    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        threshold=0.60,
    )

    predictions = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 105)]
    result = combiner(predictions, predicted_quality=0.75)
    # result is 100 (accept) or None (reject)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Protocol

import numpy as np

# =============================================================================
# Core Types
# =============================================================================

@dataclass(frozen=True)
class FrameModelConfig:
    """
    Configuration for a frame-level model.

    Every model can have a quality threshold that filters training data.
    - quality_threshold=0.0: Train on ALL data (no filtering) - "regular" model
    - quality_threshold>0.0: Train only on samples with quality >= threshold - "specialized"

    This unifies the concept of "specialized" and "regular" models:
    a specialized model is simply a model trained on higher-quality data.

    Note: The quality threshold uses GROUND TRUTH quality during training
    (from annotations), not predicted quality. This is not "cheating" because
    we're just selecting which training samples to use.

    Examples:
        FrameModelConfig('hgb')           # Regular HGB, trained on all data
        FrameModelConfig('hgb', 0.5)      # Specialized HGB, trained on quality >= 0.5
        FrameModelConfig('cnn', 0.7)      # Specialized CNN, trained on quality >= 0.7
    """
    model_type: str  # 'cnn', 'hgb', 'lstm'
    quality_threshold: float = 0.0

    def __post_init__(self):
        if self.quality_threshold < 0.0 or self.quality_threshold > 1.0:
            raise ValueError(f"quality_threshold must be in [0, 1], got {self.quality_threshold}")

    @property
    def key(self) -> str:
        """
        Unique identifier for this model configuration.

        Returns:
            'hgb' for regular models, 'hgb_0.5' for specialized models
        """
        if self.quality_threshold == 0.0:
            return self.model_type
        return f"{self.model_type}_{self.quality_threshold}"

    @property
    def is_specialized(self) -> bool:
        """Whether this model is trained on filtered (high-quality) data."""
        return self.quality_threshold > 0.0

    @classmethod
    def from_spec(cls, spec: 'str | FrameModelConfig') -> 'FrameModelConfig':
        """
        Create from string or FrameModelConfig (for backward compatibility).

        Args:
            spec: Either a string like 'cnn', 'hgb_0.5' or a FrameModelConfig

        Returns:
            FrameModelConfig instance

        Examples:
            FrameModelConfig.from_spec('cnn')      # FrameModelConfig('cnn', 0.0)
            FrameModelConfig.from_spec('hgb_0.5')  # FrameModelConfig('hgb', 0.5)
        """
        if isinstance(spec, FrameModelConfig):
            return spec

        # Parse string like 'hgb' or 'hgb_0.5'
        if '_' in spec:
            parts = spec.rsplit('_', 1)
            try:
                threshold = float(parts[1])
                return cls(parts[0], threshold)
            except ValueError:
                # Not a number after underscore, treat whole string as model type
                pass

        return cls(spec, 0.0)

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        if self.quality_threshold == 0.0:
            return f"FrameModelConfig('{self.model_type}')"
        return f"FrameModelConfig('{self.model_type}', {self.quality_threshold})"


def normalize_model_configs(
    specs: 'tuple[str | FrameModelConfig, ...] | list[str | FrameModelConfig]'
) -> tuple[FrameModelConfig, ...]:
    """
    Normalize a sequence of model specs to FrameModelConfig instances.

    Args:
        specs: Tuple/list of strings or FrameModelConfig instances

    Returns:
        Tuple of FrameModelConfig instances

    Examples:
        normalize_model_configs(('cnn', 'hgb'))
        # -> (FrameModelConfig('cnn'), FrameModelConfig('hgb'))

        normalize_model_configs(('cnn', 'hgb_0.5'))
        # -> (FrameModelConfig('cnn'), FrameModelConfig('hgb', 0.5))
    """
    return tuple(FrameModelConfig.from_spec(s) for s in specs)


@dataclass
class ModelPrediction:
    """One model's prediction output."""
    model: str                        # e.g., 'cnn', 'hgb', 'lstm'
    frame: int                        # Predicted frame number
    confidence: float | None = None   # Optional self-reported confidence


class Combiner(Protocol):
    """
    Protocol for combiners that produce final predictions.

    A Combiner takes model predictions + quality + optional features
    and returns either a final frame prediction or None to reject.
    """

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        """
        Produce final prediction or reject.

        Args:
            predictions: List of ModelPrediction from each model
            predicted_quality: Predicted quality score in [0, 1]
            features: Optional raw features (for learned combiners)

        Returns:
            Final frame prediction, or None to reject
        """
        ...


# =============================================================================
# Utility Functions
# =============================================================================

def frames(predictions: list[ModelPrediction]) -> np.ndarray:
    """Extract frame values from predictions as numpy array."""
    return np.array([p.frame for p in predictions])


def get_prediction(predictions: list[ModelPrediction], model: str) -> ModelPrediction:
    """Get prediction from a specific model."""
    for p in predictions:
        if p.model == model:
            return p
    raise KeyError(f"Model '{model}' not found in predictions: {[p.model for p in predictions]}")


def median_frame(predictions: list[ModelPrediction]) -> int:
    """Compute median of model predictions."""
    return int(np.median(frames(predictions)))


def mean_frame(predictions: list[ModelPrediction]) -> int:
    """Compute mean of model predictions (rounded)."""
    return int(round(np.mean(frames(predictions))))


def disagreement(predictions: list[ModelPrediction], metric: str = 'range') -> float:
    """
    Compute disagreement between model predictions.

    Args:
        predictions: List of model predictions
        metric: How to measure disagreement:
            - 'range': max - min (default, used by current pipeline)
            - 'std': standard deviation
            - 'mad': median absolute deviation

    Returns:
        Disagreement value (0 = perfect agreement)
    """
    f = frames(predictions)
    if len(f) <= 1:
        return 0.0

    if metric == 'range':
        return float(np.max(f) - np.min(f))
    elif metric == 'std':
        return float(np.std(f))
    elif metric == 'mad':
        median = np.median(f)
        return float(np.median(np.abs(f - median)))
    else:
        raise ValueError(f"Unknown disagreement metric: {metric}")


def agreement_score(
    predictions: list[ModelPrediction],
    scale: float,
    transform: str = 'linear',
    metric: str = 'range',
) -> float:
    """
    Compute agreement score from model predictions.

    Args:
        predictions: List of model predictions
        scale: Disagreement at which agreement becomes 0
        transform: How to transform normalized disagreement:
            - 'linear': 1 - normalized
            - 'sqrt': 1 - sqrt(normalized)
            - 'quadratic': 1 - normalized^2
            - 'sigmoid': 1 / (1 + exp(4 * (normalized - 0.5)))
        metric: Disagreement metric ('range', 'std', 'mad')

    Returns:
        Agreement score in [0, 1] (higher = better agreement)
    """
    dis = disagreement(predictions, metric)
    normalized = min(dis / scale, 1.0)

    if transform == 'linear':
        return 1.0 - normalized
    elif transform == 'sqrt':
        return 1.0 - math.sqrt(normalized)
    elif transform == 'quadratic':
        return 1.0 - normalized * normalized
    elif transform == 'sigmoid':
        return 1.0 / (1.0 + math.exp(4.0 * (normalized - 0.5)))
    else:
        raise ValueError(f"Unknown transform: {transform}")


# =============================================================================
# Combiner Implementations
# =============================================================================

@dataclass
class ThresholdCombiner:
    """
    Combiner that reproduces the current acceptance formula behavior.

    Computes a combined score from agreement and predicted quality,
    then accepts if score >= threshold and returns the primary model's prediction.

    This is the main combiner for production use.

    If primary_model is None or 'median', uses median of all model predictions
    instead of a specific model's prediction.

    Score function options:
        - 'weighted': agreement_weight * agreement + quality_weight * quality
        - 'min': min(agreement, quality) - AND-style, both must be good
        - 'product': agreement * quality - multiplicative, penalizes weakness in either
    """
    primary_model: str | None = 'cnn'
    threshold: float = 0.60
    agreement_weight: float = 0.4
    quality_weight: float = 0.6
    disagreement_scale: float = 15.0
    disagreement_metric: str = 'range'
    agreement_transform: str = 'sqrt'
    score_function: str = 'weighted'  # Options: 'weighted', 'min', 'product'

    # Cached last score for debugging/metrics
    _last_score: float = field(default=0.0, repr=False, compare=False)
    _last_agreement: float = field(default=0.0, repr=False, compare=False)

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        """Apply threshold-based acceptance and return primary model prediction."""
        # Compute agreement score
        self._last_agreement = agreement_score(
            predictions,
            scale=self.disagreement_scale,
            transform=self.agreement_transform,
            metric=self.disagreement_metric,
        )

        # Compute combined score based on score_function
        if self.score_function == 'weighted':
            self._last_score = (
                self.agreement_weight * self._last_agreement +
                self.quality_weight * predicted_quality
            )
        elif self.score_function == 'min':
            self._last_score = min(self._last_agreement, predicted_quality)
        elif self.score_function == 'product':
            self._last_score = self._last_agreement * predicted_quality
        else:
            raise ValueError(f"Unknown score_function: {self.score_function}")

        # Accept if score >= threshold
        if self._last_score >= self.threshold:
            if self.primary_model is None or self.primary_model == 'median':
                return median_frame(predictions)
            return get_prediction(predictions, self.primary_model).frame
        return None

    def get_score(self) -> float:
        """Get the last computed accept score."""
        return self._last_score

    def get_agreement(self) -> float:
        """Get the last computed agreement score."""
        return self._last_agreement

    def to_config(self) -> dict:
        """Convert to serializable config dict."""
        return {
            'type': 'ThresholdCombiner',
            'params': {
                'primary_model': self.primary_model,
                'threshold': self.threshold,
                'agreement_weight': self.agreement_weight,
                'quality_weight': self.quality_weight,
                'disagreement_scale': self.disagreement_scale,
                'disagreement_metric': self.disagreement_metric,
                'agreement_transform': self.agreement_transform,
                'score_function': self.score_function,
            }
        }

    @classmethod
    def from_config(cls, config: dict) -> 'ThresholdCombiner':
        """Create from config dict."""
        return cls(**config['params'])

    @classmethod
    def grid(cls, **param_lists) -> list['ThresholdCombiner']:
        """
        Generate all combinations of parameters.

        Example:
            combiners = ThresholdCombiner.grid(
                agreement_transform=['sqrt', 'linear'],
                disagreement_scale=[5, 10, 15],
                threshold=[0.60],
            )
            # Returns 6 ThresholdCombiner instances
        """
        # Get default values
        defaults = {
            'primary_model': 'cnn',
            'threshold': 0.60,
            'agreement_weight': 0.4,
            'quality_weight': 0.6,
            'disagreement_scale': 15.0,
            'disagreement_metric': 'range',
            'agreement_transform': 'sqrt',
            'score_function': 'weighted',
        }

        # Convert single values to lists
        for key, value in param_lists.items():
            if not isinstance(value, (list, tuple)):
                param_lists[key] = [value]

        # Fill in defaults for unspecified params
        for key, default in defaults.items():
            if key not in param_lists:
                param_lists[key] = [default]

        # Generate all combinations
        keys = list(param_lists.keys())
        values = [param_lists[k] for k in keys]

        combiners = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            combiners.append(cls(**params))

        return combiners

    def __str__(self) -> str:
        return (f"ThresholdCombiner({self.agreement_transform}, "
                f"scale={self.disagreement_scale}, thresh={self.threshold})")

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MedianCombiner:
    """
    Always accept, use median of model predictions.

    Baseline combiner - no rejection.
    """

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        return median_frame(predictions)

    def to_config(self) -> dict:
        return {'type': 'MedianCombiner', 'params': {}}

    @classmethod
    def from_config(cls, config: dict) -> 'MedianCombiner':
        return cls()


@dataclass
class NamedModelCombiner:
    """
    Always accept, use a specific model's prediction.

    Useful as a baseline or when you trust one model most.
    """
    model: str = 'cnn'

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        return get_prediction(predictions, self.model).frame

    def to_config(self) -> dict:
        return {'type': 'NamedModelCombiner', 'params': {'model': self.model}}

    @classmethod
    def from_config(cls, config: dict) -> 'NamedModelCombiner':
        return cls(**config['params'])


@dataclass
class QualityGatedCombiner:
    """
    Accept if predicted quality >= threshold, use median.

    Simple quality-only gating.
    """
    threshold: float = 0.5

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        if predicted_quality >= self.threshold:
            # return median_frame(predictions)
            return mean_frame(predictions)  # TEMP JUST FOR PERF TESTING
        return None

    def to_config(self) -> dict:
        return {'type': 'QualityGatedCombiner', 'params': {'threshold': self.threshold}}

    @classmethod
    def from_config(cls, config: dict) -> 'QualityGatedCombiner':
        return cls(**config['params'])


@dataclass
class QualityGatedModelCombiner:
    """
    Accept if predicted quality >= threshold, use a specific model's prediction.

    This tests whether agreement helps PREDICTION even when using quality-only
    for ACCEPTANCE. Allows choosing the best individual model for final output.

    If primary_model is None or 'median', uses median of all predictions.
    """
    threshold: float = 0.7
    primary_model: str | None = 'cnn'

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        if predicted_quality >= self.threshold:
            if self.primary_model is None or self.primary_model == 'median':
                return median_frame(predictions)
            return get_prediction(predictions, self.primary_model).frame
        return None

    def to_config(self) -> dict:
        return {
            'type': 'QualityGatedModelCombiner',
            'params': {
                'threshold': self.threshold,
                'primary_model': self.primary_model,
            }
        }

    @classmethod
    def from_config(cls, config: dict) -> 'QualityGatedModelCombiner':
        return cls(**config['params'])


@dataclass
class SpecializedModelCombiner:
    """
    Accept if predicted quality >= threshold, use a specialized model for prediction.

    The specialized_model attribute tells the pipeline which trained specialized
    model to use for final prediction (e.g., 'hgb_0.4', 'cnn_0.5').

    This combiner is designed to work with pipelines that train multiple
    specialized models at different quality thresholds.
    """
    threshold: float = 0.7
    specialized_model: str = 'specialized'  # Key of specialized model to use

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        if predicted_quality >= self.threshold:
            # Return median as placeholder - pipeline will override with specialized model
            return median_frame(predictions)
        return None

    def to_config(self) -> dict:
        return {
            'type': 'SpecializedModelCombiner',
            'params': {
                'threshold': self.threshold,
                'specialized_model': self.specialized_model,
            }
        }

    @classmethod
    def from_config(cls, config: dict) -> 'SpecializedModelCombiner':
        return cls(**config['params'])


@dataclass
class AgreementGatedCombiner:
    """
    Accept if disagreement <= threshold, use median.

    Simple agreement-only gating.
    """
    max_disagreement: float = 10.0
    disagreement_metric: str = 'range'

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        dis = disagreement(predictions, self.disagreement_metric)
        if dis <= self.max_disagreement:
            return median_frame(predictions)
        return None

    def to_config(self) -> dict:
        return {
            'type': 'AgreementGatedCombiner',
            'params': {
                'max_disagreement': self.max_disagreement,
                'disagreement_metric': self.disagreement_metric,
            }
        }

    @classmethod
    def from_config(cls, config: dict) -> 'AgreementGatedCombiner':
        return cls(**config['params'])


@dataclass
class MajorityVoteCombiner:
    """
    Accept if majority of models agree within tolerance.

    Uses the consensus prediction when enough models agree.
    """
    tolerance: float = 10.0  # Models within this many frames are considered "agreeing"
    min_agreement_ratio: float = 0.5  # At least this fraction must agree

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        if len(predictions) < 2:
            return predictions[0].frame if predictions else None

        f = frames(predictions)
        n = len(f)

        # Find largest cluster of agreeing predictions
        best_count = 0
        best_center = None

        for i, center in enumerate(f):
            count = np.sum(np.abs(f - center) <= self.tolerance)
            if count > best_count:
                best_count = count
                best_center = center

        # Accept if enough models agree
        if best_count >= n * self.min_agreement_ratio:
            # Return median of agreeing predictions
            agreeing = f[np.abs(f - best_center) <= self.tolerance]
            return int(np.median(agreeing))
        return None

    def to_config(self) -> dict:
        return {
            'type': 'MajorityVoteCombiner',
            'params': {
                'tolerance': self.tolerance,
                'min_agreement_ratio': self.min_agreement_ratio,
            }
        }

    @classmethod
    def from_config(cls, config: dict) -> 'MajorityVoteCombiner':
        return cls(**config['params'])


# =============================================================================
# Serialization Registry
# =============================================================================

COMBINER_REGISTRY: dict[str, type] = {
    'ThresholdCombiner': ThresholdCombiner,
    'MedianCombiner': MedianCombiner,
    'NamedModelCombiner': NamedModelCombiner,
    'QualityGatedCombiner': QualityGatedCombiner,
    'QualityGatedModelCombiner': QualityGatedModelCombiner,
    'SpecializedModelCombiner': SpecializedModelCombiner,
    'AgreementGatedCombiner': AgreementGatedCombiner,
    'MajorityVoteCombiner': MajorityVoteCombiner,
}


def combiner_to_config(combiner: Any) -> dict:
    """
    Convert a combiner to a serializable config dict.

    Args:
        combiner: A combiner instance with to_config() method

    Returns:
        Dict with 'type' and 'params' keys
    """
    if hasattr(combiner, 'to_config'):
        return combiner.to_config()
    raise ValueError(f"Combiner {type(combiner).__name__} does not support serialization")


def combiner_from_config(config: dict) -> Any:
    """
    Create a combiner from a config dict.

    Args:
        config: Dict with 'type' and 'params' keys

    Returns:
        Combiner instance
    """
    combiner_type = config['type']
    if combiner_type not in COMBINER_REGISTRY:
        raise ValueError(f"Unknown combiner type: {combiner_type}. "
                        f"Available: {list(COMBINER_REGISTRY.keys())}")

    cls = COMBINER_REGISTRY[combiner_type]
    return cls.from_config(config)


# =============================================================================
# Helper to create default combiner
# =============================================================================

def default_combiner() -> ThresholdCombiner:
    """Create the default combiner (sqrt, scale=15, threshold=0.60)."""
    return ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        threshold=0.60,
    )
