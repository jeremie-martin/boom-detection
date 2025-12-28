"""Boom detection dataset loader and utilities."""

# =============================================================================
# Core exports (always available - no ML dependencies required)
# =============================================================================

from .loader import (
    Annotation,
    Dataset,
    Simulation,
    SimulationHeader,
    load_annotations,
    load_dataset,
    load_simulation,
    X1, Y1, X2, Y2, TH1, TH2, W1, W2,
    FIELD_NAMES,
)

# Evaluation types that don't require ML
from .evaluation import (
    SelectivePrediction,
    compute_selective_metrics,
    compute_selective_metrics_with_rc,
    compute_risk_coverage_curve,
    compute_all_metrics,
    mae,
    median_ae,
    rmse,
    RunArtifact,
)

__all__ = [
    # Loader (always available)
    "Annotation",
    "Dataset",
    "Simulation",
    "SimulationHeader",
    "load_annotations",
    "load_dataset",
    "load_simulation",
    "X1", "Y1", "X2", "Y2", "TH1", "TH2", "W1", "W2",
    "FIELD_NAMES",
    # Core evaluation (always available)
    "SelectivePrediction",
    "compute_selective_metrics",
    "compute_selective_metrics_with_rc",
    "compute_risk_coverage_curve",
    "compute_all_metrics",
    "mae",
    "median_ae",
    "rmse",
    "RunArtifact",
]

# =============================================================================
# Optional ML-dependent exports (require --extra ml)
# =============================================================================

_ML_AVAILABLE = False

try:
    # Features module requires scipy for some functions
    from .features import (
        FeatureExtractor,
        FeatureConfig,
        FeatureCache,
        extract_features,
        get_feature_names,
        CAUSTIC_CONFIG,
        PRODUCTION_CONFIG,
    )
    __all__.extend([
        "FeatureExtractor",
        "FeatureConfig",
        "FeatureCache",
        "extract_features",
        "get_feature_names",
        "CAUSTIC_CONFIG",
        "PRODUCTION_CONFIG",
    ])

    # Evaluation framework (requires sklearn)
    from .evaluation import (
        Evaluator,
        EvaluationResult,
        FoldResult,
        cross_validate,
        CachedEvaluator,
        CachedPredictor,
        CachedSelectivePredictor,
        MultiSeedResult,
        MultiSeedSelectiveResult,
        robust_evaluate,
    )
    __all__.extend([
        "Evaluator",
        "EvaluationResult",
        "FoldResult",
        "cross_validate",
        "CachedEvaluator",
        "CachedPredictor",
        "CachedSelectivePredictor",
        "MultiSeedResult",
        "MultiSeedSelectiveResult",
        "robust_evaluate",
    ])

    # Frame-level models (require sklearn)
    from .frame_models import (
        FrameLevelRegressor,
        FrameLevelClassifier,
        get_frame_level_predictors,
    )
    __all__.extend([
        "FrameLevelRegressor",
        "FrameLevelClassifier",
        "get_frame_level_predictors",
    ])

    # Quality models (require sklearn)
    from .quality_models import (
        MeanQualityPredictor,
        MedianQualityPredictor,
        QualityRegressor,
        QualityClassifier,
        BoomAwareQualityPredictor,
        get_quality_predictors,
    )
    __all__.extend([
        "MeanQualityPredictor",
        "MedianQualityPredictor",
        "QualityRegressor",
        "QualityClassifier",
        "BoomAwareQualityPredictor",
        "get_quality_predictors",
    ])

    # Pipeline components (require sklearn)
    from .pipeline import (
        QualityFilter,
        QualityAwarePipeline,
        ConditionalPipeline,
    )
    __all__.extend([
        "QualityFilter",
        "QualityAwarePipeline",
        "ConditionalPipeline",
    ])

    # Model agreement (require sklearn)
    from .model_agreement import (
        ModelAgreementAnalyzer,
        ConfidenceWeightedEnsemble,
    )
    __all__.extend([
        "ModelAgreementAnalyzer",
        "ConfidenceWeightedEnsemble",
    ])

    _ML_AVAILABLE = True

except ImportError:
    pass

# =============================================================================
# Optional torch-dependent exports (require torch)
# =============================================================================

_TORCH_AVAILABLE = False

try:
    from .sequence_models import (
        CNNClassifier,
        LSTMClassifier,
        TransformerClassifier,
        SequenceTrainer,
        get_sequence_models,
    )
    __all__.extend([
        "CNNClassifier",
        "LSTMClassifier",
        "TransformerClassifier",
        "SequenceTrainer",
        "get_sequence_models",
    ])

    from .ensemble import (
        AdaptiveEnsemble,
    )
    __all__.extend([
        "AdaptiveEnsemble",
    ])

    # Deploy pipeline requires both sklearn and torch
    if _ML_AVAILABLE:
        from .deploy_pipeline import (
            BoomDetectionPipeline,
            cross_validate as cv_pipeline,
        )
        __all__.extend([
            "BoomDetectionPipeline",
            "cv_pipeline",
        ])

    _TORCH_AVAILABLE = True

except ImportError:
    pass


def check_ml_dependencies() -> dict[str, bool]:
    """
    Check which optional ML dependencies are available.

    Returns:
        Dict with availability status for each dependency group.
    """
    return {
        "ml": _ML_AVAILABLE,  # sklearn, scipy
        "torch": _TORCH_AVAILABLE,  # pytorch
        "full": _ML_AVAILABLE and _TORCH_AVAILABLE,  # all dependencies
    }
