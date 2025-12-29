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

# Core metrics (pure numpy, no sklearn/scipy) - from metrics.py
from .metrics import (
    SelectivePrediction,
    compute_selective_metrics,
    compute_selective_metrics_with_rc,
    compute_risk_coverage_curve,
    compute_all_metrics,
    mae,
    median_ae,
    rmse,
    RunArtifact,
    # Decision-centric metrics
    coverage_at_max_mae,
    min_mae_at_coverage,
    find_optimal_threshold,
)

# Combiner abstraction (pure numpy, no ML dependencies)
from .combine import (
    ModelPrediction,
    Combiner,
    ThresholdCombiner,
    MedianCombiner,
    NamedModelCombiner,
    QualityGatedCombiner,
    AgreementGatedCombiner,
    MajorityVoteCombiner,
    combiner_to_config,
    combiner_from_config,
    default_combiner,
    # Utility functions
    frames,
    get_prediction,
    median_frame,
    mean_frame,
    disagreement,
    agreement_score,
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
    # Decision-centric metrics
    "coverage_at_max_mae",
    "min_mae_at_coverage",
    "find_optimal_threshold",
    # Combiner abstraction (always available)
    "ModelPrediction",
    "Combiner",
    "ThresholdCombiner",
    "MedianCombiner",
    "NamedModelCombiner",
    "QualityGatedCombiner",
    "AgreementGatedCombiner",
    "MajorityVoteCombiner",
    "combiner_to_config",
    "combiner_from_config",
    "default_combiner",
    "frames",
    "get_prediction",
    "median_frame",
    "mean_frame",
    "disagreement",
    "agreement_score",
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

    # Evaluation framework (requires sklearn + scipy)
    from .evaluation import (
        EvaluationResult,
        FoldResult,
        cross_validate,
        CachedEvaluator,
        CachedPredictor,
        CachedSelectivePredictor,
        MultiSeedResult,
        MultiSeedSelectiveResult,
        robust_evaluate,
        CombinerExperiment,
        CachedSample,
    )
    __all__.extend([
        "EvaluationResult",
        "FoldResult",
        "cross_validate",
        "CachedEvaluator",
        "CachedPredictor",
        "CachedSelectivePredictor",
        "MultiSeedResult",
        "MultiSeedSelectiveResult",
        "robust_evaluate",
        "CombinerExperiment",
        "CachedSample",
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
