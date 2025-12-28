"""Boom detection dataset loader and utilities."""

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

from .evaluation import (
    Evaluator,
    EvaluationResult,
    FoldResult,
    cross_validate,
    compute_all_metrics,
    mae,
    median_ae,
    rmse,
)

from .features import (
    FeatureExtractor,
    FeatureConfig,
    FeatureCache,
    extract_features,
    get_feature_names,
    CAUSTIC_CONFIG,
)

from .frame_models import (
    FrameLevelRegressor,
    FrameLevelClassifier,
    get_frame_level_predictors,
)

from .sequence_models import (
    CNNClassifier,
    LSTMClassifier,
    TransformerClassifier,
    SequenceTrainer,
    get_sequence_models,
)

from .ensemble import (
    AdaptiveEnsemble,
)

from .quality_models import (
    MeanQualityPredictor,
    MedianQualityPredictor,
    QualityRegressor,
    QualityClassifier,
    BoomAwareQualityPredictor,
    get_quality_predictors,
)

from .pipeline import (
    QualityFilter,
    QualityAwarePipeline,
    ConditionalPipeline,
)

from .model_agreement import (
    ModelAgreementAnalyzer,
    ConfidenceWeightedEnsemble,
)

from .deploy_pipeline import (
    BoomDetectionPipeline,
    cross_validate as cv_pipeline,
)

__all__ = [
    # Loader
    "Annotation",
    "Dataset",
    "Simulation",
    "SimulationHeader",
    "load_annotations",
    "load_dataset",
    "load_simulation",
    "X1", "Y1", "X2", "Y2", "TH1", "TH2", "W1", "W2",
    "FIELD_NAMES",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "FoldResult",
    "cross_validate",
    "compute_all_metrics",
    "mae",
    "median_ae",
    "rmse",
    # Features
    "FeatureExtractor",
    "FeatureConfig",
    "FeatureCache",
    "extract_features",
    "get_feature_names",
    "CAUSTIC_CONFIG",
    # Frame-level models
    "FrameLevelRegressor",
    "FrameLevelClassifier",
    "get_frame_level_predictors",
    # Sequence models
    "CNNClassifier",
    "LSTMClassifier",
    "TransformerClassifier",
    "SequenceTrainer",
    "get_sequence_models",
    # Ensemble
    "AdaptiveEnsemble",
    # Quality models
    "MeanQualityPredictor",
    "MedianQualityPredictor",
    "QualityRegressor",
    "QualityClassifier",
    "BoomAwareQualityPredictor",
    "get_quality_predictors",
    # Pipeline
    "QualityFilter",
    "QualityAwarePipeline",
    "ConditionalPipeline",
    # Model agreement
    "ModelAgreementAnalyzer",
    "ConfidenceWeightedEnsemble",
    # Deployable pipeline
    "BoomDetectionPipeline",
    "cv_pipeline",
]
