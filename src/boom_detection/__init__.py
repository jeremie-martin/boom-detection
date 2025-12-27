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

from .baselines import (
    MeanPredictor,
    MedianPredictor,
    VarianceThresholdPredictor,
    DerivativeThresholdPredictor,
    SecondDerivativePredictor,
    LinearRegressionPredictor,
    get_all_baselines,
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
    # Baselines
    "MeanPredictor",
    "MedianPredictor",
    "VarianceThresholdPredictor",
    "DerivativeThresholdPredictor",
    "SecondDerivativePredictor",
    "LinearRegressionPredictor",
    "get_all_baselines",
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
]
