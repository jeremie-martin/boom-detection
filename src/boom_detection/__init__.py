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
    extract_features,
    get_feature_names,
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
    "extract_features",
    "get_feature_names",
    # Baselines
    "MeanPredictor",
    "MedianPredictor",
    "VarianceThresholdPredictor",
    "DerivativeThresholdPredictor",
    "SecondDerivativePredictor",
    "LinearRegressionPredictor",
    "get_all_baselines",
]
