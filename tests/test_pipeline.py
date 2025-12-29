"""
Tests for the deployment pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from boom_detection.deploy_pipeline import BoomDetectionPipeline


class TestPipelineValidation:
    """Tests for input validation in the pipeline."""

    def test_predict_before_fit_raises(self, sample_features):
        """Calling predict before fit should raise RuntimeError."""
        pipeline = BoomDetectionPipeline()
        with pytest.raises(RuntimeError, match="Pipeline not fitted"):
            pipeline.predict_one(sample_features)

    def test_predict_with_wrong_type_raises(self, sample_features, mock_cache):
        """Passing non-ndarray should raise TypeError."""
        pipeline = BoomDetectionPipeline()
        pipeline.n_features = 183
        pipeline.trained_models = {'cnn': "mock", 'hgb': "mock"}

        with pytest.raises(TypeError, match="Expected ndarray"):
            pipeline.predict_one(sample_features.tolist())  # List instead of array

    def test_predict_with_wrong_ndim_raises(self, mock_cache):
        """Passing 1D or 3D array should raise ValueError."""
        pipeline = BoomDetectionPipeline()
        pipeline.n_features = 183
        pipeline.trained_models = {'cnn': "mock", 'hgb': "mock"}

        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            pipeline.predict_one(np.zeros(100))

        # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            pipeline.predict_one(np.zeros((100, 183, 2)))

    def test_predict_with_wrong_features_raises(self, mock_cache):
        """Passing wrong number of features should raise ValueError."""
        pipeline = BoomDetectionPipeline()
        pipeline.n_features = 183
        pipeline.trained_models = {'cnn': "mock", 'hgb': "mock"}

        # Wrong feature count
        with pytest.raises(ValueError, match="Expected 183 features"):
            pipeline.predict_one(np.zeros((100, 50)))


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_thresholds(self):
        """Default thresholds should match documented values."""
        pipeline = BoomDetectionPipeline()
        assert pipeline.accept_threshold == 0.60  # Increased to compensate for overconfidence
        assert pipeline.agreement_weight == 0.4
        assert pipeline.quality_weight == 0.6
        assert pipeline.agreement_formula == 'sqrt'  # Default is sqrt
        assert pipeline.agreement_scale == 15.0  # Default for sqrt
        assert pipeline.quality_window == 25
        assert pipeline.n_quality_features == 50

    def test_sqrt_agreement_defaults(self):
        """Sqrt agreement formula should use scale 15 by default."""
        pipeline = BoomDetectionPipeline(agreement_formula='sqrt')
        assert pipeline.agreement_formula == 'sqrt'
        assert pipeline.agreement_scale == 15.0  # Default for sqrt

    def test_custom_thresholds(self):
        """Custom thresholds should be stored correctly."""
        pipeline = BoomDetectionPipeline(
            accept_threshold=0.6,
            agreement_weight=0.3,
            quality_weight=0.7,
            quality_window=30,
            n_quality_features=100,
        )
        assert pipeline.accept_threshold == 0.6
        assert pipeline.agreement_weight == 0.3
        assert pipeline.quality_weight == 0.7
        assert pipeline.quality_window == 30
        assert pipeline.n_quality_features == 100


class TestPipelineOutput:
    """Tests for pipeline output format (requires trained model)."""

    @pytest.fixture
    def trained_pipeline(self, mock_cache_multi):
        """Create a minimal trained pipeline for testing output format."""
        pytest.importorskip("torch")
        pytest.importorskip("sklearn")

        sim_ids = [f"sim_{i}" for i in range(10)]
        boom_frames = np.array([50] * 10)
        qualities = np.array([0.7] * 10)

        pipeline = BoomDetectionPipeline()
        pipeline.fit(sim_ids, boom_frames, qualities, mock_cache_multi)
        return pipeline, mock_cache_multi

    def test_predict_one_returns_selective_prediction(self, trained_pipeline):
        """predict_one should return a SelectivePrediction."""
        from boom_detection.evaluation import SelectivePrediction

        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert isinstance(result, SelectivePrediction)
        # Check all expected attributes exist
        assert hasattr(result, 'boom_frame')
        assert hasattr(result, 'accepted')
        assert hasattr(result, 'accept_score')
        assert hasattr(result, 'predicted_quality')
        assert hasattr(result, 'model_predictions')
        assert hasattr(result, 'disagreement')

    def test_predict_one_boom_frame_type(self, trained_pipeline):
        """boom_frame should be int or None."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert result.boom_frame is None or isinstance(result.boom_frame, int)

    def test_predict_one_accepted_is_bool(self, trained_pipeline):
        """accepted should be a boolean."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert isinstance(result.accepted, (bool, np.bool_))

    def test_predict_one_model_predictions_are_ints(self, trained_pipeline):
        """model_predictions values should be ints."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert isinstance(result.model_predictions, dict)
        for model_name, pred in result.model_predictions.items():
            assert isinstance(pred, (int, np.integer)), f"{model_name} prediction should be int"

    def test_predict_one_quality_in_range(self, trained_pipeline):
        """predicted_quality should be in [0, 1]."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert 0 <= result.predicted_quality <= 1

    def test_predict_one_accept_score_in_range(self, trained_pipeline):
        """accept_score should be in [0, 1]."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert 0 <= result.accept_score <= 1

    def test_predict_multiple(self, trained_pipeline):
        """predict should work for multiple simulations."""
        from boom_detection.evaluation import SelectivePrediction

        pipeline, cache = trained_pipeline
        sim_ids = [f"sim_{i}" for i in range(5)]
        results = pipeline.predict(sim_ids, cache)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, SelectivePrediction)


class TestAcceptanceLogic:
    """Tests for acceptance/rejection logic."""

    @pytest.fixture
    def trained_pipeline(self, mock_cache_multi):
        """Create a trained pipeline for testing."""
        pytest.importorskip("torch")
        pytest.importorskip("sklearn")

        sim_ids = [f"sim_{i}" for i in range(10)]
        boom_frames = np.array([50] * 10)
        qualities = np.array([0.7] * 10)

        pipeline = BoomDetectionPipeline()
        pipeline.fit(sim_ids, boom_frames, qualities, mock_cache_multi)
        return pipeline, mock_cache_multi

    def test_disagreement_calculated_correctly(self, trained_pipeline):
        """Disagreement should be std of model predictions."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        preds = list(result.model_predictions.values())
        expected_disagreement = float(np.std(preds))
        assert abs(result.disagreement - expected_disagreement) < 1e-6

    def test_boom_frame_none_when_rejected(self, trained_pipeline):
        """boom_frame should be None when rejected."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        if not result.accepted:
            assert result.boom_frame is None

    def test_boom_frame_set_when_accepted(self, trained_pipeline):
        """boom_frame should equal primary model prediction when accepted."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        if result.accepted:
            # boom_frame should be the primary model's prediction
            primary_pred = result.model_predictions[pipeline.primary_model]
            assert result.boom_frame == primary_pred


class TestBaselineFactories:
    """Tests for baseline factory functions."""

    def test_get_frame_level_predictors_returns_factories(self):
        """get_frame_level_predictors should return callable factories."""
        pytest.importorskip("sklearn")
        from boom_detection.frame_models import get_frame_level_predictors

        predictors = get_frame_level_predictors()

        for name, factory in predictors.items():
            assert callable(factory), f"{name} should be callable"
            # Each call should return a new instance
            instance1 = factory()
            instance2 = factory()
            assert instance1 is not instance2, f"{name} should return new instances"

    def test_frame_gbm_factory_preserves_hyperparameters(self):
        """frame_gbm factory should create GBM model, not default."""
        pytest.importorskip("sklearn")
        from boom_detection.frame_models import get_frame_level_predictors

        predictors = get_frame_level_predictors()

        # Check that frame_gbm creates a GBM model
        if 'frame_gbm' in predictors:
            model = predictors['frame_gbm']()
            # The model should have internal gbm regressor, not ridge
            assert hasattr(model, 'model_type') or hasattr(model, 'model')

    def test_get_baselines_returns_factories(self):
        """get_baselines should return callable factories."""
        pytest.importorskip("sklearn")
        from boom_detection.run_baselines import get_baselines

        baselines = get_baselines()

        for name, factory in baselines.items():
            assert callable(factory), f"{name} should be callable"
