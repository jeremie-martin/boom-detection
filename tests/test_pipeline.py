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
        pipeline.cnn = "mock"  # Not None so we pass the fitted check
        pipeline.hgb = "mock"

        with pytest.raises(TypeError, match="Expected ndarray"):
            pipeline.predict_one(sample_features.tolist())  # List instead of array

    def test_predict_with_wrong_ndim_raises(self, mock_cache):
        """Passing 1D or 3D array should raise ValueError."""
        pipeline = BoomDetectionPipeline()
        pipeline.n_features = 183
        pipeline.cnn = "mock"
        pipeline.hgb = "mock"

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
        pipeline.cnn = "mock"
        pipeline.hgb = "mock"

        # Wrong feature count
        with pytest.raises(ValueError, match="Expected 183 features"):
            pipeline.predict_one(np.zeros((100, 50)))


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_thresholds(self):
        """Default thresholds should match documented values."""
        pipeline = BoomDetectionPipeline()
        assert pipeline.agreement_threshold == 5
        assert pipeline.quality_threshold == 0.55
        assert pipeline.quality_window == 25
        assert pipeline.n_quality_features == 50

    def test_custom_thresholds(self):
        """Custom thresholds should be stored correctly."""
        pipeline = BoomDetectionPipeline(
            agreement_threshold=3,
            quality_threshold=0.7,
            quality_window=30,
            n_quality_features=100,
        )
        assert pipeline.agreement_threshold == 3
        assert pipeline.quality_threshold == 0.7
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

    def test_predict_one_returns_dict(self, trained_pipeline):
        """predict_one should return a dict with expected keys."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert isinstance(result, dict)
        expected_keys = {
            'boom_frame', 'accepted', 'cnn_pred', 'hgb_pred',
            'disagreement', 'predicted_quality'
        }
        assert set(result.keys()) == expected_keys

    def test_predict_one_boom_frame_type(self, trained_pipeline):
        """boom_frame should be int or None."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert result['boom_frame'] is None or isinstance(result['boom_frame'], int)

    def test_predict_one_accepted_is_bool(self, trained_pipeline):
        """accepted should be a boolean."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert isinstance(result['accepted'], bool)

    def test_predict_one_predictions_are_ints(self, trained_pipeline):
        """cnn_pred and hgb_pred should be ints."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert isinstance(result['cnn_pred'], int)
        assert isinstance(result['hgb_pred'], int)

    def test_predict_one_quality_in_range(self, trained_pipeline):
        """predicted_quality should be in [0, 1]."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        assert 0 <= result['predicted_quality'] <= 1

    def test_predict_multiple(self, trained_pipeline):
        """predict should work for multiple simulations."""
        pipeline, cache = trained_pipeline
        sim_ids = [f"sim_{i}" for i in range(5)]
        results = pipeline.predict(sim_ids, cache)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
            assert 'boom_frame' in result


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
        """Disagreement should be |cnn_pred - hgb_pred|."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        expected_disagreement = abs(result['cnn_pred'] - result['hgb_pred'])
        assert result['disagreement'] == expected_disagreement

    def test_boom_frame_none_when_rejected(self, trained_pipeline):
        """boom_frame should be None when rejected."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        if not result['accepted']:
            assert result['boom_frame'] is None

    def test_boom_frame_set_when_accepted(self, trained_pipeline):
        """boom_frame should equal cnn_pred when accepted."""
        pipeline, cache = trained_pipeline
        result = pipeline.predict_one(cache["sim_0"])

        if result['accepted']:
            assert result['boom_frame'] == result['cnn_pred']
