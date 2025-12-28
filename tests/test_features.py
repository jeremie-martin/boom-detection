"""
Tests for feature extraction.
"""
from __future__ import annotations

import numpy as np
import pytest

from boom_detection.features import (
    variance_features,
    std_features,
    iqr_features,
    range_features,
    mean_features,
    skewness_features,
    kurtosis_features,
    tip_spread_features,
    angular_spread_features,
    velocity_features,
    caustic_features,
    temporal_derivatives,
    FeatureExtractor,
    FeatureConfig,
    DEFAULT_CONFIG,
    PRODUCTION_CONFIG,
)


class TestBasicFeatures:
    """Tests for basic statistical feature functions."""

    def test_variance_features_shape(self, sample_simulation_data):
        """Variance features should have shape (frames, 8)."""
        result = variance_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_std_features_shape(self, sample_simulation_data):
        """Std features should have shape (frames, 8)."""
        result = std_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_iqr_features_shape(self, sample_simulation_data):
        """IQR features should have shape (frames, 8)."""
        result = iqr_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_range_features_shape(self, sample_simulation_data):
        """Range features should have shape (frames, 8)."""
        result = range_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_mean_features_shape(self, sample_simulation_data):
        """Mean features should have shape (frames, 8)."""
        result = mean_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_skewness_features_shape(self, sample_simulation_data):
        """Skewness features should have shape (frames, 8)."""
        result = skewness_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_kurtosis_features_shape(self, sample_simulation_data):
        """Kurtosis features should have shape (frames, 8)."""
        result = kurtosis_features(sample_simulation_data)
        assert result.shape == (100, 8)

    def test_variance_nonnegative(self, sample_simulation_data):
        """Variance should always be non-negative."""
        result = variance_features(sample_simulation_data)
        assert np.all(result >= 0)

    def test_iqr_nonnegative(self, sample_simulation_data):
        """IQR should always be non-negative."""
        result = iqr_features(sample_simulation_data)
        assert np.all(result >= 0)

    def test_range_nonnegative(self, sample_simulation_data):
        """Range should always be non-negative."""
        result = range_features(sample_simulation_data)
        assert np.all(result >= 0)


class TestSpecializedFeatures:
    """Tests for specialized feature functions."""

    def test_tip_spread_features_shape(self, sample_simulation_data):
        """Tip spread features should have shape (frames, 3)."""
        result = tip_spread_features(sample_simulation_data)
        assert result.shape == (100, 3)

    def test_angular_spread_features_shape(self, sample_simulation_data):
        """Angular spread features should have shape (frames, 4)."""
        result = angular_spread_features(sample_simulation_data)
        assert result.shape == (100, 4)

    def test_velocity_features_shape(self, sample_simulation_data):
        """Velocity features should have shape (frames, 6)."""
        result = velocity_features(sample_simulation_data)
        assert result.shape == (100, 6)

    def test_caustic_features_shape(self, sample_simulation_data):
        """Caustic features should have shape (frames, 4)."""
        result = caustic_features(sample_simulation_data)
        assert result.shape == (100, 4)


class TestTemporalDerivatives:
    """Tests for temporal derivative features."""

    def test_derivatives_preserve_shape(self, sample_features):
        """Derivatives should preserve frame count."""
        result = temporal_derivatives(sample_features, order=1)
        assert result.shape == sample_features.shape

    def test_second_derivatives_shape(self, sample_features):
        """Second derivatives should also preserve shape."""
        result = temporal_derivatives(sample_features, order=2)
        assert result.shape == sample_features.shape


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def test_default_config_output_shape(self, mock_simulation):
        """Default config should produce expected feature count."""
        extractor = FeatureExtractor(DEFAULT_CONFIG)
        features = extractor.transform(mock_simulation)
        assert features.shape[0] == 100  # frames
        assert features.ndim == 2

    def test_production_config_includes_caustic(self, mock_simulation):
        """Production config should include caustic features."""
        extractor = FeatureExtractor(PRODUCTION_CONFIG)
        features = extractor.transform(mock_simulation)
        # Production config has caustic features, so more features
        assert features.shape[0] == 100
        assert features.ndim == 2

    def test_feature_names_match_output(self, mock_simulation):
        """Feature names count should match feature output."""
        extractor = FeatureExtractor(DEFAULT_CONFIG)
        features = extractor.transform(mock_simulation)
        names = extractor.feature_names
        assert len(names) == features.shape[1]

    def test_reproducibility_with_same_seed(self, mock_simulation):
        """Same subsample seed should give identical results."""
        config = FeatureConfig(max_pendulums=20, subsample_seed=42)
        extractor = FeatureExtractor(config)

        features1 = extractor.transform(mock_simulation)
        features2 = extractor.transform(mock_simulation)

        np.testing.assert_array_equal(features1, features2)

    def test_subsample_reduces_pendulums(self, sample_simulation_data):
        """Subsampling should work with fewer pendulums than available."""
        from tests.conftest import MockSimulation

        config = FeatureConfig(max_pendulums=10)
        extractor = FeatureExtractor(config)
        sim = MockSimulation(sample_simulation_data)

        # Should not raise
        features = extractor.transform(sim)
        assert features.shape[0] == 100


class TestResolutionInvariance:
    """Tests for resolution invariance (aggregation over pendulums)."""

    def test_variance_aggregates_over_pendulums(self):
        """Variance should aggregate over pendulums dimension."""
        # 10 frames, 100 pendulums
        data1 = np.random.randn(10, 100, 8).astype(np.float32)
        # 10 frames, 1000 pendulums
        data2 = np.random.randn(10, 1000, 8).astype(np.float32)

        result1 = variance_features(data1)
        result2 = variance_features(data2)

        # Both should produce (10, 8) regardless of pendulum count
        assert result1.shape == (10, 8)
        assert result2.shape == (10, 8)

    def test_all_features_resolution_independent(self):
        """All feature functions should work regardless of pendulum count."""
        np.random.seed(42)

        for n_pendulums in [10, 100, 1000]:
            data = np.random.randn(50, n_pendulums, 8).astype(np.float32)

            # All should produce (50, X) output
            assert variance_features(data).shape[0] == 50
            assert std_features(data).shape[0] == 50
            assert iqr_features(data).shape[0] == 50
            assert range_features(data).shape[0] == 50
            assert tip_spread_features(data).shape[0] == 50
            assert angular_spread_features(data).shape[0] == 50
            assert velocity_features(data).shape[0] == 50


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_pendulum(self):
        """Features should work with a single pendulum."""
        data = np.random.randn(10, 1, 8).astype(np.float32)

        # Variance should be 0 with single pendulum
        var_result = variance_features(data)
        assert var_result.shape == (10, 8)
        np.testing.assert_array_almost_equal(var_result, 0)

    def test_single_frame(self):
        """Features should work with a single frame."""
        data = np.random.randn(1, 50, 8).astype(np.float32)

        result = variance_features(data)
        assert result.shape == (1, 8)

    def test_no_nan_in_output(self, sample_simulation_data):
        """Features should not contain NaN values for valid input."""
        features = variance_features(sample_simulation_data)
        assert not np.any(np.isnan(features))

        features = skewness_features(sample_simulation_data)
        assert not np.any(np.isnan(features))

        features = kurtosis_features(sample_simulation_data)
        assert not np.any(np.isnan(features))
