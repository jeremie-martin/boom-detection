"""
Smoke tests for boom detection pipeline.

These are high-leverage integration tests that verify end-to-end functionality.
They catch issues that unit tests might miss, like save/load mismatches.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def multi_sim_data(self):
        """Create multiple simulations with features for training."""
        np.random.seed(42)
        n_sims = 10
        n_frames = 100
        n_features = 183

        features = {}
        boom_frames = []
        qualities = []
        sim_ids = []

        for i in range(n_sims):
            sim_id = f"sim_{i}"
            sim_ids.append(sim_id)
            # Features with slight variation
            features[sim_id] = np.random.randn(n_frames, n_features).astype(np.float32)
            boom_frames.append(np.random.randint(30, 70))
            qualities.append(np.random.uniform(0.3, 0.9))

        class MockCache:
            def __init__(self, feats):
                self._cache = feats

            def __getitem__(self, key):
                return self._cache[key]

            def __contains__(self, key):
                return key in self._cache

        return {
            'sim_ids': sim_ids,
            'boom_frames': np.array(boom_frames),
            'qualities': np.array(qualities),
            'cache': MockCache(features),
        }

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch required"
    )
    def test_train_predict_save_load_roundtrip(self, multi_sim_data):
        """Full pipeline: train → predict → save → load → predict again."""
        from boom_detection.combine import ThresholdCombiner
        from boom_detection.deploy_pipeline import BoomDetectionPipeline

        # Train pipeline
        pipeline = BoomDetectionPipeline(
            combiner=ThresholdCombiner(threshold=0.4),
            seed=42,
        )
        pipeline.fit(
            multi_sim_data['sim_ids'],
            multi_sim_data['boom_frames'],
            multi_sim_data['qualities'],
            multi_sim_data['cache'],
        )

        # Get predictions
        preds_before = pipeline.predict(
            multi_sim_data['sim_ids'],
            multi_sim_data['cache'],
        )

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "pipeline"
            pipeline.save(save_path)

            # Load pipeline
            loaded = BoomDetectionPipeline.from_pretrained(save_path)

            # Get predictions from loaded pipeline
            preds_after = loaded.predict(
                multi_sim_data['sim_ids'],
                multi_sim_data['cache'],
            )

        # Verify predictions match
        for before, after in zip(preds_before, preds_after):
            assert before.model_predictions == after.model_predictions, "Model predictions should match"
            assert before.accepted == after.accepted, "Acceptance should match"
            assert abs(before.predicted_quality - after.predicted_quality) < 0.01, \
                "Quality predictions should match"

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch required"
    )
    def test_cnn_architecture_roundtrip(self):
        """Verify CNN architecture is preserved across save/load."""
        import torch
        from boom_detection.sequence_models import CNNClassifier

        # Create CNN with specific architecture
        original = CNNClassifier(
            n_features=100,
            hidden_dim=64,
            kernel_sizes=(5, 11, 21),
            dropout=0.25,
        )

        # Verify attributes are stored
        assert original.n_features == 100
        assert original.hidden_dim == 64
        assert original.kernel_sizes == (5, 11, 21)
        assert original.dropout == 0.25

        # Save and load state dict with architecture
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            checkpoint = {
                'state_dict': original.state_dict(),
                'n_features': original.n_features,
                'hidden_dim': original.hidden_dim,
                'kernel_sizes': original.kernel_sizes,
                'dropout': original.dropout,
            }
            torch.save(checkpoint, f.name)

            # Load
            loaded_checkpoint = torch.load(f.name, weights_only=True)
            loaded = CNNClassifier(
                n_features=loaded_checkpoint['n_features'],
                hidden_dim=loaded_checkpoint['hidden_dim'],
                kernel_sizes=tuple(loaded_checkpoint['kernel_sizes']),
                dropout=loaded_checkpoint['dropout'],
            )
            loaded.load_state_dict(loaded_checkpoint['state_dict'])

        # Verify architecture matches
        assert loaded.n_features == original.n_features
        assert loaded.hidden_dim == original.hidden_dim
        assert loaded.kernel_sizes == original.kernel_sizes
        assert loaded.dropout == original.dropout

        # Verify predictions match
        original.eval()
        loaded.eval()
        test_input = torch.randn(1, 50, 100)
        with torch.no_grad():
            orig_out = original(test_input)
            loaded_out = loaded(test_input)
        assert torch.allclose(orig_out, loaded_out), "Outputs should match after load"


class TestFeatureCacheConsistency:
    """Tests for feature cache consistency."""

    def test_feature_extractor_reproducibility(self, sample_simulation_data, mock_simulation):
        """Same data should produce identical features."""
        from boom_detection.features import FeatureExtractor, FeatureConfig

        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        # Extract twice using transform
        features1 = extractor.transform(mock_simulation)
        features2 = extractor.transform(mock_simulation)

        assert np.allclose(features1, features2), "Features should be deterministic"

    def test_feature_names_match_count(self, sample_simulation_data, mock_simulation):
        """Feature names count should match actual feature count."""
        from boom_detection.features import FeatureExtractor, FeatureConfig

        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        features = extractor.transform(mock_simulation)
        names = extractor.feature_names

        assert len(names) == features.shape[1], \
            f"Expected {features.shape[1]} names, got {len(names)}"

    def test_feature_index_resolver(self, sample_simulation_data):
        """Feature index resolver should find features correctly."""
        from boom_detection.features import FeatureExtractor, FeatureConfig, FeatureCache

        config = FeatureConfig()

        # Create a minimal cache
        class MinimalCache(FeatureCache):
            def __init__(self, cfg):
                self.extractor = FeatureExtractor(cfg)
                self._name_to_idx = None

        cache = MinimalCache(config)

        # Get some feature names
        names = cache.feature_names
        assert len(names) > 0

        # Test index lookup
        for i, name in enumerate(names[:5]):  # Test first 5
            idx = cache.index(name)
            assert idx == i, f"Expected index {i} for {name}, got {idx}"

        # Test indices lookup
        first_three = names[:3]
        indices = cache.indices(*first_three)
        assert indices == [0, 1, 2]

        # Test unknown feature
        with pytest.raises(KeyError, match="not found"):
            cache.index("nonexistent_feature")


class TestSelectivePredictionSerialization:
    """Tests for SelectivePrediction serialization."""

    def test_roundtrip(self):
        """Verify SelectivePrediction survives serialization roundtrip."""
        from boom_detection.metrics import SelectivePrediction

        original = SelectivePrediction(
            boom_frame=50,
            accepted=True,
            accept_score=0.68,
            predicted_quality=0.75,
            model_predictions={'cnn': 50, 'hgb': 48},
            disagreement=1.0,  # std of [50, 48]
        )

        # Convert to dict and back
        d = original.to_dict()
        loaded = SelectivePrediction.from_dict(d)

        # Verify all fields match
        assert loaded.boom_frame == original.boom_frame
        assert loaded.accepted == original.accepted
        assert loaded.model_predictions == original.model_predictions
        assert loaded.disagreement == original.disagreement
        assert loaded.predicted_quality == original.predicted_quality
        assert loaded.accept_score == original.accept_score

    def test_rejected_prediction_roundtrip(self):
        """Verify rejected predictions serialize correctly."""
        from boom_detection.metrics import SelectivePrediction

        original = SelectivePrediction(
            boom_frame=None,  # Rejected
            accepted=False,
            accept_score=0.35,
            predicted_quality=0.3,
            model_predictions={'cnn': 50, 'hgb': 65},
            disagreement=7.5,  # std of [50, 65]
        )

        d = original.to_dict()
        loaded = SelectivePrediction.from_dict(d)

        assert loaded.boom_frame is None
        assert loaded.accepted is False
        assert loaded.accept_score == 0.35


def _make_pred(boom_frame, accepted, accept_score, predicted_quality, preds=None):
    """Helper to create SelectivePrediction with sensible defaults."""
    from boom_detection.metrics import SelectivePrediction
    if preds is None:
        preds = {'cnn': boom_frame or 50, 'hgb': boom_frame or 50}
    return SelectivePrediction(
        boom_frame=boom_frame,
        accepted=accepted,
        accept_score=accept_score,
        predicted_quality=predicted_quality,
        model_predictions=preds,
        disagreement=float(max(preds.values()) - min(preds.values())) if len(preds) > 1 else 0.0,
    )


class TestMetricsConsistency:
    """Tests for metrics calculation consistency."""

    def test_selective_metrics_coverage(self):
        """Coverage should match n_accepted / n_total."""
        from boom_detection.metrics import compute_selective_metrics

        predictions = [
            _make_pred(50, True, 0.9, 0.9, {'cnn': 50, 'hgb': 50}),
            _make_pred(60, True, 0.8, 0.8, {'cnn': 60, 'hgb': 62}),
            _make_pred(None, False, 0.3, 0.3, {'cnn': 70, 'hgb': 85}),
            _make_pred(None, False, 0.2, 0.2, {'cnn': 80, 'hgb': 95}),
        ]
        true_booms = np.array([51, 61, 75, 85])

        metrics = compute_selective_metrics(predictions, true_booms)

        assert metrics['n_total'] == 4
        assert metrics['n_accepted'] == 2
        assert metrics['coverage'] == 0.5

    def test_decision_centric_metrics(self):
        """Test coverage_at_max_mae and min_mae_at_coverage."""
        from boom_detection.metrics import (
            coverage_at_max_mae,
            min_mae_at_coverage,
        )

        # Create predictions with varying quality
        # Sorted by accept_score: 0.95, 0.85, 0.65, 0.35
        predictions = [
            _make_pred(50, True, 0.95, 0.9, {'cnn': 50, 'hgb': 50}),  # Error: 1
            _make_pred(60, True, 0.85, 0.8, {'cnn': 60, 'hgb': 60}),  # Error: 2
            _make_pred(70, True, 0.65, 0.6, {'cnn': 70, 'hgb': 70}),  # Error: 5
            _make_pred(80, True, 0.35, 0.3, {'cnn': 80, 'hgb': 80}),  # Error: 10
        ]
        true_booms = np.array([51, 62, 75, 90])

        # Test coverage_at_max_mae
        # At k=1: mae=1.0 <= 3.0 ✓
        # At k=2: mae=(1+2)/2=1.5 <= 3.0 ✓
        # At k=3: mae=(1+2+5)/3=2.67 <= 3.0 ✓
        # At k=4: mae=(1+2+5+10)/4=4.5 > 3.0 ✗
        # So max coverage at mae <= 3.0 is 3/4 = 0.75
        cov = coverage_at_max_mae(predictions, true_booms, max_mae=3.0)
        assert cov == 0.75, f"Expected 0.75, got {cov}"

        # Test min_mae_at_coverage
        # At 50% coverage (2 samples), sorted by score: errors [1, 2]
        # mae = (1 + 2) / 2 = 1.5
        mae = min_mae_at_coverage(predictions, true_booms, target_coverage=0.5)
        assert mae == 1.5, f"Expected 1.5, got {mae}"


class TestRunArtifactRoundtrip:
    """Tests for RunArtifact save/load."""

    def test_save_and_load(self):
        """RunArtifact should survive save/load roundtrip."""
        import tempfile
        from pathlib import Path
        from boom_detection.metrics import RunArtifact

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 50}),
            _make_pred(None, False, 0.3, 0.3, {'cnn': 60, 'hgb': 75}),
        ]
        true_booms = np.array([51, 65])
        true_qualities = np.array([0.9, 0.4])

        config = {'accept_threshold': 0.5, 'test_param': 'value'}

        artifact = RunArtifact.create(
            config=config,
            predictions=predictions,
            true_booms=true_booms,
            true_qualities=true_qualities,
            sim_ids=['sim_0', 'sim_1'],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_run'
            artifact.save(path)

            # Verify files exist
            assert (path / 'config.json').exists()
            assert (path / 'metrics.json').exists()
            assert (path / 'predictions.jsonl').exists()
            assert (path / 'environment.json').exists()

            # Load and verify
            loaded = RunArtifact.load(path)

            assert loaded.config == config
            assert loaded.metrics['coverage'] == 0.5
            assert len(loaded.predictions) == 2
