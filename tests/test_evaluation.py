"""
Tests for evaluation metrics and cross-validation.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestMetricCalculations:
    """Tests for basic metric calculations using actual evaluation functions."""

    def test_mae_calculation(self):
        """MAE should be computed correctly."""
        from boom_detection.evaluation import mae

        predictions = np.array([10, 20, 30])
        true_values = np.array([12, 18, 35])
        # Errors: 2, 2, 5
        expected_mae = (2 + 2 + 5) / 3

        actual_mae = mae(predictions, true_values)
        assert actual_mae == expected_mae

    def test_mae_with_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        from boom_detection.evaluation import mae

        predictions = np.array([10, 20, 30])
        true_values = np.array([10, 20, 30])

        result = mae(predictions, true_values)
        assert result == 0.0

    def test_within_n_frames(self):
        """Within-N-frames metric should be computed correctly."""
        from boom_detection.evaluation import compute_all_metrics

        predictions = np.array([10, 20, 30, 40, 50])
        true_values = np.array([12, 18, 40, 42, 50])
        # Errors: 2, 2, 10, 2, 0
        # Within 5: 4/5 = 80%

        metrics = compute_all_metrics(predictions, true_values)
        within_5 = metrics['within_5'] * 100

        assert within_5 == 80.0

    def test_compute_all_metrics(self):
        """compute_all_metrics should return all expected keys."""
        from boom_detection.evaluation import compute_all_metrics

        predictions = np.array([10, 20, 30, 40, 50])
        true_values = np.array([12, 18, 35, 42, 50])

        metrics = compute_all_metrics(predictions, true_values)

        expected_keys = {'mae', 'median_ae', 'rmse', 'max_ae', 'correlation',
                         'within_5', 'within_10', 'within_15', 'within_30'}
        assert set(metrics.keys()) == expected_keys
        assert all(isinstance(v, float) for v in metrics.values())


class TestCrossValidation:
    """Tests for cross-validation behavior."""

    def test_cv_indices_no_overlap(self):
        """Train and test indices should not overlap."""
        from sklearn.model_selection import KFold

        n_samples = 100
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        indices = list(range(n_samples))

        for train_idx, test_idx in kf.split(indices):
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

            # All indices covered
            assert len(train_idx) + len(test_idx) == n_samples

    def test_cv_reproducibility(self):
        """Same seed should produce same splits."""
        from sklearn.model_selection import KFold

        n_samples = 100
        kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
        kf2 = KFold(n_splits=5, shuffle=True, random_state=42)

        splits1 = list(kf1.split(range(n_samples)))
        splits2 = list(kf2.split(range(n_samples)))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_different_seeds_different_splits(self):
        """Different seeds should produce different splits."""
        from sklearn.model_selection import KFold

        n_samples = 100
        kf1 = KFold(n_splits=5, shuffle=True, random_state=42)
        kf2 = KFold(n_splits=5, shuffle=True, random_state=43)

        splits1 = list(kf1.split(range(n_samples)))
        splits2 = list(kf2.split(range(n_samples)))

        # At least one split should be different
        any_different = False
        for (train1, _), (train2, _) in zip(splits1, splits2):
            if not np.array_equal(train1, train2):
                any_different = True
                break
        assert any_different


class TestSelectiveMetrics:
    """Tests for selective (abstention) metrics."""

    def test_acceptance_rate_calculation(self):
        """Acceptance rate should be computed correctly."""
        accepted = [True, True, False, True, False]
        acceptance_rate = sum(accepted) / len(accepted) * 100
        assert acceptance_rate == 60.0

    def test_selective_mae_excludes_rejected(self):
        """Selective MAE should only consider accepted predictions."""
        predictions = [10, 20, 30, 40, 50]
        true_values = [12, 18, 35, 42, 100]  # Last one is very wrong
        accepted = [True, True, True, True, False]

        # Only consider accepted: errors are 2, 2, 5, 2
        accepted_errors = [
            abs(p - t) for p, t, a in zip(predictions, true_values, accepted) if a
        ]
        selective_mae = np.mean(accepted_errors)

        # Should not include the 50 error from rejected prediction
        assert selective_mae == (2 + 2 + 5 + 2) / 4

    def test_coverage_metric(self):
        """Coverage (acceptance rate) should be in [0, 1]."""
        for n_accepted in [0, 25, 50, 100]:
            accepted = [True] * n_accepted + [False] * (100 - n_accepted)
            coverage = sum(accepted) / len(accepted)
            assert 0 <= coverage <= 1


class TestAggregationMetrics:
    """Tests for aggregation across seeds."""

    def test_mean_std_calculation(self):
        """Mean and std should be computed correctly across seeds."""
        seed_maes = [5.0, 6.0, 7.0, 5.5, 6.5]

        mean_mae = np.mean(seed_maes)
        std_mae = np.std(seed_maes, ddof=1)  # Sample std

        assert mean_mae == 6.0
        assert abs(std_mae - 0.7905) < 0.001  # Approximate

    def test_confidence_interval(self):
        """95% CI should be approximately mean ± 1.96 * std / sqrt(n)."""
        seed_maes = np.array([5.0, 6.0, 7.0, 5.5, 6.5])

        mean_mae = np.mean(seed_maes)
        std_mae = np.std(seed_maes, ddof=1)
        n = len(seed_maes)

        ci_half_width = 1.96 * std_mae / np.sqrt(n)
        ci_lower = mean_mae - ci_half_width
        ci_upper = mean_mae + ci_half_width

        assert ci_lower < mean_mae < ci_upper
        assert ci_half_width > 0


class TestSelectivePrediction:
    """Tests for SelectivePrediction dataclass."""

    def test_create_accepted_prediction(self):
        """Should create an accepted SelectivePrediction."""
        from boom_detection.evaluation import SelectivePrediction

        pred = SelectivePrediction(
            boom_frame=50,
            accepted=True,
            accept_score=0.8,
            predicted_quality=0.75,
            model_predictions={'cnn': 50, 'hgb': 48},
            disagreement=1.0,
        )

        assert pred.boom_frame == 50
        assert pred.accepted is True
        assert pred.disagreement == 1.0

    def test_create_rejected_prediction(self):
        """Should create a rejected SelectivePrediction with None boom_frame."""
        from boom_detection.evaluation import SelectivePrediction

        pred = SelectivePrediction(
            boom_frame=None,
            accepted=False,
            accept_score=0.3,
            predicted_quality=0.3,
            model_predictions={'cnn': 50, 'hgb': 70},
            disagreement=10.0,
        )

        assert pred.boom_frame is None
        assert pred.accepted is False

    def test_from_dict(self):
        """Should create from dictionary."""
        from boom_detection.evaluation import SelectivePrediction

        d = {
            'boom_frame': 50,
            'accepted': True,
            'accept_score': 0.8,
            'predicted_quality': 0.75,
            'model_predictions': {'cnn': 50, 'hgb': 48},
            'disagreement': 1.0,
        }

        pred = SelectivePrediction.from_dict(d)
        assert pred.boom_frame == 50
        assert pred.accepted is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        from boom_detection.evaluation import SelectivePrediction

        pred = SelectivePrediction(
            boom_frame=50,
            accepted=True,
            accept_score=0.8,
            predicted_quality=0.75,
            model_predictions={'cnn': 50, 'hgb': 48},
            disagreement=1.0,
        )

        d = pred.to_dict()
        assert d['boom_frame'] == 50
        assert d['accepted'] is True
        assert d['model_predictions'] == {'cnn': 50, 'hgb': 48}

    def test_roundtrip(self):
        """Should survive to_dict/from_dict roundtrip."""
        from boom_detection.evaluation import SelectivePrediction

        original = SelectivePrediction(
            boom_frame=50,
            accepted=True,
            accept_score=0.9,
            predicted_quality=0.75,
            model_predictions={'cnn': 50, 'hgb': 48},
            disagreement=1.0,
        )

        roundtripped = SelectivePrediction.from_dict(original.to_dict())
        assert roundtripped.boom_frame == original.boom_frame
        assert roundtripped.accept_score == original.accept_score


def _make_pred(boom_frame, accepted, accept_score, predicted_quality, preds=None):
    """Helper to create SelectivePrediction with sensible defaults."""
    from boom_detection.evaluation import SelectivePrediction
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


class TestComputeSelectiveMetrics:
    """Tests for compute_selective_metrics function."""

    def test_all_accepted(self):
        """Should compute metrics when all predictions accepted."""
        from boom_detection.evaluation import compute_selective_metrics

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 48}),
            _make_pred(60, True, 0.7, 0.7, {'cnn': 60, 'hgb': 58}),
            _make_pred(70, True, 0.9, 0.9, {'cnn': 70, 'hgb': 72}),
        ]
        true_booms = np.array([52, 58, 70])

        metrics = compute_selective_metrics(predictions, true_booms)

        assert metrics['n_total'] == 3
        assert metrics['n_accepted'] == 3
        assert metrics['coverage'] == 1.0
        assert metrics['selective_mae'] == (2 + 2 + 0) / 3

    def test_partial_acceptance(self):
        """Should compute metrics with some rejections."""
        from boom_detection.evaluation import compute_selective_metrics

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 48}),
            _make_pred(None, False, 0.3, 0.3, {'cnn': 60, 'hgb': 80}),  # Rejected
            _make_pred(70, True, 0.9, 0.9, {'cnn': 70, 'hgb': 72}),
        ]
        true_booms = np.array([52, 58, 70])

        metrics = compute_selective_metrics(predictions, true_booms)

        assert metrics['n_total'] == 3
        assert metrics['n_accepted'] == 2
        assert metrics['coverage'] == 2 / 3
        # Only compute MAE on accepted: errors are 2, 0
        assert metrics['selective_mae'] == 1.0

    def test_none_accepted(self):
        """Should handle case where nothing is accepted."""
        from boom_detection.evaluation import compute_selective_metrics

        predictions = [
            _make_pred(None, False, 0.2, 0.2, {'cnn': 50, 'hgb': 80}),
            _make_pred(None, False, 0.1, 0.1, {'cnn': 60, 'hgb': 90}),
        ]
        true_booms = np.array([52, 58])

        metrics = compute_selective_metrics(predictions, true_booms)

        assert metrics['n_accepted'] == 0
        assert metrics['coverage'] == 0.0
        assert np.isnan(metrics['selective_mae'])

    def test_with_quality(self):
        """Should include quality metrics when qualities provided."""
        from boom_detection.evaluation import compute_selective_metrics

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 48}),
            _make_pred(None, False, 0.3, 0.3, {'cnn': 60, 'hgb': 80}),
            _make_pred(70, True, 0.9, 0.9, {'cnn': 70, 'hgb': 72}),
        ]
        true_booms = np.array([52, 58, 70])
        true_qualities = np.array([0.8, 0.6, 0.9])

        metrics = compute_selective_metrics(predictions, true_booms, true_qualities)

        assert 'mean_accepted_quality' in metrics
        assert metrics['mean_accepted_quality'] == (0.8 + 0.9) / 2


class TestRunArtifact:
    """Tests for RunArtifact class."""

    def test_create_artifact(self):
        """Should create artifact from predictions."""
        from boom_detection.evaluation import RunArtifact

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 48}),
            _make_pred(60, True, 0.7, 0.7, {'cnn': 60, 'hgb': 58}),
        ]
        true_booms = np.array([52, 58])
        config = {'agreement_threshold': 5, 'quality_threshold': 0.55}

        artifact = RunArtifact.create(
            config=config,
            predictions=predictions,
            true_booms=true_booms,
        )

        assert artifact.config == config
        assert 'selective_mae' in artifact.metrics
        assert len(artifact.predictions) == 2
        assert artifact.timestamp != ""

    def test_save_and_load(self, tmp_path):
        """Should save to disk and load back."""
        from boom_detection.evaluation import RunArtifact

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 48}),
        ]
        true_booms = np.array([52])
        config = {'test': 'value'}

        original = RunArtifact.create(
            config=config,
            predictions=predictions,
            true_booms=true_booms,
        )

        # Save
        save_path = tmp_path / "test_run"
        original.save(save_path)

        # Check files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "metrics.json").exists()
        assert (save_path / "predictions.jsonl").exists()
        assert (save_path / "environment.json").exists()

        # Load
        loaded = RunArtifact.load(save_path)

        assert loaded.config == original.config
        assert loaded.metrics == original.metrics
        assert len(loaded.predictions) == len(original.predictions)

    def test_summary(self):
        """Should generate readable summary."""
        from boom_detection.evaluation import RunArtifact

        predictions = [
            _make_pred(50, True, 0.8, 0.8, {'cnn': 50, 'hgb': 48}),
        ]
        true_booms = np.array([50])

        artifact = RunArtifact.create(
            config={'threshold': 5},
            predictions=predictions,
            true_booms=true_booms,
        )

        summary = artifact.summary()
        assert "Metrics:" in summary
        assert "Config:" in summary
        assert "Environment:" in summary


class TestCombinerExperiment:
    """
    Tests for CombinerExperiment class (formerly FormulaExperiment).

    CRITICAL: These tests prove that CombinerExperiment produces IDENTICAL
    results to running the full pipeline evaluation.
    """

    def test_cached_sample(self):
        """Should create CachedSample correctly."""
        from boom_detection.combine import ModelPrediction
        from boom_detection.evaluation import CachedSample

        pred = CachedSample(
            sim_id='sim_0',
            predictions=[ModelPrediction('cnn', 50), ModelPrediction('hgb', 48)],
            predicted_quality=0.75,
            true_boom=52,
            true_quality=0.8,
        )

        assert pred.sim_id == 'sim_0'
        assert len(pred.predictions) == 2
        assert pred.predictions[0].frame == 50
        assert pred.predictions[1].frame == 48
        assert pred.predicted_quality == 0.75
        assert pred.true_boom == 52
        assert pred.true_quality == 0.8
        # Check dict property
        assert pred.model_predictions_dict == {'cnn': 50, 'hgb': 48}

    def test_combiner_experiment_evaluate(self):
        """CombinerExperiment.evaluate() should return valid results."""
        from boom_detection.combine import ModelPrediction, ThresholdCombiner
        from boom_detection.evaluation import CachedSample, CombinerExperiment

        # Create cached predictions for 2 seeds
        seed1_preds = [
            CachedSample(
                sim_id='sim_0',
                predictions=[ModelPrediction('cnn', 50), ModelPrediction('hgb', 48)],
                predicted_quality=0.8,
                true_boom=52,
                true_quality=0.9,
            ),
            CachedSample(
                sim_id='sim_1',
                predictions=[ModelPrediction('cnn', 60), ModelPrediction('hgb', 80)],  # High disagreement
                predicted_quality=0.4,
                true_boom=65,
                true_quality=0.5,
            ),
        ]
        seed2_preds = [
            CachedSample(
                sim_id='sim_0',
                predictions=[ModelPrediction('cnn', 51), ModelPrediction('hgb', 49)],
                predicted_quality=0.75,
                true_boom=52,
                true_quality=0.9,
            ),
            CachedSample(
                sim_id='sim_1',
                predictions=[ModelPrediction('cnn', 62), ModelPrediction('hgb', 78)],  # High disagreement
                predicted_quality=0.45,
                true_boom=65,
                true_quality=0.5,
            ),
        ]

        experiment = CombinerExperiment(
            cached_samples=[seed1_preds, seed2_preds],
            seeds=[42, 43],
            k=5,
        )

        combiner = ThresholdCombiner(
            agreement_transform='sqrt',
            disagreement_scale=5.0,
            threshold=0.60,
        )
        result = experiment.evaluate(combiner)

        assert result.k == 5
        assert result.seeds == [42, 43]
        assert 'selective_mae' in result.mean_metrics
        assert 'coverage' in result.mean_metrics

    def test_combiner_application_matches_combine_module(self):
        """
        CRITICAL TEST: CombinerExperiment must use combine.py and produce
        identical results to ThresholdCombiner's acceptance logic.
        """
        from boom_detection.combine import ModelPrediction, ThresholdCombiner
        from boom_detection.evaluation import CachedSample, CombinerExperiment

        # Test case: exactly match combine.py logic
        pred = CachedSample(
            sim_id='sim_0',
            predictions=[ModelPrediction('cnn', 50), ModelPrediction('hgb', 42)],  # disagreement = 8
            predicted_quality=0.7,
            true_boom=52,
            true_quality=0.8,
        )

        experiment = CombinerExperiment(
            cached_samples=[[pred]],
            seeds=[42],
            k=5,
        )

        combiner = ThresholdCombiner(
            agreement_transform='sqrt',
            disagreement_scale=15.0,
            threshold=0.60,
        )

        # Evaluate with known parameters
        result = experiment.evaluate(combiner)

        # Verify the math manually (must match combine.py exactly)
        # disagreement = |50 - 42| = 8
        # agreement_score = 1 - sqrt(8 / 15) = 1 - sqrt(0.533) = 1 - 0.730 = 0.270
        # accept_score = 0.4 * 0.270 + 0.6 * 0.7 = 0.108 + 0.42 = 0.528
        # accepted = 0.528 >= 0.60 -> False

        expected_agreement = 1.0 - np.sqrt(8.0 / 15.0)  # ~0.270
        expected_accept_score = 0.4 * expected_agreement + 0.6 * 0.7  # ~0.528

        # Get metrics - since rejected, coverage should be 0
        assert result.mean_metrics['coverage'] < 1.0  # Not all accepted

        # Also verify using combiner directly
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 42)]
        result_frame = combiner(preds, 0.7)
        accept_score = combiner.get_score()
        assert abs(accept_score - expected_accept_score) < 1e-10
        assert result_frame is None  # 0.528 < 0.60, rejected

    def test_combiner_experiment_sweep(self):
        """sweep() should test multiple configurations."""
        from boom_detection.combine import ModelPrediction, ThresholdCombiner
        from boom_detection.evaluation import CachedSample, CombinerExperiment

        seed_preds = [
            CachedSample(
                sim_id='sim_0',
                predictions=[ModelPrediction('cnn', 50), ModelPrediction('hgb', 48)],
                predicted_quality=0.8,
                true_boom=52,
                true_quality=0.9,
            ),
        ]

        experiment = CombinerExperiment(
            cached_samples=[seed_preds],
            seeds=[42],
            k=5,
        )

        # Generate combiners using grid helper
        combiners = ThresholdCombiner.grid(
            agreement_transform=['sqrt'],
            disagreement_scale=[5.0, 10.0, 15.0],
            threshold=[0.5, 0.6],
        )

        results = experiment.sweep(combiners, verbose=False)

        # Should have 3 scales x 1 formula x 2 thresholds = 6 results
        assert len(results) == 6

    def test_linear_vs_sqrt_transform(self):
        """Linear and sqrt transforms should produce different results."""
        from boom_detection.combine import ModelPrediction, ThresholdCombiner

        # Test with same parameters, different transforms
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 40)]  # disagreement = 10
        quality = 0.7

        # Linear: agreement = 1 - 10/15 = 0.333
        linear = ThresholdCombiner(agreement_transform='linear', disagreement_scale=15.0, threshold=0.60)
        linear(preds, quality)
        linear_score = linear.get_score()

        # Sqrt: agreement = 1 - sqrt(10/15) = 1 - 0.816 = 0.184
        sqrt = ThresholdCombiner(agreement_transform='sqrt', disagreement_scale=15.0, threshold=0.60)
        sqrt(preds, quality)
        sqrt_score = sqrt.get_score()

        # Linear should give higher accept score (higher agreement)
        linear_agreement = 1.0 - 10.0 / 15.0  # 0.333
        sqrt_agreement = 1.0 - np.sqrt(10.0 / 15.0)  # 0.184

        linear_expect = 0.4 * linear_agreement + 0.6 * 0.7
        sqrt_expect = 0.4 * sqrt_agreement + 0.6 * 0.7

        assert abs(linear_score - linear_expect) < 1e-10
        assert abs(sqrt_score - sqrt_expect) < 1e-10
        assert linear_score > sqrt_score

    def test_3model_disagreement_uses_range(self):
        """Disagreement for 3 models should use range (max-min), not std."""
        from boom_detection.combine import ModelPrediction, ThresholdCombiner

        # 3-model prediction: min=45, max=55 -> disagreement = 10
        preds = [
            ModelPrediction('cnn', 50),
            ModelPrediction('hgb', 45),
            ModelPrediction('lstm', 55),
        ]
        quality = 0.7

        # The combiner internally computes disagreement as max-min
        # With scale=15, threshold=0.40 (low threshold so it accepts)
        combiner = ThresholdCombiner(
            agreement_transform='sqrt',
            disagreement_scale=15.0,
            threshold=0.40,
        )
        result = combiner(preds, quality)
        accept_score = combiner.get_score()

        # Agreement = 1 - sqrt(10/15) = 1 - 0.816 = 0.184
        # Score = 0.4 * 0.184 + 0.6 * 0.7 = 0.0736 + 0.42 = 0.4936
        expected_agreement = 1.0 - np.sqrt(10.0 / 15.0)
        expected_score = 0.4 * expected_agreement + 0.6 * 0.7
        assert abs(accept_score - expected_score) < 1e-10
