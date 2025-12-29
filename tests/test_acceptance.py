"""
Tests for combiner functions (formerly acceptance functions).
"""
from __future__ import annotations

import numpy as np
import pytest

from boom_detection.combine import (
    ModelPrediction,
    ThresholdCombiner,
    MedianCombiner,
    NamedModelCombiner,
    QualityGatedCombiner,
    AgreementGatedCombiner,
    MajorityVoteCombiner,
    COMBINER_REGISTRY,
    combiner_to_config,
    combiner_from_config,
    default_combiner,
    frames,
    get_prediction,
    median_frame,
    mean_frame,
    disagreement,
    agreement_score,
)


class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""

    def test_basic_creation(self):
        pred = ModelPrediction('cnn', 100)
        assert pred.model == 'cnn'
        assert pred.frame == 100
        assert pred.confidence is None

    def test_with_confidence(self):
        pred = ModelPrediction('hgb', 105, confidence=0.9)
        assert pred.confidence == 0.9


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_frames(self):
        preds = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 110)]
        f = frames(preds)
        assert np.array_equal(f, np.array([100, 110]))

    def test_get_prediction(self):
        preds = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 110)]
        assert get_prediction(preds, 'cnn').frame == 100
        assert get_prediction(preds, 'hgb').frame == 110

        with pytest.raises(KeyError):
            get_prediction(preds, 'lstm')

    def test_median_frame(self):
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 110), ModelPrediction('c', 120)]
        assert median_frame(preds) == 110

    def test_mean_frame(self):
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 110)]
        assert mean_frame(preds) == 105

    def test_disagreement_range(self):
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 120)]
        assert disagreement(preds, 'range') == 20.0

    def test_disagreement_std(self):
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 110)]
        assert abs(disagreement(preds, 'std') - 5.0) < 0.01

    def test_agreement_score_linear(self):
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 110)]  # disagreement = 10
        score = agreement_score(preds, scale=20.0, transform='linear')
        # normalized = 10/20 = 0.5, agreement = 1 - 0.5 = 0.5
        assert abs(score - 0.5) < 0.01

    def test_agreement_score_sqrt(self):
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 100)]  # perfect agreement
        score = agreement_score(preds, scale=10.0, transform='sqrt')
        assert score == 1.0


class TestThresholdCombiner:
    """Tests for ThresholdCombiner."""

    def test_perfect_agreement_high_quality_accepts(self):
        """Perfect agreement + high quality should accept."""
        combiner = ThresholdCombiner(threshold=0.60)
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 50)]

        result = combiner(preds, predicted_quality=0.8)

        # agreement = 1.0, quality = 0.8
        # score = 0.4 * 1.0 + 0.6 * 0.8 = 0.88 >= 0.60
        assert result == 50
        assert abs(combiner.get_score() - 0.88) < 1e-10

    def test_high_disagreement_low_quality_rejects(self):
        """High disagreement + low quality should reject."""
        combiner = ThresholdCombiner(disagreement_scale=15.0, threshold=0.60)
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 70)]

        result = combiner(preds, predicted_quality=0.3)

        # disagreement = 20, capped at scale=15 -> agreement = 1 - sqrt(1) = 0
        # score = 0.4 * 0 + 0.6 * 0.3 = 0.18 < 0.60
        assert result is None
        assert combiner.get_score() < 0.60

    def test_scale_parameter(self):
        """Lower scale should make combiner more selective."""
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 55)]  # disagreement = 5
        quality = 0.7

        # With scale=15: agreement = 1 - sqrt(5/15) = 1 - 0.577 = 0.423
        combiner_permissive = ThresholdCombiner(disagreement_scale=15.0, threshold=0.60)
        combiner_permissive(preds, quality)
        score_permissive = combiner_permissive.get_score()

        # With scale=5: agreement = 1 - sqrt(5/5) = 1 - 1.0 = 0
        combiner_selective = ThresholdCombiner(disagreement_scale=5.0, threshold=0.60)
        combiner_selective(preds, quality)
        score_selective = combiner_selective.get_score()

        assert score_permissive > score_selective

    def test_primary_model_used(self):
        """Should return the primary model's prediction when accepted."""
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 52)]

        combiner_cnn = ThresholdCombiner(primary_model='cnn', threshold=0.50)
        combiner_hgb = ThresholdCombiner(primary_model='hgb', threshold=0.50)

        assert combiner_cnn(preds, 0.8) == 50
        assert combiner_hgb(preds, 0.8) == 52

    def test_documented_result_reproduction(self):
        """Test we can reproduce documented config behavior."""
        # sqrt/scale=5/threshold=0.60 is the documented "most selective" config
        combiner = ThresholdCombiner(
            agreement_transform='sqrt',
            disagreement_scale=5.0,
            threshold=0.60,
        )

        # Perfect agreement + high quality should accept
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 50)]
        assert combiner(preds, 0.9) == 50

        # High disagreement should reject even with high quality
        # disagreement=10 with scale=5 -> agreement = 1 - sqrt(min(10/5, 1)) = 0
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 60)]
        result = combiner(preds, 0.9)
        # score = 0.4 * 0 + 0.6 * 0.9 = 0.54 < 0.60
        assert result is None

    def test_grid_generation(self):
        """Test grid generation for parameter sweeps."""
        combiners = ThresholdCombiner.grid(
            agreement_transform=['sqrt', 'linear'],
            disagreement_scale=[5, 10],
            threshold=[0.60],
        )

        assert len(combiners) == 4
        transforms = {c.agreement_transform for c in combiners}
        scales = {c.disagreement_scale for c in combiners}
        assert transforms == {'sqrt', 'linear'}
        assert scales == {5, 10}

    def test_to_config_from_config(self):
        """Test serialization round-trip."""
        original = ThresholdCombiner(
            primary_model='hgb',
            agreement_transform='sigmoid',
            disagreement_scale=20.0,
            threshold=0.55,
        )

        config = original.to_config()
        restored = ThresholdCombiner.from_config(config)

        assert restored.primary_model == 'hgb'
        assert restored.agreement_transform == 'sigmoid'
        assert restored.disagreement_scale == 20.0
        assert restored.threshold == 0.55


class TestLinearTransform:
    """Tests for linear agreement transform."""

    def test_linear_transition(self):
        """Linear transform should decrease linearly with disagreement."""
        results = []
        for disagreement_val in [0, 2, 4, 6, 8, 10]:
            preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 50 + disagreement_val)]
            combiner = ThresholdCombiner(
                agreement_transform='linear',
                disagreement_scale=10.0,
                threshold=0.0,  # Always accept for testing
            )
            combiner(preds, 0.5)  # Fixed quality
            results.append(combiner.get_score())

        # Should decrease linearly
        diffs = [results[i] - results[i + 1] for i in range(len(results) - 1)]
        # All differences should be approximately equal (linear)
        for d in diffs[1:]:
            assert abs(d - diffs[0]) < 0.01


class TestQuadraticTransform:
    """Tests for quadratic agreement transform."""

    def test_quadratic_more_permissive_near_zero(self):
        """Quadratic should be more permissive than linear for small disagreements."""
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 53)]  # disagreement = 3
        quality = 0.5

        linear = ThresholdCombiner(agreement_transform='linear', disagreement_scale=10.0, threshold=0.0)
        quadratic = ThresholdCombiner(agreement_transform='quadratic', disagreement_scale=10.0, threshold=0.0)

        linear(preds, quality)
        quadratic(preds, quality)

        # For small disagreement relative to scale, quadratic gives higher agreement
        # linear: agreement = 1 - 0.3 = 0.7
        # quadratic: agreement = 1 - 0.3^2 = 1 - 0.09 = 0.91
        assert quadratic.get_score() > linear.get_score()


class TestOtherCombiners:
    """Tests for other combiner implementations."""

    def test_median_combiner(self):
        """MedianCombiner always accepts and uses median."""
        combiner = MedianCombiner()
        preds = [ModelPrediction('a', 100), ModelPrediction('b', 110), ModelPrediction('c', 120)]

        result = combiner(preds, 0.0)  # Quality ignored
        assert result == 110

    def test_named_model_combiner(self):
        """NamedModelCombiner always accepts and uses specific model."""
        combiner = NamedModelCombiner(model='hgb')
        preds = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 110)]

        result = combiner(preds, 0.0)
        assert result == 110

    def test_quality_gated_combiner_accepts(self):
        """QualityGatedCombiner accepts when quality >= threshold."""
        combiner = QualityGatedCombiner(threshold=0.5)
        preds = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 110)]

        assert combiner(preds, 0.6) == 105  # median
        assert combiner(preds, 0.4) is None  # rejected

    def test_agreement_gated_combiner(self):
        """AgreementGatedCombiner accepts when disagreement <= threshold."""
        combiner = AgreementGatedCombiner(max_disagreement=10.0)
        preds_agree = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 105)]
        preds_disagree = [ModelPrediction('cnn', 100), ModelPrediction('hgb', 120)]

        assert combiner(preds_agree, 0.0) == 102  # median, within tolerance
        assert combiner(preds_disagree, 0.0) is None  # too much disagreement

    def test_majority_vote_combiner(self):
        """MajorityVoteCombiner accepts when majority agree."""
        combiner = MajorityVoteCombiner(tolerance=5.0, min_agreement_ratio=0.5)

        # 2 out of 3 agree within tolerance
        preds = [
            ModelPrediction('a', 100),
            ModelPrediction('b', 103),
            ModelPrediction('c', 150),  # outlier
        ]
        result = combiner(preds, 0.0)
        assert result is not None

        # None agree within tolerance
        preds_disagree = [
            ModelPrediction('a', 100),
            ModelPrediction('b', 150),
            ModelPrediction('c', 200),
        ]
        assert combiner(preds_disagree, 0.0) is None


class TestRegistry:
    """Tests for the combiner registry."""

    def test_all_combiners_registered(self):
        """All expected combiners should be in registry."""
        expected = {
            'ThresholdCombiner',
            'MedianCombiner',
            'NamedModelCombiner',
            'QualityGatedCombiner',
            'AgreementGatedCombiner',
            'MajorityVoteCombiner',
        }
        assert set(COMBINER_REGISTRY.keys()) == expected

    def test_combiner_to_config_from_config(self):
        """Test generic serialization functions."""
        combiner = ThresholdCombiner(threshold=0.55)
        config = combiner_to_config(combiner)
        restored = combiner_from_config(config)

        assert isinstance(restored, ThresholdCombiner)
        assert restored.threshold == 0.55

    def test_combiner_from_config_invalid(self):
        """combiner_from_config should raise for invalid types."""
        with pytest.raises(ValueError, match="Unknown combiner type"):
            combiner_from_config({'type': 'InvalidCombiner', 'params': {}})

    def test_default_combiner(self):
        """default_combiner should create standard configuration."""
        combiner = default_combiner()
        assert isinstance(combiner, ThresholdCombiner)
        assert combiner.agreement_transform == 'sqrt'
        assert combiner.disagreement_scale == 15.0
        assert combiner.threshold == 0.60


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model(self):
        """Single model should have zero disagreement."""
        combiner = ThresholdCombiner(threshold=0.60)
        preds = [ModelPrediction('cnn', 50)]
        result = combiner(preds, 0.8)

        # disagreement = 0, agreement = 1.0
        # score = 0.4 * 1.0 + 0.6 * 0.8 = 0.88
        assert result == 50
        assert abs(combiner.get_score() - 0.88) < 1e-10

    def test_three_models(self):
        """Three models should use range (max-min) for disagreement."""
        combiner = ThresholdCombiner(disagreement_scale=15.0, threshold=0.0)

        # Models: 45, 50, 55 -> range = 10
        preds = [
            ModelPrediction('cnn', 50),
            ModelPrediction('hgb', 45),
            ModelPrediction('lstm', 55),
        ]
        combiner(preds, 0.7)

        # Agreement = 1 - sqrt(10/15) = 1 - 0.816 = 0.184
        expected_agreement = 1.0 - np.sqrt(10.0 / 15.0)
        expected_score = 0.4 * expected_agreement + 0.6 * 0.7
        assert abs(combiner.get_score() - expected_score) < 1e-10

    def test_zero_quality(self):
        """Zero quality should still work."""
        combiner = ThresholdCombiner(threshold=0.3)
        preds = [ModelPrediction('cnn', 50), ModelPrediction('hgb', 50)]
        result = combiner(preds, 0.0)

        # score = 0.4 * 1.0 + 0.6 * 0.0 = 0.4 >= 0.3
        assert abs(combiner.get_score() - 0.4) < 1e-10
        assert result == 50

    def test_extreme_disagreement(self):
        """Extreme disagreement beyond scale should cap at 0 agreement."""
        combiner = ThresholdCombiner(disagreement_scale=10.0, threshold=0.0)
        preds = [ModelPrediction('cnn', 0), ModelPrediction('hgb', 100)]
        combiner(preds, 0.5)

        # disagreement = 100, capped at scale=10 -> agreement = 1 - sqrt(1) = 0
        # score = 0.4 * 0 + 0.6 * 0.5 = 0.3
        assert abs(combiner.get_score() - 0.3) < 1e-10
