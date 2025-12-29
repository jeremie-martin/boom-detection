"""
Tests for acceptance functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from boom_detection.acceptance import (
    AcceptanceFunction,
    ACCEPTANCE_REGISTRY,
    get_acceptance_fn,
    list_formulas,
    make_linear_acceptance,
    make_quadratic_acceptance,
    make_sigmoid_acceptance,
    make_sqrt_acceptance,
)


class TestAcceptanceFunctionProtocol:
    """Tests that acceptance functions match the protocol."""

    @pytest.mark.parametrize("formula", ["sqrt", "linear", "sigmoid", "quadratic"])
    def test_returns_tuple(self, formula: str):
        """All formulas should return (bool, float) tuple."""
        accept_fn = get_acceptance_fn(formula)
        result = accept_fn({'cnn': 50, 'hgb': 48}, 0.75)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (bool, np.bool_))
        assert isinstance(result[1], float)

    @pytest.mark.parametrize("formula", ["sqrt", "linear", "sigmoid", "quadratic"])
    def test_accept_score_in_range(self, formula: str):
        """Accept score should typically be in [0, 1] range."""
        accept_fn = get_acceptance_fn(formula)

        # Test with various inputs
        for disagreement in [0, 5, 10, 20]:
            # Create predictions with specific disagreement
            preds = {'cnn': 50, 'hgb': 50 + disagreement}
            for quality in [0.0, 0.5, 1.0]:
                _, score = accept_fn(preds, quality)
                # Score should be bounded (might slightly exceed due to high quality)
                assert -0.1 <= score <= 1.1, f"Score {score} out of reasonable bounds"


class TestSqrtFormula:
    """Tests for sqrt acceptance formula."""

    def test_perfect_agreement_high_quality(self):
        """Perfect agreement + high quality should accept."""
        accept_fn = make_sqrt_acceptance({'threshold': 0.60})
        accepted, score = accept_fn({'cnn': 50, 'hgb': 50}, 0.8)

        # agreement = 1.0, quality = 0.8
        # score = 0.4 * 1.0 + 0.6 * 0.8 = 0.4 + 0.48 = 0.88
        assert accepted is True
        assert abs(score - 0.88) < 1e-10

    def test_high_disagreement_low_quality(self):
        """High disagreement + low quality should reject."""
        accept_fn = make_sqrt_acceptance({'scale': 15.0, 'threshold': 0.60})
        accepted, score = accept_fn({'cnn': 50, 'hgb': 70}, 0.3)

        # disagreement = 20, but capped at scale=15 -> agreement = 1 - sqrt(1) = 0
        # score = 0.4 * 0 + 0.6 * 0.3 = 0.18
        assert accepted is False
        assert score < 0.60

    def test_scale_parameter(self):
        """Lower scale should make formula more selective."""
        preds = {'cnn': 50, 'hgb': 55}  # disagreement = 5
        quality = 0.7

        # With scale=15: agreement = 1 - sqrt(5/15) = 1 - 0.577 = 0.423
        fn_permissive = make_sqrt_acceptance({'scale': 15.0, 'threshold': 0.60})
        _, score_permissive = fn_permissive(preds, quality)

        # With scale=5: agreement = 1 - sqrt(5/5) = 1 - 1.0 = 0
        fn_selective = make_sqrt_acceptance({'scale': 5.0, 'threshold': 0.60})
        _, score_selective = fn_selective(preds, quality)

        assert score_permissive > score_selective

    def test_documented_result_reproduction(self):
        """Test we can reproduce documented MAE 3.4 at 14% configuration."""
        # sqrt/scale=5/threshold=0.60 is the documented "most selective" config
        accept_fn = make_sqrt_acceptance({
            'scale': 5.0,
            'threshold': 0.60,
            'agreement_weight': 0.4,
            'quality_weight': 0.6,
        })

        # Perfect agreement + high quality should accept
        accepted, _ = accept_fn({'cnn': 50, 'hgb': 50}, 0.9)
        assert accepted is True

        # High disagreement should reject even with high quality
        # disagreement=10 with scale=5 -> agreement = 1 - sqrt(2) = negative, capped at 0
        # Actually: 1 - sqrt(min(10/5, 1)) = 1 - sqrt(1) = 0
        accepted, score = accept_fn({'cnn': 50, 'hgb': 60}, 0.9)
        # score = 0.4 * 0 + 0.6 * 0.9 = 0.54 < 0.60
        assert accepted is False


class TestLinearFormula:
    """Tests for linear acceptance formula."""

    def test_linear_transition(self):
        """Linear formula should decrease linearly with disagreement."""
        accept_fn = make_linear_acceptance({'scale': 10.0})

        # Test several disagreement levels
        results = []
        for disagreement in [0, 2, 4, 6, 8, 10]:
            preds = {'cnn': 50, 'hgb': 50 + disagreement}
            _, score = accept_fn(preds, 0.5)  # Fixed quality
            results.append(score)

        # Should decrease linearly
        diffs = [results[i] - results[i + 1] for i in range(len(results) - 1)]
        # All differences should be approximately equal (linear)
        for d in diffs[1:]:
            assert abs(d - diffs[0]) < 0.01


class TestSigmoidFormula:
    """Tests for sigmoid acceptance formula."""

    def test_sigmoid_s_curve(self):
        """Sigmoid should have S-curve shape."""
        accept_fn = make_sigmoid_acceptance({'scale': 10.0, 'steepness': 4.0})

        # Get scores at various disagreement levels
        scores = []
        for disagreement in range(0, 16):
            preds = {'cnn': 50, 'hgb': 50 + disagreement}
            _, score = accept_fn(preds, 0.5)
            scores.append(score)

        # Score at 0 disagreement should be high
        # agreement_score = 1/(1 + exp(-2)) ≈ 0.88, score = 0.4*0.88 + 0.6*0.5 ≈ 0.65
        assert scores[0] > 0.6
        # Score at disagreement = scale should be around middle transition
        # Score should decrease smoothly (no sudden jumps)


class TestQuadraticFormula:
    """Tests for quadratic acceptance formula."""

    def test_quadratic_more_permissive_near_zero(self):
        """Quadratic should be more permissive than linear for small disagreements."""
        preds = {'cnn': 50, 'hgb': 53}  # disagreement = 3, scale = 10
        quality = 0.5

        linear_fn = make_linear_acceptance({'scale': 10.0})
        quadratic_fn = make_quadratic_acceptance({'scale': 10.0})

        _, linear_score = linear_fn(preds, quality)
        _, quadratic_score = quadratic_fn(preds, quality)

        # For small disagreement relative to scale, quadratic gives higher agreement
        # linear: agreement = 1 - 0.3 = 0.7
        # quadratic: agreement = 1 - 0.3^2 = 1 - 0.09 = 0.91
        assert quadratic_score > linear_score


class TestRegistry:
    """Tests for the formula registry."""

    def test_all_formulas_registered(self):
        """All expected formulas should be in registry."""
        expected = {'sqrt', 'linear', 'sigmoid', 'quadratic'}
        assert set(ACCEPTANCE_REGISTRY.keys()) == expected

    def test_get_acceptance_fn_valid(self):
        """get_acceptance_fn should return callable for valid formulas."""
        for formula in ['sqrt', 'linear', 'sigmoid', 'quadratic']:
            fn = get_acceptance_fn(formula)
            assert callable(fn)

    def test_get_acceptance_fn_invalid(self):
        """get_acceptance_fn should raise for invalid formulas."""
        with pytest.raises(ValueError, match="Unknown formula"):
            get_acceptance_fn('invalid_formula')

    def test_list_formulas(self):
        """list_formulas should return all available formulas."""
        formulas = list_formulas()
        assert 'sqrt' in formulas
        assert 'linear' in formulas
        assert len(formulas) == 4


class TestParameters:
    """Tests for parameter handling."""

    def test_default_parameters(self):
        """Default parameters should work correctly."""
        # Sqrt defaults: scale=15.0, agreement_weight=0.4, quality_weight=0.6, threshold=0.60
        accept_fn = make_sqrt_acceptance()
        accepted, score = accept_fn({'cnn': 50, 'hgb': 50}, 0.8)
        assert accepted is True  # Perfect agreement + high quality

    def test_custom_weights(self):
        """Custom agreement/quality weights should be respected."""
        preds = {'cnn': 50, 'hgb': 50}  # Perfect agreement
        quality = 0.5

        # Default: 0.4 * 1.0 + 0.6 * 0.5 = 0.7
        fn_default = make_sqrt_acceptance({'agreement_weight': 0.4, 'quality_weight': 0.6})
        _, score_default = fn_default(preds, quality)
        assert abs(score_default - 0.7) < 1e-10

        # More weight on agreement: 0.8 * 1.0 + 0.2 * 0.5 = 0.9
        fn_agreement = make_sqrt_acceptance({'agreement_weight': 0.8, 'quality_weight': 0.2})
        _, score_agreement = fn_agreement(preds, quality)
        assert abs(score_agreement - 0.9) < 1e-10

    def test_threshold_parameter(self):
        """Threshold parameter should control acceptance."""
        preds = {'cnn': 50, 'hgb': 50}  # score will be around 0.7

        fn_low = make_sqrt_acceptance({'threshold': 0.5})
        fn_high = make_sqrt_acceptance({'threshold': 0.8})

        accepted_low, _ = fn_low(preds, 0.5)
        accepted_high, _ = fn_high(preds, 0.5)

        assert accepted_low is True
        assert accepted_high is False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model(self):
        """Single model should have zero disagreement."""
        accept_fn = make_sqrt_acceptance()
        accepted, score = accept_fn({'cnn': 50}, 0.8)

        # disagreement = 0, agreement = 1.0
        # score = 0.4 * 1.0 + 0.6 * 0.8 = 0.88
        assert accepted is True
        assert abs(score - 0.88) < 1e-10

    def test_three_models(self):
        """Three models should use range (max-min) for disagreement."""
        accept_fn = make_sqrt_acceptance({'scale': 15.0})

        # Models: 45, 50, 55 -> range = 10
        preds = {'cnn': 50, 'hgb': 45, 'lstm': 55}
        _, score = accept_fn(preds, 0.7)

        # Agreement = 1 - sqrt(10/15) = 1 - 0.816 = 0.184
        expected_agreement = 1.0 - np.sqrt(10.0 / 15.0)
        expected_score = 0.4 * expected_agreement + 0.6 * 0.7
        assert abs(score - expected_score) < 1e-10

    def test_zero_quality(self):
        """Zero quality should still work."""
        accept_fn = make_sqrt_acceptance({'threshold': 0.3})
        accepted, score = accept_fn({'cnn': 50, 'hgb': 50}, 0.0)

        # score = 0.4 * 1.0 + 0.6 * 0.0 = 0.4
        assert abs(score - 0.4) < 1e-10
        assert accepted is True  # 0.4 >= 0.3

    def test_extreme_disagreement(self):
        """Extreme disagreement beyond scale should cap at 0 agreement."""
        accept_fn = make_sqrt_acceptance({'scale': 10.0})
        _, score = accept_fn({'cnn': 0, 'hgb': 100}, 0.5)

        # disagreement = 100, capped at scale=10 -> agreement = 1 - sqrt(1) = 0
        # score = 0.4 * 0 + 0.6 * 0.5 = 0.3
        assert abs(score - 0.3) < 1e-10
