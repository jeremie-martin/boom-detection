"""
Model agreement analysis for boom detection.

Key finding: When multiple models agree on the boom frame, predictions are
much more accurate. Model disagreement correlates strongly with boom quality
(Spearman -0.498) and with prediction error.

Usage:
    from boom_detection.model_agreement import ModelAgreementAnalyzer

    analyzer = ModelAgreementAnalyzer()
    analyzer.fit(train_ids, train_boom_frames, cache)
    predictions, confidences = analyzer.predict_with_confidence(test_ids, cache)
"""

from __future__ import annotations

import numpy as np

from .features import FeatureCache
from .frame_models import FrameLevelClassifier


class ModelAgreementAnalyzer:
    """
    Analyze agreement between multiple models as a confidence signal.

    When models agree, predictions tend to be accurate.
    When models disagree, predictions are unreliable (often low-quality booms).
    """

    def __init__(
        self,
        primary_model: str = 'hist_gbm',
        secondary_models: list[str] | None = None,
    ):
        self.primary_model = primary_model
        self.secondary_models = secondary_models or ['logistic']
        self._models: dict[str, object] = {}

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit all models."""
        # Primary model
        self._models['primary'] = FrameLevelClassifier(
            model=self.primary_model,
            n_estimators=200,
            max_depth=7,
        )
        self._models['primary'].fit(sim_ids, boom_frames, cache)

        # Secondary models
        for i, model_type in enumerate(self.secondary_models):
            key = f'secondary_{i}'
            self._models[key] = FrameLevelClassifier(model=model_type)
            self._models[key].fit(sim_ids, boom_frames, cache)

    def predict_with_confidence(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict boom frames with confidence scores.

        Confidence is based on model agreement:
        - High confidence: Models agree within 10 frames
        - Low confidence: Models disagree significantly

        Returns:
            predictions: Primary model predictions
            confidences: Agreement-based confidence (0-1)
        """
        # Get all predictions
        all_preds = []
        for model in self._models.values():
            preds = model.predict(sim_ids, cache)
            all_preds.append(preds)

        all_preds = np.array(all_preds)

        # Primary prediction
        predictions = all_preds[0]

        # Confidence based on agreement (inverse of std)
        model_std = np.std(all_preds, axis=0)

        # Normalize: 0 std -> confidence 1, high std -> confidence 0
        # Use exponential decay: conf = exp(-std / scale)
        scale = 20  # frames - at 20 frame disagreement, confidence ~0.37
        confidences = np.exp(-model_std / scale)

        return predictions, confidences

    def get_agreement_stats(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
    ) -> dict[str, np.ndarray]:
        """Get detailed agreement statistics."""
        all_preds = []
        for model in self._models.values():
            preds = model.predict(sim_ids, cache)
            all_preds.append(preds)

        all_preds = np.array(all_preds)

        return {
            'std': np.std(all_preds, axis=0),
            'range': np.ptp(all_preds, axis=0),
            'mean': np.mean(all_preds, axis=0),
            'median': np.median(all_preds, axis=0),
        }


class ConfidenceWeightedEnsemble:
    """
    Ensemble that weights predictions by model confidence.

    Uses model agreement to determine how much to trust each model.
    When models agree, use the mean. When they disagree, favor the
    model that is typically more accurate.
    """

    def __init__(
        self,
        agreement_threshold: float = 10.0,  # frames
    ):
        self.agreement_threshold = agreement_threshold
        self._primary = None
        self._secondary = None

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit both models."""
        self._primary = FrameLevelClassifier(
            model='hist_gbm',
            n_estimators=200,
            max_depth=7,
        )
        self._primary.fit(sim_ids, boom_frames, cache)

        self._secondary = FrameLevelClassifier(model='logistic')
        self._secondary.fit(sim_ids, boom_frames, cache)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Predict using confidence-weighted ensemble."""
        pred1 = self._primary.predict(sim_ids, cache)
        pred2 = self._secondary.predict(sim_ids, cache)

        disagreement = np.abs(pred1 - pred2)

        # When models agree, use mean
        # When they disagree, favor primary (typically more accurate)
        agree_mask = disagreement <= self.agreement_threshold

        predictions = np.zeros(len(sim_ids))
        predictions[agree_mask] = (pred1[agree_mask] + pred2[agree_mask]) / 2
        predictions[~agree_mask] = pred1[~agree_mask]

        return np.round(predictions).astype(int)


class AgreementBasedSelector:
    """
    Select prediction from the model that is most confident.

    For each simulation, choose the prediction from the model
    whose prediction is closest to the consensus.
    """

    def __init__(self, models: list[str] | None = None):
        self.model_types = models or ['hist_gbm', 'logistic']
        self._models: list[object] = []

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit all models."""
        self._models = []
        for model_type in self.model_types:
            model = FrameLevelClassifier(
                model=model_type,
                n_estimators=200 if model_type == 'hist_gbm' else 100,
                max_depth=7 if model_type == 'hist_gbm' else 5,
            )
            model.fit(sim_ids, boom_frames, cache)
            self._models.append(model)

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        """Select prediction closest to consensus."""
        all_preds = np.array([m.predict(sim_ids, cache) for m in self._models])
        median_pred = np.median(all_preds, axis=0)

        # For each simulation, pick the prediction closest to median
        predictions = []
        for i in range(len(sim_ids)):
            distances = np.abs(all_preds[:, i] - median_pred[i])
            best_model = np.argmin(distances)
            predictions.append(int(all_preds[best_model, i]))

        return np.array(predictions)


def analyze_model_agreement(
    predictions_dict: dict[str, np.ndarray],
    boom_frames: np.ndarray,
    qualities: np.ndarray,
) -> dict:
    """
    Analyze agreement between multiple model predictions.

    Args:
        predictions_dict: Dict of {model_name: predictions}
        boom_frames: Ground truth boom frames
        qualities: Ground truth quality scores

    Returns:
        Analysis results
    """
    from scipy.stats import spearmanr

    pred_array = np.array(list(predictions_dict.values()))
    model_std = np.std(pred_array, axis=0)
    model_range = np.ptp(pred_array, axis=0)
    mean_pred = np.mean(pred_array, axis=0)
    mean_error = np.abs(mean_pred - boom_frames)

    # Correlations
    std_quality_corr, std_quality_p = spearmanr(model_std, qualities)
    std_error_corr, std_error_p = spearmanr(model_std, mean_error)

    # Split by agreement level
    low_var_mask = model_std <= np.median(model_std)

    return {
        'std_quality_correlation': std_quality_corr,
        'std_quality_pvalue': std_quality_p,
        'std_error_correlation': std_error_corr,
        'std_error_pvalue': std_error_p,
        'low_var_mae': np.mean(mean_error[low_var_mask]),
        'high_var_mae': np.mean(mean_error[~low_var_mask]),
        'mean_model_std': np.mean(model_std),
        'median_model_std': np.median(model_std),
    }
