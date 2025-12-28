"""
Multi-stage pipeline for boom detection.

The pipeline consists of:
1. Quality Filter: Predict boom quality and filter to high-quality simulations
2. Frame Detector: Predict boom frame (trained on high-quality only)

This improves accuracy because:
- Low-quality simulations have ambiguous boom moments (harder to predict)
- Training on high-quality data gives cleaner signal
- Filtering allows returning "uncertain" for low-quality sims

Usage:
    from boom_detection.pipeline import QualityAwarePipeline

    pipeline = QualityAwarePipeline(quality_threshold=0.5)
    pipeline.fit(train_ids, train_boom_frames, train_qualities, cache)
    predictions, confidence = pipeline.predict(test_ids, cache)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import FeatureCache
from .quality_models import QualityRegressor, QualityClassifier
from .frame_models import FrameLevelClassifier


class QualityFilter:
    """
    Filter simulations based on predicted quality.

    Can use either regression (predict score) or classification (predict high/low).
    """

    def __init__(
        self,
        mode: str = 'regression',  # 'regression' or 'classification'
        threshold: float = 0.5,
        model: str = 'ridge',
    ):
        self.mode = mode
        self.threshold = threshold
        self.model_type = model

        if mode == 'regression':
            self._model = QualityRegressor(model=model)
        else:
            self._model = QualityClassifier(model=model, threshold=threshold)

    def fit(
        self,
        sim_ids: list[str],
        qualities: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit the quality prediction model."""
        self._model.fit(sim_ids, qualities, cache)

    def predict_quality(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
    ) -> np.ndarray:
        """Predict quality scores."""
        if self.mode == 'regression':
            return self._model.predict(sim_ids, cache)
        else:
            return self._model.predict_proba(sim_ids, cache)

    def filter(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
    ) -> tuple[list[str], list[str], np.ndarray]:
        """
        Filter simulations by quality.

        Returns:
            high_quality_ids: IDs with quality >= threshold
            low_quality_ids: IDs with quality < threshold
            predicted_qualities: All predicted quality scores
        """
        qualities = self.predict_quality(sim_ids, cache)
        is_high = qualities >= self.threshold

        high_ids = [sid for sid, h in zip(sim_ids, is_high) if h]
        low_ids = [sid for sid, h in zip(sim_ids, is_high) if not h]

        return high_ids, low_ids, qualities


class QualityAwarePipeline:
    """
    Two-stage pipeline: quality filter + frame detector.

    Stage 1: Predict quality, filter to high-quality simulations
    Stage 2: Predict boom frame (model trained on high-quality only)

    For low-quality simulations, returns a "fallback" prediction
    (e.g., middle of simulation) with low confidence.
    """

    def __init__(
        self,
        quality_threshold: float = 0.5,
        quality_model: str = 'ridge',
        frame_model: str = 'hist_gbm',
        frame_n_estimators: int = 200,
        frame_max_depth: int = 7,
    ):
        self.quality_threshold = quality_threshold
        self.quality_model = quality_model
        self.frame_model = frame_model
        self.frame_n_estimators = frame_n_estimators
        self.frame_max_depth = frame_max_depth

        self._quality_filter = QualityFilter(
            mode='regression',
            threshold=quality_threshold,
            model=quality_model,
        )
        self._frame_detector = FrameLevelClassifier(
            model=frame_model,
            n_estimators=frame_n_estimators,
            max_depth=frame_max_depth,
        )
        self._fitted = False

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        qualities: np.ndarray,
        cache: FeatureCache,
    ) -> dict:
        """
        Fit both stages of the pipeline.

        Args:
            sim_ids: Simulation IDs
            boom_frames: Ground truth boom frames
            qualities: Ground truth quality scores
            cache: Feature cache

        Returns:
            Training statistics
        """
        # Stage 1: Fit quality filter on all data
        self._quality_filter.fit(sim_ids, qualities, cache)

        # Stage 2: Fit frame detector on high-quality data only
        high_quality_mask = qualities >= self.quality_threshold
        high_ids = [sid for sid, h in zip(sim_ids, high_quality_mask) if h]
        high_booms = boom_frames[high_quality_mask]

        if len(high_ids) < 5:
            # Not enough high-quality samples, use all data
            self._frame_detector.fit(sim_ids, boom_frames, cache)
            train_mode = 'all_data'
        else:
            self._frame_detector.fit(high_ids, high_booms, cache)
            train_mode = 'high_quality_only'

        self._fitted = True

        return {
            'n_total': len(sim_ids),
            'n_high_quality': len(high_ids),
            'train_mode': train_mode,
        }

    def predict(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict boom frames with confidence.

        Returns:
            predictions: Predicted boom frames
            confidences: Confidence scores (quality predictions)
            is_high_quality: Boolean mask for high-quality predictions
        """
        if not self._fitted:
            raise ValueError("Pipeline not fitted")

        # Stage 1: Predict quality
        qualities = self._quality_filter.predict_quality(sim_ids, cache)
        is_high = qualities >= self.quality_threshold

        # Stage 2: Predict boom frames
        predictions = self._frame_detector.predict(sim_ids, cache)

        # Confidence = predicted quality (higher quality = more confidence)
        confidences = np.clip(qualities, 0, 1)

        return predictions, confidences, is_high

    def evaluate(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        qualities: np.ndarray,
        cache: FeatureCache,
    ) -> dict:
        """
        Evaluate pipeline performance.

        Returns metrics for:
        - Quality prediction (MAE, accuracy)
        - Frame prediction overall
        - Frame prediction on high-quality only
        - Frame prediction on low-quality only
        """
        preds, confs, is_high = self.predict(sim_ids, cache)

        # Quality prediction metrics
        quality_mae = np.mean(np.abs(confs - qualities))
        quality_correct = np.mean((confs >= self.quality_threshold) == (qualities >= self.quality_threshold))

        # Frame prediction metrics (all)
        frame_mae_all = np.mean(np.abs(preds - boom_frames))
        within_10_all = np.mean(np.abs(preds - boom_frames) <= 10)

        # Frame prediction metrics (high-quality only)
        true_high = qualities >= self.quality_threshold
        if true_high.sum() > 0:
            high_preds = preds[true_high]
            high_booms = boom_frames[true_high]
            frame_mae_high = np.mean(np.abs(high_preds - high_booms))
            within_10_high = np.mean(np.abs(high_preds - high_booms) <= 10)
        else:
            frame_mae_high = np.nan
            within_10_high = np.nan

        # Frame prediction metrics (low-quality only)
        true_low = ~true_high
        if true_low.sum() > 0:
            low_preds = preds[true_low]
            low_booms = boom_frames[true_low]
            frame_mae_low = np.mean(np.abs(low_preds - low_booms))
            within_10_low = np.mean(np.abs(low_preds - low_booms) <= 10)
        else:
            frame_mae_low = np.nan
            within_10_low = np.nan

        # Predicted high-quality metrics
        pred_high = is_high
        if pred_high.sum() > 0:
            pred_high_preds = preds[pred_high]
            pred_high_booms = boom_frames[pred_high]
            frame_mae_pred_high = np.mean(np.abs(pred_high_preds - pred_high_booms))
            within_10_pred_high = np.mean(np.abs(pred_high_preds - pred_high_booms) <= 10)
        else:
            frame_mae_pred_high = np.nan
            within_10_pred_high = np.nan

        return {
            'quality_mae': quality_mae,
            'quality_accuracy': quality_correct,
            'frame_mae_all': frame_mae_all,
            'within_10_all': within_10_all,
            'frame_mae_true_high': frame_mae_high,
            'within_10_true_high': within_10_high,
            'frame_mae_true_low': frame_mae_low,
            'within_10_true_low': within_10_low,
            'frame_mae_pred_high': frame_mae_pred_high,
            'within_10_pred_high': within_10_pred_high,
            'n_pred_high': pred_high.sum(),
            'n_true_high': true_high.sum(),
        }


class ConditionalPipeline:
    """
    Alternative pipeline that uses different frame models based on quality.

    - High-quality: Use precise frame detector (trained on high-quality)
    - Low-quality: Use fallback (e.g., middle of simulation, or broad predictor)
    """

    def __init__(
        self,
        quality_threshold: float = 0.5,
        high_model: str = 'hist_gbm',
        low_fallback: str = 'middle',  # 'middle', 'variance_threshold', 'model'
    ):
        self.quality_threshold = quality_threshold
        self.high_model = high_model
        self.low_fallback = low_fallback

        self._quality_filter = QualityFilter(
            mode='regression',
            threshold=quality_threshold,
            model='ridge',
        )
        self._high_detector = FrameLevelClassifier(model=high_model)
        self._low_detector = None
        if low_fallback == 'model':
            self._low_detector = FrameLevelClassifier(model='hist_gbm')

        self._fitted = False

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        qualities: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Fit pipeline."""
        # Quality filter
        self._quality_filter.fit(sim_ids, qualities, cache)

        # High-quality detector
        high_mask = qualities >= self.quality_threshold
        high_ids = [sid for sid, h in zip(sim_ids, high_mask) if h]
        high_booms = boom_frames[high_mask]

        if len(high_ids) >= 5:
            self._high_detector.fit(high_ids, high_booms, cache)
        else:
            self._high_detector.fit(sim_ids, boom_frames, cache)

        # Low-quality detector (if using model fallback)
        if self._low_detector is not None:
            low_ids = [sid for sid, h in zip(sim_ids, high_mask) if not h]
            low_booms = boom_frames[~high_mask]
            if len(low_ids) >= 3:
                self._low_detector.fit(low_ids, low_booms, cache)
            else:
                self._low_detector.fit(sim_ids, boom_frames, cache)

        self._fitted = True

    def predict(
        self,
        sim_ids: list[str],
        cache: FeatureCache,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with quality-conditional model.

        Returns:
            predictions: Boom frame predictions
            qualities: Predicted quality scores
        """
        if not self._fitted:
            raise ValueError("Pipeline not fitted")

        # Get quality predictions
        _, _, qualities = self._quality_filter.filter(sim_ids, cache)
        is_high = qualities >= self.quality_threshold

        predictions = np.zeros(len(sim_ids), dtype=int)

        # High-quality: use precise detector
        high_ids = [sid for sid, h in zip(sim_ids, is_high) if h]
        if high_ids:
            high_preds = self._high_detector.predict(high_ids, cache)
            high_idx = [i for i, h in enumerate(is_high) if h]
            for idx, pred in zip(high_idx, high_preds):
                predictions[idx] = pred

        # Low-quality: use fallback
        low_ids = [sid for sid, h in zip(sim_ids, is_high) if not h]
        if low_ids:
            if self.low_fallback == 'middle':
                # Predict middle of simulation
                for i, (sid, h) in enumerate(zip(sim_ids, is_high)):
                    if not h:
                        n_frames = len(cache[sid])
                        predictions[i] = n_frames // 2
            elif self._low_detector is not None:
                low_preds = self._low_detector.predict(low_ids, cache)
                low_idx = [i for i, h in enumerate(is_high) if not h]
                for idx, pred in zip(low_idx, low_preds):
                    predictions[idx] = pred

        return predictions, qualities
