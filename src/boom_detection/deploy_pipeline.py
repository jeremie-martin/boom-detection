"""
Deployable boom detection pipeline.

This implements the best-performing approach:
1. Run CNN and HistGBM in parallel
2. Use a Combiner to decide whether to accept and what frame to return
3. Default combiner uses agreement + quality with sqrt transform

Performance (5-fold CV × 5 seeds):
- MAE: 4.9 ± 1.3 frames at 30% coverage (sqrt/scale=15/threshold=0.60)
- MAE: 3.4 ± 0.8 frames at 14% coverage (sqrt/scale=5/threshold=0.60)

Key features:
- Unified Combiner abstraction for acceptance + prediction
- CNN prediction (not HGB) - more accurate when models agree
- Random Forest for quality (not Ridge) - better correlation
- Larger CNN kernels (5,11,21) - capture longer-range patterns

Usage:
    # Evaluate with robust multi-seed CV
    uv run python -m boom_detection.deploy_pipeline data --evaluate

    # Quick single-seed evaluation (for development)
    uv run python -m boom_detection.deploy_pipeline data --evaluate --quick

    # Custom configuration
    uv run python -m boom_detection.deploy_pipeline data --evaluate --scale 5 --threshold 0.60

    # Train final models and save
    uv run python -m boom_detection.deploy_pipeline data --train --output models/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor

from .loader import load_dataset
from .features import FeatureCache, FeatureConfig, PRODUCTION_CONFIG
from .logging_config import logger
from .metrics import SelectivePrediction
from .sequence_models import CNNClassifier, LSTMClassifier, SequenceTrainer
from .combine import (
    ModelPrediction,
    ThresholdCombiner,
    Combiner,
    combiner_to_config,
    combiner_from_config,
    default_combiner,
    disagreement,
    FrameModelConfig,
    normalize_model_configs,
)

# Valid frame model types
VALID_MODEL_TYPES = ('cnn', 'hgb', 'lstm')


class BoomDetectionPipeline:
    """
    Production-ready boom detection pipeline.

    Uses a Combiner to decide whether to accept predictions and produce final output.
    The default ThresholdCombiner uses model agreement + predicted quality.

    Supports N frame prediction models. Each model can be configured with a
    quality_threshold for training data filtering:
    - quality_threshold=0.0: Train on ALL data (regular model)
    - quality_threshold>0.0: Train only on samples with quality >= threshold (specialized)

    Example configurations:
        # 2-model (regular CNN + HGB)
        BoomDetectionPipeline(frame_models=('cnn', 'hgb'))

        # 3-model with LSTM
        BoomDetectionPipeline(frame_models=('cnn', 'hgb', 'lstm'))

        # Specialized HGB trained on high-quality data only
        BoomDetectionPipeline(frame_models=(
            'cnn',
            'hgb',
            FrameModelConfig('hgb', 0.5),  # hgb_0.5: trained on quality >= 0.5
        ))
    """

    def __init__(
        self,
        combiner: Combiner | None = None,
        quality_window: int = 25,
        n_quality_features: int = 50,
        seed: int | None = None,
        calibrate_quality: bool = True,
        frame_models: tuple[str | FrameModelConfig, ...] = ('cnn', 'hgb'),
        feature_config: FeatureConfig | None = None,
    ):
        """
        Initialize boom detection pipeline.

        Args:
            combiner: Combiner instance for acceptance/prediction. If None, uses
                     default ThresholdCombiner(sqrt, scale=15, threshold=0.60).
            quality_window: Window around predicted boom for quality features
            n_quality_features: Top correlated features for quality prediction
            seed: Random seed for reproducibility
            calibrate_quality: Calibrate quality predictions with isotonic regression
            frame_models: Which frame models to use. Can be strings ('cnn', 'hgb', 'lstm')
                         or FrameModelConfig instances for specialized models.
                         Examples:
                           ('cnn', 'hgb')  # Regular 2-model
                           ('cnn', 'hgb', 'lstm')  # Regular 3-model
                           ('cnn', 'hgb', FrameModelConfig('hgb', 0.5))  # With specialized
                           ('cnn', 'hgb_0.5')  # String shorthand for specialized
            feature_config: Feature extraction config (saved with model)
        """
        # Normalize frame_models to FrameModelConfig instances
        self.model_configs = normalize_model_configs(frame_models)

        # Validate model types
        for config in self.model_configs:
            if config.model_type not in VALID_MODEL_TYPES:
                raise ValueError(
                    f"Invalid model type: {config.model_type}. "
                    f"Must be one of {VALID_MODEL_TYPES}"
                )

        # Set up combiner
        if combiner is None:
            combiner = default_combiner()
        self.combiner = combiner

        # Validate that primary_model (if ThresholdCombiner) matches a model config
        if isinstance(combiner, ThresholdCombiner):
            # Check if primary_model matches any model's type (for regular models)
            # or key (for specialized models)
            model_keys = {cfg.key for cfg in self.model_configs}
            model_types = {cfg.model_type for cfg in self.model_configs}
            if combiner.primary_model not in model_keys and combiner.primary_model not in model_types:
                raise ValueError(
                    f"Combiner's primary_model '{combiner.primary_model}' "
                    f"must match a model in frame_models. Available: {model_keys}"
                )

        self.quality_window = quality_window
        self.n_quality_features = n_quality_features
        self.seed = seed
        self.calibrate_quality = calibrate_quality
        # Keep frame_models as tuple of strings for backward compatibility
        self.frame_models = tuple(cfg.key for cfg in self.model_configs)
        self.feature_config = feature_config if feature_config is not None else PRODUCTION_CONFIG

        # Trained models (set during fit)
        # Key is the model's unique key (e.g., 'cnn', 'hgb', 'hgb_0.5')
        self.trained_models: dict[str, Any] = {}
        self.trainers: dict[str, SequenceTrainer] = {}
        self.quality_model = None
        self.quality_calibrator = None
        self.quality_feature_indices = None
        self.n_features = None

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self.seed = seed

    def set_combiner(self, combiner: Combiner) -> None:
        """
        Update the combiner at runtime.

        Useful for CombinerExperiment to test different configurations
        without recreating the pipeline.

        Args:
            combiner: New combiner instance
        """
        self.combiner = combiner

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        qualities: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Train all models on the given data."""
        fit_start = time.time()
        logger.info("Training pipeline on {} simulations with models: {}",
                   len(sim_ids), self.frame_models)

        # Get feature count
        self.n_features = cache[sim_ids[0]].shape[1]

        # Train each frame model (unified handling for regular and specialized)
        for i, config in enumerate(self.model_configs):
            # Derive deterministic seed for this model
            model_seed = (self.seed + i * 1000) if self.seed is not None else 42

            # Filter training data by quality threshold
            if config.quality_threshold > 0.0:
                # Specialized model: filter to high-quality samples
                mask = qualities >= config.quality_threshold
                train_sim_ids = [sid for sid, m in zip(sim_ids, mask) if m]
                train_booms = boom_frames[mask]

                n_samples = len(train_sim_ids)
                logger.info("Training {} '{}' on {}/{} samples (quality >= {})",
                           config.model_type, config.key, n_samples, len(sim_ids),
                           config.quality_threshold)

                if n_samples < 5:
                    logger.warning("Too few samples ({}) for model '{}', skipping",
                                  n_samples, config.key)
                    continue
            else:
                # Regular model: train on all data
                train_sim_ids = sim_ids
                train_booms = boom_frames

            # Train the model based on type
            if config.model_type == 'cnn':
                self._train_cnn(config.key, train_sim_ids, train_booms, cache, model_seed)
            elif config.model_type == 'hgb':
                self._train_hgb(config.key, train_sim_ids, train_booms, cache, model_seed)
            elif config.model_type == 'lstm':
                self._train_lstm(config.key, train_sim_ids, train_booms, cache, model_seed)

        # Train quality predictor
        quality_seed = (self.seed + len(self.model_configs) * 1000) if self.seed is not None else 42
        self._train_quality_model(sim_ids, boom_frames, qualities, cache, quality_seed)

        total_time = time.time() - fit_start
        logger.info("Pipeline training complete in {:.1f}s", total_time)

    def _train_cnn(self, model_key: str, sim_ids, boom_frames, cache, seed):
        """Train CNN model."""
        logger.debug("Training CNN '{}'...", model_key)
        start = time.time()
        model = CNNClassifier(
            n_features=self.n_features,
            hidden_dim=64,
            kernel_sizes=(5, 11, 21)
        )
        trainer = SequenceTrainer(
            model, lr=0.5e-3, epochs=30, patience=5, batch_size=4, augment=False,
            seed=seed,
        )
        trainer.fit(sim_ids, boom_frames, cache)
        self.trained_models[model_key] = model
        self.trainers[model_key] = trainer
        logger.debug("CNN '{}' trained in {:.1f}s", model_key, time.time() - start)

    def _train_lstm(self, model_key: str, sim_ids, boom_frames, cache, seed):
        """Train LSTM model."""
        logger.debug("Training LSTM '{}'...", model_key)
        start = time.time()
        model = LSTMClassifier(
            n_features=self.n_features,
            hidden_dim=128,
            n_layers=2,
            dropout=0.3,
        )
        trainer = SequenceTrainer(
            model, lr=0.5e-3, epochs=30, patience=5, batch_size=4, augment=False,
            seed=seed,
        )
        trainer.fit(sim_ids, boom_frames, cache)
        self.trained_models[model_key] = model
        self.trainers[model_key] = trainer
        logger.debug("LSTM '{}' trained in {:.1f}s", model_key, time.time() - start)

    def _train_hgb(self, model_key: str, sim_ids, boom_frames, cache, seed):
        """Train HistGradientBoosting model."""
        logger.debug("Training HistGBM '{}'...", model_key)
        start = time.time()

        # Build frame-level training data
        X_train, y_train = [], []
        for sid, boom in zip(sim_ids, boom_frames):
            feats = cache[sid]
            for t in range(len(feats)):
                X_train.append(feats[t])
                y_train.append(1 if t >= boom else 0)

        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=7, random_state=seed
        )
        model.fit(np.array(X_train), np.array(y_train))
        self.trained_models[model_key] = model
        logger.debug("HistGBM '{}' trained in {:.1f}s on {} frames", model_key, time.time() - start, len(X_train))

    def _train_quality_model(self, sim_ids, boom_frames, qualities, cache, seed):
        """Train quality prediction model."""
        logger.debug("Training quality predictor...")

        # Add jitter to boom frame during training to simulate prediction noise
        # This prevents train/inference mismatch since at inference we use predicted boom
        jitter_std = 5  # Standard deviation of jitter (typical model error is ~5-10 frames)
        rng = np.random.RandomState(seed)

        X_qual = []
        for sid, boom in zip(sim_ids, boom_frames):
            feats = cache[sid]
            # Add jitter to boom frame to simulate prediction uncertainty
            jittered_boom = int(boom + rng.normal(0, jitter_std))
            jittered_boom = max(0, min(jittered_boom, len(feats) - 1))

            start = max(0, jittered_boom - self.quality_window)
            end = min(len(feats), jittered_boom + self.quality_window)
            X_qual.append(feats[start:end].mean(axis=0))
        X_qual = np.array(X_qual)

        # Select top features by correlation with quality
        correlations = []
        for i in range(X_qual.shape[1]):
            r, _ = spearmanr(X_qual[:, i], qualities)
            correlations.append((i, abs(r) if not np.isnan(r) else 0))
        correlations.sort(key=lambda x: x[1], reverse=True)
        self.quality_feature_indices = [c[0] for c in correlations[:self.n_quality_features]]

        # Train on selected features using Random Forest
        X_qual_selected = X_qual[:, self.quality_feature_indices]
        self.quality_model = RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=seed
        )
        self.quality_model.fit(X_qual_selected, qualities)

        # Optional: Calibrate quality predictions using isotonic regression
        if self.calibrate_quality:
            from sklearn.isotonic import IsotonicRegression

            # Get out-of-bag predictions for calibration (leave-one-out)
            raw_predictions = np.zeros(len(qualities))
            for i in range(len(X_qual_selected)):
                mask = np.ones(len(X_qual_selected), dtype=bool)
                mask[i] = False
                temp_model = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=seed
                )
                temp_model.fit(X_qual_selected[mask], qualities[mask])
                raw_predictions[i] = temp_model.predict([X_qual_selected[i]])[0]

            # Fit isotonic regression calibrator
            self.quality_calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip'
            )
            self.quality_calibrator.fit(raw_predictions, qualities)
            logger.debug("Quality calibrator fitted")

    def predict_one(self, features: np.ndarray) -> SelectivePrediction:
        """
        Predict boom frame for a single simulation.

        Args:
            features: Shape (frames, n_features)

        Returns:
            SelectivePrediction with model predictions and confidence scores

        Raises:
            TypeError: If features is not a numpy array
            ValueError: If features has wrong shape
            RuntimeError: If pipeline has not been fitted
        """
        # Input validation
        if not self.trained_models:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        if not isinstance(features, np.ndarray):
            raise TypeError(f"Expected ndarray, got {type(features).__name__}")
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array (frames, features), got {features.ndim}D")
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {features.shape[1]}. "
                "Ensure feature extraction uses the same config as training."
            )

        # Get predictions from all models (unified handling)
        model_preds = []
        model_predictions_dict = {}
        for model_key in self.frame_models:
            if model_key not in self.trained_models:
                # Model was skipped during training (e.g., too few samples)
                continue
            frame = self._predict_with_model(model_key, features)
            model_preds.append(ModelPrediction(model_key, frame))
            model_predictions_dict[model_key] = frame

        # Predict quality (using median of all model predictions)
        frame_preds = [p.frame for p in model_preds]
        median_pred = int(np.median(frame_preds))
        predicted_quality = self._predict_quality(features, median_pred)

        # Call combiner with all model predictions
        result = self.combiner(model_preds, predicted_quality, features)
        accepted = result is not None

        # Get accept_score if available (for ThresholdCombiner)
        accept_score = 0.0
        if hasattr(self.combiner, 'get_score'):
            accept_score = self.combiner.get_score()
        elif accepted:
            accept_score = 1.0  # For non-threshold combiners

        # Compute disagreement for metadata
        dis = disagreement(model_preds, metric='range')

        return SelectivePrediction(
            boom_frame=result,
            accepted=accepted,
            accept_score=float(accept_score),
            predicted_quality=predicted_quality,
            model_predictions=model_predictions_dict,
            disagreement=dis,
        )

    def _predict_with_model(self, model_name: str, features: np.ndarray) -> int:
        """Get boom frame prediction from a specific model."""
        import torch

        # Check if model exists
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")

        model = self.trained_models[model_name]

        # Check if it's a PyTorch model (has trainer) - works for both base and specialized models
        if model_name in self.trainers:
            trainer = self.trainers[model_name]
            model.eval()
            with torch.no_grad():
                feats_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
                feats_t = feats_t.to(trainer.device)
                logits = model(feats_t)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            crossings = np.where(probs >= 0.5)[0]
            return int(crossings[0]) if len(crossings) > 0 else int(np.argmax(probs))

        # Otherwise it's an HGB-type model (sklearn)
        else:
            probs = model.predict_proba(features)[:, 1]
            crossings = np.where(probs >= 0.5)[0]
            return int(crossings[0]) if len(crossings) > 0 else int(np.argmax(probs))

    def _predict_quality(self, features: np.ndarray, boom_estimate: int) -> float:
        """Predict quality score given features and boom estimate."""
        start = max(0, boom_estimate - self.quality_window)
        end = min(len(features), boom_estimate + self.quality_window)
        window_feats = features[start:end].mean(axis=0)
        window_feats_selected = window_feats[self.quality_feature_indices]
        raw_quality = float(self.quality_model.predict([window_feats_selected])[0])

        if self.quality_calibrator is not None:
            return float(np.clip(
                self.quality_calibrator.predict([raw_quality])[0], 0, 1
            ))
        return float(np.clip(raw_quality, 0, 1))

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> list[SelectivePrediction]:
        """Predict for multiple simulations."""
        return [self.predict_one(cache[sid]) for sid in sim_ids]

    def save(self, path: Path) -> None:
        """
        Save the pipeline to a directory.

        Creates model files for each frame model (cnn.pt, lstm.pt, hgb.joblib, etc.)
        plus quality.joblib, config.json, and feature_config.json.

        Model files are named by their key (e.g., 'cnn.pt', 'hgb_0.5.joblib').
        """
        import torch
        import joblib
        from dataclasses import asdict

        if not self.trained_models:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each frame model using its config
        for config in self.model_configs:
            model_key = config.key
            if model_key not in self.trained_models:
                # Model was skipped during training
                continue

            model = self.trained_models[model_key]

            if config.model_type in ('cnn', 'lstm'):
                # PyTorch models - save with architecture params
                torch.save({
                    'state_dict': model.state_dict(),
                    'n_features': model.n_features,
                    'hidden_dim': model.hidden_dim,
                    'model_type': config.model_type,
                    'model_key': model_key,
                    # Model-specific params
                    **({'kernel_sizes': model.kernel_sizes, 'dropout': model.dropout}
                       if config.model_type == 'cnn' else
                       {'n_layers': model.n_layers, 'dropout': model.dropout}),
                }, path / f'{model_key}.pt')
            elif config.model_type == 'hgb':
                joblib.dump(model, path / f'{model_key}.joblib')

        # Save quality model
        joblib.dump(self.quality_model, path / 'quality.joblib')
        if self.quality_calibrator is not None:
            joblib.dump(self.quality_calibrator, path / 'quality_calibrator.joblib')

        # Serialize combiner
        if not hasattr(self.combiner, 'to_config'):
            raise ValueError(
                "Cannot save pipeline with non-serializable combiner. "
                "Use a combiner from boom_detection.combine that supports to_config()."
            )

        # Serialize model configs for loading
        model_configs_serialized = [
            {'model_type': cfg.model_type, 'quality_threshold': cfg.quality_threshold}
            for cfg in self.model_configs
        ]

        config = {
            'frame_models': list(self.frame_models),  # For backward compat (list of keys)
            'model_configs': model_configs_serialized,  # Full config for loading
            'combiner': combiner_to_config(self.combiner),
            'quality_window': self.quality_window,
            'n_quality_features': self.n_quality_features,
            'n_features': self.n_features,
            'quality_feature_indices': self.quality_feature_indices,
            'calibrate_quality': self.calibrate_quality,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save feature extraction config
        feature_config_dict = asdict(self.feature_config)
        with open(path / 'feature_config.json', 'w') as f:
            json.dump(feature_config_dict, f, indent=2)

        logger.info("Pipeline saved to {} (models: {})", path, self.frame_models)

    @classmethod
    def from_pretrained(cls, path: Path, device: str = 'auto') -> 'BoomDetectionPipeline':
        """
        Load a pretrained pipeline from a directory.

        Args:
            path: Directory containing saved models
            device: Device for PyTorch models ('auto', 'cuda', 'cpu')

        Returns:
            Loaded BoomDetectionPipeline ready for inference
        """
        import torch
        import joblib

        path = Path(path)

        # Load config
        with open(path / 'config.json') as f:
            config = json.load(f)

        # Load feature config
        feature_config = None
        feature_config_path = path / 'feature_config.json'
        if feature_config_path.exists():
            with open(feature_config_path) as f:
                feature_config_dict = json.load(f)
                feature_config = FeatureConfig(**feature_config_dict)

        # Reconstruct model configs
        # New format has 'model_configs', old format only has 'frame_models' (strings)
        if 'model_configs' in config:
            model_configs = tuple(
                FrameModelConfig(cfg['model_type'], cfg['quality_threshold'])
                for cfg in config['model_configs']
            )
        else:
            # Backward compat: parse from frame_models strings
            model_configs = normalize_model_configs(config['frame_models'])

        # Load combiner config
        combiner = combiner_from_config(config['combiner'])

        # Create pipeline with saved config
        pipeline = cls(
            combiner=combiner,
            quality_window=config['quality_window'],
            n_quality_features=config['n_quality_features'],
            calibrate_quality=config.get('calibrate_quality', False),
            frame_models=model_configs,
            feature_config=feature_config,
        )

        pipeline.n_features = config['n_features']
        pipeline.quality_feature_indices = config['quality_feature_indices']

        # Set up device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load each frame model using its config
        for cfg in model_configs:
            model_key = cfg.key
            model_type = cfg.model_type

            if model_type in ('cnn', 'lstm'):
                model_path = path / f'{model_key}.pt'
                if not model_path.exists():
                    logger.warning("Model file {} not found, skipping", model_path)
                    continue

                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                if model_type == 'cnn':
                    model = CNNClassifier(
                        n_features=checkpoint['n_features'],
                        hidden_dim=checkpoint['hidden_dim'],
                        kernel_sizes=tuple(checkpoint['kernel_sizes']),
                        dropout=checkpoint.get('dropout', 0.3),
                    )
                else:  # lstm
                    model = LSTMClassifier(
                        n_features=checkpoint['n_features'],
                        hidden_dim=checkpoint['hidden_dim'],
                        n_layers=checkpoint.get('n_layers', 2),
                        dropout=checkpoint.get('dropout', 0.3),
                    )
                model.load_state_dict(checkpoint['state_dict'])
                model = model.to(device)
                model.eval()
                pipeline.trained_models[model_key] = model
                pipeline.trainers[model_key] = SequenceTrainer(model, device=device)

            elif model_type == 'hgb':
                model_path = path / f'{model_key}.joblib'
                if not model_path.exists():
                    logger.warning("Model file {} not found, skipping", model_path)
                    continue
                pipeline.trained_models[model_key] = joblib.load(model_path)

        # Load quality model
        pipeline.quality_model = joblib.load(path / 'quality.joblib')

        # Load calibrator if available
        calibrator_path = path / 'quality_calibrator.joblib'
        if calibrator_path.exists():
            pipeline.quality_calibrator = joblib.load(calibrator_path)

        return pipeline

    def predict_simulation(self, simulation: np.ndarray) -> SelectivePrediction:
        """
        Predict boom frame for a raw simulation array.

        This is the main inference method for deployment. It handles feature
        extraction internally using the saved feature config.

        Args:
            simulation: Raw simulation data, shape (frames, pendulums, 8)
                       where 8 = [x1, y1, x2, y2, th1, th2, w1, w2]

        Returns:
            SelectivePrediction with boom_frame, accepted, and quality scores

        Example:
            >>> pipeline = BoomDetectionPipeline.from_pretrained('models/v1')
            >>> sim_data = np.load('new_simulation.npy')  # or load_simulation(...)
            >>> result = pipeline.predict_simulation(sim_data)
            >>> if result.accepted:
            ...     print(f"Boom at frame {result.boom_frame}")
        """
        from .features import FeatureExtractor

        if not self.trained_models:
            raise RuntimeError("Pipeline not loaded. Use from_pretrained() first.")

        # Wrap raw array in a simple object with .data attribute
        # (FeatureExtractor.transform expects simulation.data)
        class SimulationWrapper:
            def __init__(self, data):
                self.data = data

        wrapped = SimulationWrapper(simulation)

        # Extract features using the saved config
        extractor = FeatureExtractor(self.feature_config)
        features = extractor.transform(wrapped)

        # Validate feature count matches training
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"Feature count mismatch: got {features.shape[1]}, expected {self.n_features}. "
                "Make sure the feature_config matches what was used during training."
            )

        return self.predict_one(features)

    def predict_file(self, simulation_path: str | Path) -> SelectivePrediction:
        """
        Predict boom frame for a simulation file.

        Convenience method that loads the simulation and predicts in one call.

        Args:
            simulation_path: Path to simulation_data.bin file

        Returns:
            SelectivePrediction with boom_frame, accepted, and quality scores

        Example:
            >>> pipeline = BoomDetectionPipeline.from_pretrained('models/v1')
            >>> result = pipeline.predict_file('data/simulations/run_xxx/simulation_data.bin')
            >>> print(f"Accepted: {result.accepted}, Boom: {result.boom_frame}")
        """
        from .loader import load_simulation

        if not self.trained_models:
            raise RuntimeError("Pipeline not loaded. Use from_pretrained() first.")

        simulation = load_simulation(simulation_path)
        return self.predict_simulation(simulation.data)


def cross_validate(
    sim_ids: list[str],
    boom_frames: np.ndarray,
    qualities: np.ndarray,
    cache: FeatureCache,
    n_splits: int = 5,
    seeds: list[int] | None = None,
    combiner: Combiner | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run robust multi-seed cross-validation using the unified evaluation framework.

    This is a convenience wrapper around CachedEvaluator.cross_validate_selective()
    that returns a backward-compatible dict format.

    For new code, prefer using CachedEvaluator directly:
        evaluator = CachedEvaluator(dataset, cache)
        result = evaluator.cross_validate_selective(
            lambda: BoomDetectionPipeline(combiner=ThresholdCombiner(...)),
        )

    Args:
        sim_ids: List of simulation IDs
        boom_frames: Ground truth boom frames
        qualities: Ground truth quality scores
        cache: FeatureCache with extracted features
        n_splits: Number of CV folds
        seeds: Random seeds (default: 5 seeds for robust evaluation)
        combiner: Combiner instance (default: ThresholdCombiner with sqrt/scale=15/threshold=0.60)
        verbose: Print progress

    Returns:
        Dict with metrics including mean ± std across seeds
    """
    from .evaluation import CachedEvaluator, MultiSeedSelectiveResult

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]  # 5 seeds by default

    if combiner is None:
        combiner = default_combiner()

    # Create a minimal dataset structure for the evaluator
    class MinimalDataset:
        def __init__(self, sim_ids, booms, quals):
            from dataclasses import dataclass

            @dataclass
            class MinimalAnnotation:
                id: str
                boom_frame: int
                boom_quality: float

            self.annotations = [
                MinimalAnnotation(sid, int(b), float(q))
                for sid, b, q in zip(sim_ids, booms, quals)
            ]

    dataset = MinimalDataset(sim_ids, boom_frames, qualities)
    evaluator = CachedEvaluator(dataset, cache)

    # Capture combiner for the lambda (create fresh copies for each pipeline)
    combiner_config = combiner_to_config(combiner)

    # Use the unified evaluator
    result: MultiSeedSelectiveResult = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            combiner=combiner_from_config(combiner_config),
        ),
        k=n_splits,
        seeds=seeds,
        verbose=verbose,
    )

    # Convert to backward-compatible dict format
    return {
        'n_seeds': len(seeds),
        'n_splits': n_splits,
        'seeds': seeds,
        # Main metrics with uncertainty (backward compatible names)
        'mae_mean': result.mean_metrics.get('selective_mae', float('nan')),
        'mae_std': result.std_metrics.get('selective_mae', 0.0),
        'within_5_mean': result.mean_metrics.get('selective_within_5', float('nan')) * 100,
        'within_5_std': result.std_metrics.get('selective_within_5', 0.0) * 100,
        'acceptance_rate_mean': result.mean_metrics.get('coverage', 0.0) * 100,
        'acceptance_rate_std': result.std_metrics.get('coverage', 0.0) * 100,
        # Per-seed results for analysis
        'seed_metrics': result.seed_metrics,
        # Config
        'combiner_config': combiner_config,
        # Also expose the full result object for new code
        '_result': result,
    }


def main():
    parser = argparse.ArgumentParser(description='Boom detection pipeline')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--evaluate', action='store_true', help='Run cross-validation')
    parser.add_argument('--quick', action='store_true', help='Quick single-seed evaluation')
    parser.add_argument('--train', action='store_true', help='Train and save models')
    parser.add_argument('--output', type=Path, help='Output directory for models')
    parser.add_argument('--acceptance-formula', choices=['sqrt', 'linear', 'sigmoid', 'quadratic'],
                        default='sqrt', help='Acceptance formula (default: sqrt)')
    parser.add_argument('--scale', type=float, default=15.0,
                        help='Scale parameter for acceptance formula (default: 15.0)')
    parser.add_argument('--threshold', type=float, default=0.60,
                        help='Accept score threshold (0-1, default 0.60)')
    parser.add_argument('--production', action='store_true',
                        help='Use PRODUCTION_CONFIG with caustic features (recommended)')
    parser.add_argument('--save-run', type=Path, default=None,
                        help='Save evaluation results to directory (auto-named if "auto")')
    parser.add_argument('--predict', type=Path, default=None,
                        help='Path to trained models for inference')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    dataset = load_dataset(args.data_path)

    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    # Build feature cache
    print("Building feature cache...")
    if args.production:
        print("Using PRODUCTION_CONFIG with caustic features")
        config = PRODUCTION_CONFIG
    else:
        config = FeatureConfig()
    cache = FeatureCache(config, cache_dir='.feature_cache')
    cache.extract_all(dataset, verbose=False)

    if args.evaluate:
        # Determine seeds based on --quick flag
        seeds = [42] if args.quick else [42, 43, 44, 45, 46]
        mode = "quick (1 seed)" if args.quick else "robust (5 seeds)"

        # Build combiner from CLI args
        combiner = ThresholdCombiner(
            primary_model='cnn',
            agreement_transform=args.acceptance_formula,
            disagreement_scale=args.scale,
            threshold=args.threshold,
        )

        print(f"\nRunning {mode} 5-fold cross-validation...")
        print(f"Combiner: {combiner}")

        results = cross_validate(
            sim_ids, boom_frames, qualities, cache,
            seeds=seeds,
            combiner=combiner,
        )

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        if args.quick:
            # Single seed - just show the values
            r = results['seed_metrics'][0]
            print(f"Selective MAE: {r['selective_mae']:.2f} frames")
            print(f"Within 5 frames: {r['selective_within_5']*100:.1f}%")
            print(f"Coverage: {r['coverage']*100:.1f}%")
            print()
            print("Note: This is a quick single-seed result. For robust evaluation,")
            print("      run without --quick to get mean +/- std across 5 seeds.")
        else:
            # Multi-seed - show mean +/- std
            print(f"Selective MAE: {results['mae_mean']:.2f} +/- {results['mae_std']:.2f} frames")
            print(f"Within 5 frames: {results['within_5_mean']:.1f}% +/- {results['within_5_std']:.1f}%")
            print(f"Coverage: {results['acceptance_rate_mean']:.1f}% +/- {results['acceptance_rate_std']:.1f}%")
            print()
            print(f"Based on {results['n_seeds']} random seeds x {results['n_splits']}-fold CV")

        # Save run artifact if requested
        if args.save_run:
            import datetime
            import subprocess

            # Auto-generate run name if "auto" is specified
            run_path = args.save_run
            if str(run_path) == "auto":
                # Generate: runs/YYYY-MM-DD_HHMMSS_<git_hash>
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                try:
                    git_hash = subprocess.run(
                        ['git', 'rev-parse', '--short', 'HEAD'],
                        capture_output=True, text=True, timeout=5
                    ).stdout.strip()
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    git_hash = "unknown"
                run_path = Path(f"runs/{timestamp}_{git_hash}")

            # Create config dict
            run_config = {
                'combiner': results['combiner_config'],
                'production_features': args.production,
                'n_splits': results['n_splits'],
                'seeds': results['seeds'],
                'n_seeds': results['n_seeds'],
            }

            # Save metrics and config
            run_path.mkdir(parents=True, exist_ok=True)

            with open(run_path / 'config.json', 'w') as f:
                json.dump(run_config, f, indent=2)

            # Save aggregated metrics (use consistent "coverage" naming)
            metrics_data = {
                'selective_mae_mean': results['mae_mean'],
                'selective_mae_std': results['mae_std'],
                'selective_within_5_mean': results['within_5_mean'] / 100,  # Store as fraction
                'selective_within_5_std': results['within_5_std'] / 100,
                'coverage_mean': results['acceptance_rate_mean'] / 100,  # Store as fraction
                'coverage_std': results['acceptance_rate_std'] / 100,
            }
            with open(run_path / 'metrics.json', 'w') as f:
                json.dump(metrics_data, f, indent=2)

            # Save per-seed metrics
            with open(run_path / 'seed_metrics.jsonl', 'w') as f:
                for seed_metric in results['seed_metrics']:
                    f.write(json.dumps(seed_metric) + '\n')

            # Save environment info
            import sys
            env_info = {
                'timestamp': datetime.datetime.now().isoformat(),
                'python_version': sys.version.split()[0],
                'quick_mode': args.quick,
            }
            try:
                env_info['git_commit'] = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True, text=True, timeout=5
                ).stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                env_info['git_commit'] = "unknown"
            with open(run_path / 'environment.json', 'w') as f:
                json.dump(env_info, f, indent=2)

            print()
            print(f"Run saved to: {run_path}")
            if args.quick:
                print("WARNING: This is a quick-mode result. Don't report these numbers!")

    if args.train:
        if not args.output:
            parser.error("--output required when using --train")

        # Build combiner from CLI args
        train_combiner = ThresholdCombiner(
            primary_model='cnn',
            agreement_transform=args.acceptance_formula,
            disagreement_scale=args.scale,
            threshold=args.threshold,
        )

        print("\nTraining final models...")
        print(f"  combiner: {train_combiner}")

        pipeline = BoomDetectionPipeline(
            combiner=train_combiner,
            calibrate_quality=True,
            feature_config=config,
        )
        pipeline.fit(sim_ids, boom_frames, qualities, cache)

        # Save models
        pipeline.save(args.output)
        print(f"\nModels saved to {args.output}")

    if args.predict:
        # Inference mode: load pretrained models and predict
        print(f"\nLoading models from {args.predict}...")
        pipeline = BoomDetectionPipeline.from_pretrained(args.predict)

        print("Running predictions...")
        predictions = pipeline.predict(sim_ids, cache)

        # Output results
        n_accepted = sum(1 for p in predictions if p.accepted)
        print(f"\nPredictions: {n_accepted}/{len(predictions)} accepted")

        # Write predictions to JSONL
        output_file = args.predict / 'predictions.jsonl'
        with open(output_file, 'w') as f:
            for sid, pred in zip(sim_ids, predictions):
                result = pred.to_dict()
                result['sim_id'] = sid
                result['true_boom'] = int(boom_frames[sim_ids.index(sid)])
                result['true_quality'] = float(qualities[sim_ids.index(sid)])
                f.write(json.dumps(result) + '\n')

        print(f"Predictions written to {output_file}")

        # Compute and print metrics
        from .metrics import compute_selective_metrics
        metrics = compute_selective_metrics(predictions, boom_frames, qualities)
        print("\nMetrics:")
        print(f"  Selective MAE: {metrics['selective_mae']:.2f} frames")
        print(f"  Coverage: {metrics['coverage']:.1%}")
        print(f"  Within 5 frames: {metrics['selective_within_5']:.1%}")


if __name__ == '__main__':
    main()
