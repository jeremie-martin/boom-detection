"""
Deployable boom detection pipeline.

This implements the best-performing approach:
1. Run CNN and HistGBM in parallel
2. Check if they agree (within threshold)
3. Predict quality using features around predicted boom
4. If both filters pass → use CNN prediction (more accurate than HGB)

Performance (5-fold CV × 5 seeds):
- MAE: 6.5 ± 0.3 frames (on accepted simulations)
- Acceptance rate: ~33%
- Within 5 frames: ~60%

Key improvements from ablation/tuning:
- CNN prediction (not HGB) - more accurate when models agree
- Random Forest for quality (not Ridge) - better correlation
- Top 50 quality features with smaller window (±25) - less overfitting
- Larger CNN kernels (5,11,21) - capture longer-range patterns
- hidden_dim=64 - more capacity without overfitting

Usage:
    # Evaluate with robust multi-seed CV
    uv run python -m boom_detection.deploy_pipeline data --evaluate

    # Quick single-seed evaluation (for development)
    uv run python -m boom_detection.deploy_pipeline data --evaluate --quick

    # Train final models and save
    uv run python -m boom_detection.deploy_pipeline data --train --output models/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor

from .loader import load_dataset
from .features import FeatureCache, FeatureConfig, PRODUCTION_CONFIG
from .evaluation import SelectivePrediction
from .sequence_models import CNNClassifier, SequenceTrainer


class BoomDetectionPipeline:
    """
    Production-ready boom detection pipeline.

    Uses model agreement + predicted quality as confidence filters.
    """

    def __init__(
        self,
        agreement_threshold: int = 5,
        quality_threshold: float = 0.55,
        quality_window: int = 25,  # Smaller window works better
        n_quality_features: int = 50,  # Top correlated features
        seed: int | None = None,  # Random seed for reproducibility
        calibrate_quality: bool = True,  # Calibrate quality predictions
    ):
        self.agreement_threshold = agreement_threshold
        self.quality_threshold = quality_threshold
        self.quality_window = quality_window
        self.n_quality_features = n_quality_features
        self.seed = seed
        self.calibrate_quality = calibrate_quality

        # Models (set during training)
        self.cnn = None
        self.cnn_trainer = None
        self.hgb = None
        self.quality_model = None
        self.quality_calibrator = None  # Optional isotonic regression calibrator
        self.quality_feature_indices = None  # Top features for quality
        self.n_features = None

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self.seed = seed

    def fit(
        self,
        sim_ids: list[str],
        boom_frames: np.ndarray,
        qualities: np.ndarray,
        cache: FeatureCache,
    ) -> None:
        """Train all models on the given data."""
        # Get feature count
        self.n_features = cache[sim_ids[0]].shape[1]

        # Derive seeds for each model component (deterministic from self.seed)
        cnn_seed = self.seed
        hgb_seed = (self.seed + 1000) if self.seed is not None else 42
        quality_seed = (self.seed + 2000) if self.seed is not None else 42

        # Train CNN with optimized architecture
        # Larger kernels (5,11,21) capture longer-range temporal patterns
        # hidden_dim=64 gives more capacity without overfitting
        self.cnn = CNNClassifier(
            n_features=self.n_features,
            hidden_dim=64,
            kernel_sizes=(5, 11, 21)
        )
        self.cnn_trainer = SequenceTrainer(
            self.cnn, lr=0.5e-3, epochs=30, patience=5, batch_size=4, augment=False,
            seed=cnn_seed,
        )
        self.cnn_trainer.fit(sim_ids, boom_frames, cache)

        # Train HistGBM
        X_train, y_train = [], []
        for sid, boom in zip(sim_ids, boom_frames):
            feats = cache[sid]
            for t in range(len(feats)):
                X_train.append(feats[t])
                y_train.append(1 if t >= boom else 0)

        self.hgb = HistGradientBoostingClassifier(
            max_iter=200, max_depth=7, random_state=hgb_seed
        )
        self.hgb.fit(np.array(X_train), np.array(y_train))

        # Train quality predictor with feature selection
        # IMPORTANT: Add jitter to boom frame during training to simulate prediction noise
        # This prevents train/inference mismatch since at inference we use predicted boom
        jitter_std = 5  # Standard deviation of jitter (typical CNN/HGB error is ~5-10 frames)
        rng = np.random.RandomState(quality_seed)

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
            n_estimators=50, max_depth=5, random_state=quality_seed
        )
        self.quality_model.fit(X_qual_selected, qualities)

        # Optional: Calibrate quality predictions using isotonic regression
        # This makes the predicted quality match the empirical probability
        if self.calibrate_quality:
            from sklearn.isotonic import IsotonicRegression

            # Get out-of-bag predictions for calibration
            # Use leave-one-out since we have few samples
            raw_predictions = np.zeros(len(qualities))
            for i in range(len(X_qual_selected)):
                # Train on all except i
                mask = np.ones(len(X_qual_selected), dtype=bool)
                mask[i] = False
                temp_model = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=quality_seed
                )
                temp_model.fit(X_qual_selected[mask], qualities[mask])
                raw_predictions[i] = temp_model.predict([X_qual_selected[i]])[0]

            # Fit isotonic regression calibrator
            self.quality_calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip'
            )
            self.quality_calibrator.fit(raw_predictions, qualities)

    def predict_one(self, features: np.ndarray) -> SelectivePrediction:
        """
        Predict boom frame for a single simulation.

        Args:
            features: Shape (frames, n_features)

        Returns:
            SelectivePrediction with:
                - boom_frame: Predicted boom frame (or None if rejected)
                - accepted: Whether simulation passed filters
                - cnn_pred: CNN prediction
                - hgb_pred: HGB prediction
                - disagreement: |CNN - HGB|
                - predicted_quality: Quality prediction
                - confidence: Combined confidence score

        Raises:
            TypeError: If features is not a numpy array
            ValueError: If features has wrong shape
            RuntimeError: If pipeline has not been fitted
        """
        import torch

        # Input validation
        if self.cnn is None or self.hgb is None:
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

        # CNN prediction
        self.cnn.eval()
        with torch.no_grad():
            feats_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
            feats_t = feats_t.to(self.cnn_trainer.device)
            logits = self.cnn(feats_t)
            probs_cnn = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        crossings = np.where(probs_cnn >= 0.5)[0]
        cnn_pred = int(crossings[0]) if len(crossings) > 0 else int(np.argmax(probs_cnn))

        # HGB prediction
        probs_hgb = self.hgb.predict_proba(features)[:, 1]
        crossings = np.where(probs_hgb >= 0.5)[0]
        hgb_pred = int(crossings[0]) if len(crossings) > 0 else int(np.argmax(probs_hgb))

        # Check agreement
        disagreement = abs(cnn_pred - hgb_pred)

        # Predict quality (using average of predictions as boom estimate)
        avg_pred = int((cnn_pred + hgb_pred) / 2)
        start = max(0, avg_pred - self.quality_window)
        end = min(len(features), avg_pred + self.quality_window)
        window_feats = features[start:end].mean(axis=0)
        # Use selected features
        window_feats_selected = window_feats[self.quality_feature_indices]
        raw_quality = float(self.quality_model.predict([window_feats_selected])[0])

        # Apply calibration if available
        if self.quality_calibrator is not None:
            predicted_quality = float(np.clip(
                self.quality_calibrator.predict([raw_quality])[0], 0, 1
            ))
        else:
            predicted_quality = float(np.clip(raw_quality, 0, 1))

        # Apply filters
        accepted = (
            disagreement <= self.agreement_threshold and
            predicted_quality >= self.quality_threshold
        )

        # Compute confidence score (higher = more confident)
        # Normalize disagreement to [0, 1] and combine with quality
        agreement_score = 1.0 - min(disagreement / 10.0, 1.0)  # 0-10 frames -> 1-0
        confidence = (agreement_score + predicted_quality) / 2.0

        return SelectivePrediction(
            boom_frame=cnn_pred if accepted else None,  # CNN is more accurate
            accepted=accepted,
            cnn_pred=cnn_pred,
            hgb_pred=hgb_pred,
            disagreement=disagreement,
            predicted_quality=predicted_quality,
            confidence=confidence,
        )

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> list[SelectivePrediction]:
        """Predict for multiple simulations."""
        return [self.predict_one(cache[sid]) for sid in sim_ids]

    def save(self, path: Path) -> None:
        """
        Save the pipeline to a directory.

        Creates:
            path/cnn.pt - CNN model weights
            path/hgb.joblib - HistGBM model
            path/quality.joblib - Quality model
            path/config.json - Pipeline configuration

        Args:
            path: Directory to save to
        """
        import torch
        import joblib

        if self.cnn is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save CNN with exact architecture parameters
        torch.save({
            'state_dict': self.cnn.state_dict(),
            'n_features': self.cnn.n_features,
            'hidden_dim': self.cnn.hidden_dim,
            'kernel_sizes': self.cnn.kernel_sizes,
            'dropout': self.cnn.dropout,
        }, path / 'cnn.pt')

        # Save HGB and quality model
        joblib.dump(self.hgb, path / 'hgb.joblib')
        joblib.dump(self.quality_model, path / 'quality.joblib')
        if self.quality_calibrator is not None:
            joblib.dump(self.quality_calibrator, path / 'quality_calibrator.joblib')

        # Save config
        config = {
            'agreement_threshold': self.agreement_threshold,
            'quality_threshold': self.quality_threshold,
            'quality_window': self.quality_window,
            'n_quality_features': self.n_quality_features,
            'n_features': self.n_features,
            'quality_feature_indices': self.quality_feature_indices,
            'calibrate_quality': self.calibrate_quality,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: Path, device: str = 'auto') -> 'BoomDetectionPipeline':
        """
        Load a pretrained pipeline from a directory.

        Args:
            path: Directory containing saved models
            device: Device for CNN ('auto', 'cuda', 'cpu')

        Returns:
            Loaded BoomDetectionPipeline ready for inference
        """
        import torch
        import joblib

        path = Path(path)

        # Load config
        with open(path / 'config.json') as f:
            config = json.load(f)

        # Create pipeline with saved config
        pipeline = cls(
            agreement_threshold=config['agreement_threshold'],
            quality_threshold=config['quality_threshold'],
            quality_window=config['quality_window'],
            n_quality_features=config['n_quality_features'],
            calibrate_quality=config.get('calibrate_quality', False),
        )

        pipeline.n_features = config['n_features']
        pipeline.quality_feature_indices = config['quality_feature_indices']

        # Load CNN with exact saved architecture
        checkpoint = torch.load(path / 'cnn.pt', map_location='cpu', weights_only=True)
        pipeline.cnn = CNNClassifier(
            n_features=checkpoint['n_features'],
            hidden_dim=checkpoint['hidden_dim'],
            kernel_sizes=tuple(checkpoint['kernel_sizes']),
            dropout=checkpoint.get('dropout', 0.3),
        )
        pipeline.cnn.load_state_dict(checkpoint['state_dict'])

        # Set up device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline.cnn = pipeline.cnn.to(device)
        pipeline.cnn.eval()

        # Create minimal trainer for device tracking
        pipeline.cnn_trainer = SequenceTrainer(pipeline.cnn, device=device)

        # Load sklearn models
        pipeline.hgb = joblib.load(path / 'hgb.joblib')
        pipeline.quality_model = joblib.load(path / 'quality.joblib')

        # Load calibrator if available
        calibrator_path = path / 'quality_calibrator.joblib'
        if calibrator_path.exists():
            pipeline.quality_calibrator = joblib.load(calibrator_path)

        return pipeline


def cross_validate(
    sim_ids: list[str],
    boom_frames: np.ndarray,
    qualities: np.ndarray,
    cache: FeatureCache,
    n_splits: int = 5,
    seeds: list[int] | None = None,
    agreement_threshold: int = 5,
    quality_threshold: float = 0.55,
    verbose: bool = True,
) -> dict:
    """
    Run robust multi-seed cross-validation using the unified evaluation framework.

    This is a convenience wrapper around CachedEvaluator.cross_validate_selective()
    that returns a backward-compatible dict format.

    For new code, prefer using CachedEvaluator directly:
        evaluator = CachedEvaluator(dataset, cache)
        result = evaluator.cross_validate_selective(
            lambda: BoomDetectionPipeline(agreement_threshold=5),
        )

    Args:
        sim_ids: List of simulation IDs
        boom_frames: Ground truth boom frames
        qualities: Ground truth quality scores
        cache: FeatureCache with extracted features
        n_splits: Number of CV folds
        seeds: Random seeds (default: 5 seeds for robust evaluation)
        agreement_threshold: Max allowed disagreement between CNN and HGB
        quality_threshold: Min predicted quality to accept
        verbose: Print progress

    Returns:
        Dict with metrics including mean ± std across seeds
    """
    from .evaluation import CachedEvaluator, MultiSeedSelectiveResult

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]  # 5 seeds by default

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

    # Use the unified evaluator
    result: MultiSeedSelectiveResult = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            agreement_threshold=agreement_threshold,
            quality_threshold=quality_threshold,
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
        'agreement_threshold': agreement_threshold,
        'quality_threshold': quality_threshold,
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
    parser.add_argument('--agreement', type=int, default=5, help='Agreement threshold')
    parser.add_argument('--quality', type=float, default=0.55, help='Quality threshold')
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

        print(f"\nRunning {mode} 5-fold cross-validation...")
        print(f"Agreement threshold: {args.agreement}")
        print(f"Quality threshold: {args.quality}")

        results = cross_validate(
            sim_ids, boom_frames, qualities, cache,
            seeds=seeds,
            agreement_threshold=args.agreement,
            quality_threshold=args.quality,
        )

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        if args.quick:
            # Single seed - just show the values
            r = results['seed_metrics'][0]
            print(f"MAE: {r['selective_mae']:.2f} frames")
            print(f"Within 5 frames: {r['selective_within_5']*100:.1f}%")
            print(f"Acceptance rate: {r['coverage']*100:.1f}%")
            print()
            print("Note: This is a quick single-seed result. For robust evaluation,")
            print("      run without --quick to get mean ± std across 5 seeds.")
        else:
            # Multi-seed - show mean ± std
            print(f"MAE: {results['mae_mean']:.2f} ± {results['mae_std']:.2f} frames")
            print(f"Within 5 frames: {results['within_5_mean']:.1f}% ± {results['within_5_std']:.1f}%")
            print(f"Acceptance rate: {results['acceptance_rate_mean']:.1f}% ± {results['acceptance_rate_std']:.1f}%")
            print()
            print(f"Based on {results['n_seeds']} random seeds × {results['n_splits']}-fold CV")

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
                'agreement_threshold': args.agreement,
                'quality_threshold': args.quality,
                'production_features': args.production,
                'n_splits': results['n_splits'],
                'seeds': results['seeds'],
                'n_seeds': results['n_seeds'],
            }

            # Save metrics and config
            run_path.mkdir(parents=True, exist_ok=True)

            with open(run_path / 'config.json', 'w') as f:
                json.dump(run_config, f, indent=2)

            # Save aggregated metrics
            metrics_data = {
                'mae_mean': results['mae_mean'],
                'mae_std': results['mae_std'],
                'within_5_mean': results['within_5_mean'],
                'within_5_std': results['within_5_std'],
                'acceptance_rate_mean': results['acceptance_rate_mean'],
                'acceptance_rate_std': results['acceptance_rate_std'],
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

        print("\nTraining final models...")
        pipeline = BoomDetectionPipeline(
            agreement_threshold=args.agreement,
            quality_threshold=args.quality,
        )
        pipeline.fit(sim_ids, boom_frames, qualities, cache)

        # Save models using the new save method
        pipeline.save(args.output)
        print(f"Models saved to {args.output}")

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
        from .evaluation import compute_selective_metrics
        metrics = compute_selective_metrics(predictions, boom_frames, qualities)
        print("\nMetrics:")
        print(f"  Selective MAE: {metrics['selective_mae']:.2f} frames")
        print(f"  Coverage: {metrics['coverage']:.1%}")
        print(f"  Within 5 frames: {metrics['selective_within_5']:.1%}")


if __name__ == '__main__':
    main()
