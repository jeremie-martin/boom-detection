"""
Deployable boom detection pipeline.

This implements the best-performing approach:
1. Run CNN and HistGBM in parallel
2. Check if they agree (within threshold)
3. Predict quality using features around predicted boom
4. If both filters pass → use HGB prediction

Best configuration:
- Agreement threshold: ≤5 frames
- Predicted quality threshold: ≥0.55
- Use HGB prediction (not average)
- Result: MAE 4.0, 77% within 5 frames, 27% acceptance rate

Usage:
    # Evaluate on dataset with cross-validation
    uv run python -m boom_detection.deploy_pipeline data --evaluate

    # Train final models and save
    uv run python -m boom_detection.deploy_pipeline data --train --output models/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from .loader import load_dataset
from .features import FeatureCache, FeatureConfig
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
        quality_window: int = 50,
    ):
        self.agreement_threshold = agreement_threshold
        self.quality_threshold = quality_threshold
        self.quality_window = quality_window

        # Models (set during training)
        self.cnn = None
        self.cnn_trainer = None
        self.hgb = None
        self.quality_model = None
        self.n_features = None

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

        # Train CNN
        self.cnn = CNNClassifier(n_features=self.n_features, hidden_dim=32)
        self.cnn_trainer = SequenceTrainer(
            self.cnn, lr=0.5e-3, epochs=30, patience=5, batch_size=4, augment=False
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
            max_iter=200, max_depth=7, random_state=42
        )
        self.hgb.fit(np.array(X_train), np.array(y_train))

        # Train quality predictor
        X_qual = []
        for sid, boom in zip(sim_ids, boom_frames):
            feats = cache[sid]
            start = max(0, int(boom) - self.quality_window)
            end = min(len(feats), int(boom) + self.quality_window)
            X_qual.append(feats[start:end].mean(axis=0))

        self.quality_model = Ridge(alpha=1.0)
        self.quality_model.fit(np.array(X_qual), qualities)

    def predict_one(self, features: np.ndarray) -> dict:
        """
        Predict boom frame for a single simulation.

        Args:
            features: Shape (frames, n_features)

        Returns:
            dict with keys:
                - boom_frame: Predicted boom frame (or None if rejected)
                - accepted: Whether simulation passed filters
                - cnn_pred: CNN prediction
                - hgb_pred: HGB prediction
                - disagreement: |CNN - HGB|
                - predicted_quality: Quality prediction
        """
        import torch

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
        predicted_quality = float(np.clip(
            self.quality_model.predict([window_feats])[0], 0, 1
        ))

        # Apply filters
        accepted = (
            disagreement <= self.agreement_threshold and
            predicted_quality >= self.quality_threshold
        )

        return {
            'boom_frame': hgb_pred if accepted else None,
            'accepted': accepted,
            'cnn_pred': cnn_pred,
            'hgb_pred': hgb_pred,
            'disagreement': disagreement,
            'predicted_quality': predicted_quality,
        }

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> list[dict]:
        """Predict for multiple simulations."""
        return [self.predict_one(cache[sid]) for sid in sim_ids]


def cross_validate(
    sim_ids: list[str],
    boom_frames: np.ndarray,
    qualities: np.ndarray,
    cache: FeatureCache,
    n_splits: int = 5,
    agreement_threshold: int = 5,
    quality_threshold: float = 0.55,
) -> dict:
    """
    Run cross-validation and return metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []
    all_true = []
    all_qualities = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(sim_ids)):
        train_ids = [sim_ids[i] for i in train_idx]
        test_ids = [sim_ids[i] for i in test_idx]
        train_booms = boom_frames[train_idx]
        test_booms = boom_frames[test_idx]
        train_quals = qualities[train_idx]
        test_quals = qualities[test_idx]

        # Train pipeline
        pipeline = BoomDetectionPipeline(
            agreement_threshold=agreement_threshold,
            quality_threshold=quality_threshold,
        )
        pipeline.fit(train_ids, train_booms, train_quals, cache)

        # Predict
        results = pipeline.predict(test_ids, cache)

        for i, (res, true_boom, true_qual) in enumerate(zip(results, test_booms, test_quals)):
            res['true_boom'] = int(true_boom)
            res['true_quality'] = float(true_qual)
            all_results.append(res)
            all_true.append(true_boom)
            all_qualities.append(true_qual)

        print(f"  Fold {fold + 1}/{n_splits} complete")

    # Compute metrics
    accepted = [r for r in all_results if r['accepted']]
    rejected = [r for r in all_results if not r['accepted']]

    if accepted:
        errors = [abs(r['hgb_pred'] - r['true_boom']) for r in accepted]
        mae = np.mean(errors)
        within5 = np.mean([e <= 5 for e in errors]) * 100
        within3 = np.mean([e <= 3 for e in errors]) * 100
    else:
        mae = within5 = within3 = float('nan')

    return {
        'n_total': len(all_results),
        'n_accepted': len(accepted),
        'n_rejected': len(rejected),
        'acceptance_rate': len(accepted) / len(all_results) * 100,
        'mae': mae,
        'within_5': within5,
        'within_3': within3,
        'agreement_threshold': agreement_threshold,
        'quality_threshold': quality_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description='Boom detection pipeline')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--evaluate', action='store_true', help='Run cross-validation')
    parser.add_argument('--train', action='store_true', help='Train and save models')
    parser.add_argument('--output', type=Path, help='Output directory for models')
    parser.add_argument('--agreement', type=int, default=5, help='Agreement threshold')
    parser.add_argument('--quality', type=float, default=0.55, help='Quality threshold')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    dataset = load_dataset(args.data_path)

    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    # Build feature cache
    print("Building feature cache...")
    config = FeatureConfig()
    cache = FeatureCache(config, cache_dir='.feature_cache')
    cache.build(dataset)

    if args.evaluate:
        print(f"\nRunning 5-fold cross-validation...")
        print(f"Agreement threshold: {args.agreement}")
        print(f"Quality threshold: {args.quality}")

        results = cross_validate(
            sim_ids, boom_frames, qualities, cache,
            agreement_threshold=args.agreement,
            quality_threshold=args.quality,
        )

        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Total simulations: {results['n_total']}")
        print(f"Accepted: {results['n_accepted']} ({results['acceptance_rate']:.1f}%)")
        print(f"Rejected: {results['n_rejected']}")
        print(f"\nOn accepted simulations:")
        print(f"  MAE: {results['mae']:.2f} frames")
        print(f"  Within 5 frames: {results['within_5']:.1f}%")
        print(f"  Within 3 frames: {results['within_3']:.1f}%")

    if args.train:
        if not args.output:
            parser.error("--output required when using --train")

        print(f"\nTraining final models...")
        pipeline = BoomDetectionPipeline(
            agreement_threshold=args.agreement,
            quality_threshold=args.quality,
        )
        pipeline.fit(sim_ids, boom_frames, qualities, cache)

        # Save models
        args.output.mkdir(parents=True, exist_ok=True)
        import torch
        import joblib

        torch.save(pipeline.cnn.state_dict(), args.output / 'cnn.pt')
        joblib.dump(pipeline.hgb, args.output / 'hgb.joblib')
        joblib.dump(pipeline.quality_model, args.output / 'quality.joblib')

        # Save config
        config_dict = {
            'agreement_threshold': pipeline.agreement_threshold,
            'quality_threshold': pipeline.quality_threshold,
            'quality_window': pipeline.quality_window,
            'n_features': pipeline.n_features,
        }
        with open(args.output / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Models saved to {args.output}")


if __name__ == '__main__':
    main()
