"""
Deployable boom detection pipeline.

This implements the best-performing approach:
1. Run CNN and HistGBM in parallel
2. Check if they agree (within threshold)
3. Predict quality using features around predicted boom
4. If both filters pass → use CNN prediction (more accurate than HGB)

Performance (5-fold CV × 5 seeds):
- MAE: 7.1 ± 0.7 frames (on accepted simulations)
- Acceptance rate: ~29%
- Within 5 frames: ~50%

Note: CNN alone is more accurate than HGB when models agree.
HGB is only used as a confidence filter (agreement check).

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
            'boom_frame': cnn_pred if accepted else None,  # CNN is more accurate
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
    seeds: list[int] | None = None,
    agreement_threshold: int = 5,
    quality_threshold: float = 0.55,
    verbose: bool = True,
) -> dict:
    """
    Run robust multi-seed cross-validation.

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
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]  # 5 seeds by default

    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nSeed {seed} ({seed_idx + 1}/{len(seeds)})")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        seed_results = []

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

            for res, true_boom, true_qual in zip(results, test_booms, test_quals):
                res['true_boom'] = int(true_boom)
                res['true_quality'] = float(true_qual)
                seed_results.append(res)

        # Compute metrics for this seed
        accepted = [r for r in seed_results if r['accepted']]
        if accepted:
            errors = [abs(r['cnn_pred'] - r['true_boom']) for r in accepted]
            seed_mae = np.mean(errors)
            seed_within5 = np.mean([e <= 5 for e in errors]) * 100
            seed_acceptance = len(accepted) / len(seed_results) * 100
        else:
            seed_mae = float('nan')
            seed_within5 = float('nan')
            seed_acceptance = 0.0

        all_seed_results.append({
            'mae': seed_mae,
            'within_5': seed_within5,
            'acceptance_rate': seed_acceptance,
            'n_accepted': len(accepted),
        })

        if verbose:
            print(f"  MAE: {seed_mae:.2f}, Accepted: {len(accepted)}/{len(seed_results)}")

    # Aggregate across seeds
    maes = [r['mae'] for r in all_seed_results if not np.isnan(r['mae'])]
    within5s = [r['within_5'] for r in all_seed_results if not np.isnan(r['within_5'])]
    acceptances = [r['acceptance_rate'] for r in all_seed_results]

    return {
        'n_seeds': len(seeds),
        'n_splits': n_splits,
        'seeds': seeds,
        # Main metrics with uncertainty
        'mae_mean': float(np.mean(maes)) if maes else float('nan'),
        'mae_std': float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        'within_5_mean': float(np.mean(within5s)) if within5s else float('nan'),
        'within_5_std': float(np.std(within5s, ddof=1)) if len(within5s) > 1 else 0.0,
        'acceptance_rate_mean': float(np.mean(acceptances)),
        'acceptance_rate_std': float(np.std(acceptances, ddof=1)) if len(acceptances) > 1 else 0.0,
        # Per-seed results for analysis
        'seed_results': all_seed_results,
        # Config
        'agreement_threshold': agreement_threshold,
        'quality_threshold': quality_threshold,
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
            r = results['seed_results'][0]
            print(f"MAE: {r['mae']:.2f} frames")
            print(f"Within 5 frames: {r['within_5']:.1f}%")
            print(f"Acceptance rate: {r['acceptance_rate']:.1f}%")
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
