#!/usr/bin/env python3
"""
C1: Quality Model Alternatives.

Compare different quality prediction models:
- HistGradientBoostingRegressor (default)
- Ridge regression
- RandomForest with different configurations

Usage:
    uv run python scripts/c1_quality_models.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def extract_quality_features(features: np.ndarray, boom_frame: int, window: int = 50) -> np.ndarray:
    """Extract features around boom frame for quality prediction."""
    n_frames = features.shape[0]
    start = max(0, boom_frame - window)
    end = min(n_frames, boom_frame + window)
    window_feats = features[start:end]

    # Aggregate to fixed-size feature vector
    return np.concatenate([
        window_feats.mean(axis=0),
        window_feats.std(axis=0),
        window_feats.max(axis=0),
        np.percentile(window_feats, 90, axis=0),
        np.percentile(window_feats, 10, axis=0),
    ])


def main():
    parser = argparse.ArgumentParser(description='C1: Quality Model Alternatives')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c1_quality_models")

    logger.info("=" * 70)
    logger.info("C1: Quality Model Alternatives")
    logger.info("=" * 70)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    cache = FeatureCache(config=PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    boom_qualities = np.array([a.boom_quality for a in dataset.annotations])

    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    logger.info(f"Testing quality models on {len(sim_ids)} simulations")

    # Extract quality features for all simulations
    logger.info("Extracting quality features...")
    X_all = []
    for i, sim_id in enumerate(sim_ids):
        features = cache.get(sim_id)
        bf = int(boom_frames[i])
        qual_feats = extract_quality_features(features, bf)
        X_all.append(qual_feats)
    X_all = np.array(X_all)
    logger.info(f"Quality feature matrix: {X_all.shape}")

    # Quality models to test
    models = {
        'HistGBM (default)': lambda seed: HistGradientBoostingRegressor(
            max_iter=100, max_depth=5, random_state=seed
        ),
        'Ridge (alpha=1.0)': lambda seed: Ridge(alpha=1.0),
        'RF (n=100, d=None)': lambda seed: RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1
        ),
        'RF (n=100, d=7)': lambda seed: RandomForestRegressor(
            n_estimators=100, max_depth=7, random_state=seed, n_jobs=-1
        ),
        'RF (n=50, d=5)': lambda seed: RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1
        ),
    }

    results = {}

    for model_name, model_factory in models.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {model_name}")
        logger.info("=" * 70)

        all_predictions = []
        all_true = []
        seed_metrics = []

        for seed in args.seeds:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)

            fold_preds = []
            fold_true = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
                X_train = X_all[train_idx]
                y_train = boom_qualities[train_idx]
                X_test = X_all[test_idx]
                y_test = boom_qualities[test_idx]

                # Train model
                model = model_factory(seed * 1000 + fold_idx)
                model.fit(X_train, y_train)

                # Predict
                preds = model.predict(X_test)
                preds = np.clip(preds, 0, 1)

                fold_preds.extend(preds)
                fold_true.extend(y_test)

            # Compute metrics for this seed
            fold_preds = np.array(fold_preds)
            fold_true = np.array(fold_true)

            mae = np.mean(np.abs(fold_preds - fold_true))
            rmse = np.sqrt(np.mean((fold_preds - fold_true) ** 2))
            r, _ = stats.spearmanr(fold_preds, fold_true)

            seed_metrics.append({'mae': mae, 'rmse': rmse, 'spearman_r': r})
            all_predictions.extend(fold_preds)
            all_true.extend(fold_true)

            logger.info(f"  Seed {seed}: MAE={mae:.4f}, RMSE={rmse:.4f}, r={r:.3f}")

        # Aggregate metrics
        mean_mae = np.mean([m['mae'] for m in seed_metrics])
        std_mae = np.std([m['mae'] for m in seed_metrics])
        mean_rmse = np.mean([m['rmse'] for m in seed_metrics])
        mean_r = np.mean([m['spearman_r'] for m in seed_metrics])

        results[model_name] = {
            'mae': mean_mae,
            'mae_std': std_mae,
            'rmse': mean_rmse,
            'spearman_r': mean_r,
        }

        logger.info(f"  Overall: MAE={mean_mae:.4f} +/- {std_mae:.4f}, r={mean_r:.3f}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Model':<25} {'MAE':<12} {'RMSE':<12} {'Spearman r':<12}")
    logger.info("-" * 61)
    for model_name, r in sorted(results.items(), key=lambda x: x[1]['mae']):
        logger.info(f"{model_name:<25} {r['mae']:.4f}±{r['mae_std']:.4f} {r['rmse']:.4f}    {r['spearman_r']:.3f}")

    best_model = min(results, key=lambda x: results[x]['mae'])
    logger.info(f"\nBest model: {best_model}")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.output / 'c1_quality_models.csv'
    with open(csv_path, 'w') as f:
        f.write("model,mae,mae_std,rmse,spearman_r\n")
        for model_name, r in results.items():
            f.write(f"{model_name},{r['mae']:.4f},{r['mae_std']:.4f},{r['rmse']:.4f},{r['spearman_r']:.3f}\n")
    logger.info(f"CSV saved to: {csv_path}")

    # Report
    report_path = args.output / 'c1_quality_models.md'
    with open(report_path, 'w') as f:
        f.write("# C1: Quality Model Alternatives\n\n")
        f.write("## Objective\n")
        f.write("Compare different quality prediction models to find the best one.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(sim_ids)} simulations\n")
        f.write(f"- Quality window: 50 frames\n")
        f.write("- 5-fold cross-validation\n\n")

        f.write("## Results\n\n")
        f.write("| Model | MAE | RMSE | Spearman r |\n")
        f.write("|-------|-----|------|------------|\n")
        for model_name, r in sorted(results.items(), key=lambda x: x[1]['mae']):
            f.write(f"| {model_name} | {r['mae']:.4f} +/- {r['mae_std']:.4f} | {r['rmse']:.4f} | {r['spearman_r']:.3f} |\n")

        f.write(f"\n## Best Model\n\n**{best_model}**\n\n")

        f.write("## Analysis\n\n")
        best_mae = results[best_model]['mae']
        default_mae = results.get('HistGBM (default)', {}).get('mae', float('inf'))
        rf_default = results.get('RF (n=100, d=None)', {}).get('mae', float('inf'))

        if best_mae < default_mae - 0.01:
            f.write(f"**{best_model}** outperforms the default HistGBM by {default_mae - best_mae:.4f} MAE.\n\n")
            f.write("**Recommendation:** Consider switching to this model.\n")
        else:
            f.write("All models perform similarly. The default HistGBM is a good choice.\n\n")
            f.write("**Recommendation:** Keep the current default model.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
