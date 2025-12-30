#!/usr/bin/env python3
"""
C2: Asymmetric Quality Window.

Test whether asymmetric windows around boom improve quality prediction.
The hypothesis is that frames AFTER the boom may be more informative than before.

Usage:
    uv run python scripts/c2_asymmetric_window.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def extract_quality_features_asymmetric(
    features: np.ndarray,
    boom_frame: int,
    window_before: int,
    window_after: int,
) -> np.ndarray:
    """Extract features around boom frame with asymmetric window."""
    n_frames = features.shape[0]
    start = max(0, boom_frame - window_before)
    end = min(n_frames, boom_frame + window_after)
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
    parser = argparse.ArgumentParser(description='C2: Asymmetric Quality Window')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c2_asymmetric_window")

    logger.info("=" * 70)
    logger.info("C2: Asymmetric Quality Window")
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

    logger.info(f"Testing window configurations on {len(sim_ids)} simulations")

    # Window configurations to test
    window_configs = [
        ('Symmetric 50/50', 50, 50),
        ('Symmetric 25/25', 25, 25),
        ('After-heavy 25/75', 25, 75),
        ('After-heavy 15/85', 15, 85),
        ('Before-heavy 75/25', 75, 25),
        ('Before-heavy 85/15', 85, 15),
        ('Narrow 10/10', 10, 10),
        ('Wide 100/100', 100, 100),
    ]

    results = {}

    for config_name, window_before, window_after in window_configs:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {config_name} (before={window_before}, after={window_after})")
        logger.info("=" * 70)

        # Extract features with this window config
        X_all = []
        for i, sim_id in enumerate(sim_ids):
            features = cache.get(sim_id)
            bf = int(boom_frames[i])
            qual_feats = extract_quality_features_asymmetric(
                features, bf, window_before, window_after
            )
            X_all.append(qual_feats)
        X_all = np.array(X_all)

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
                model = HistGradientBoostingRegressor(
                    max_iter=100, max_depth=5, random_state=seed * 1000 + fold_idx
                )
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

            logger.info(f"  Seed {seed}: MAE={mae:.4f}, RMSE={rmse:.4f}, r={r:.3f}")

        # Aggregate metrics
        mean_mae = np.mean([m['mae'] for m in seed_metrics])
        std_mae = np.std([m['mae'] for m in seed_metrics])
        mean_rmse = np.mean([m['rmse'] for m in seed_metrics])
        mean_r = np.mean([m['spearman_r'] for m in seed_metrics])

        results[config_name] = {
            'window_before': window_before,
            'window_after': window_after,
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

    logger.info(f"\n{'Config':<25} {'Window':<12} {'MAE':<15} {'Spearman r':<12}")
    logger.info("-" * 64)
    for config_name, r in sorted(results.items(), key=lambda x: x[1]['mae']):
        window_str = f"{r['window_before']}/{r['window_after']}"
        logger.info(f"{config_name:<25} {window_str:<12} {r['mae']:.4f}±{r['mae_std']:.4f}  {r['spearman_r']:.3f}")

    best_config = min(results, key=lambda x: results[x]['mae'])
    logger.info(f"\nBest config: {best_config}")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.output / 'c2_asymmetric_window.csv'
    with open(csv_path, 'w') as f:
        f.write("config,window_before,window_after,mae,mae_std,rmse,spearman_r\n")
        for config_name, r in results.items():
            f.write(f"{config_name},{r['window_before']},{r['window_after']},{r['mae']:.4f},{r['mae_std']:.4f},{r['rmse']:.4f},{r['spearman_r']:.3f}\n")
    logger.info(f"CSV saved to: {csv_path}")

    # Report
    report_path = args.output / 'c2_asymmetric_window.md'
    with open(report_path, 'w') as f:
        f.write("# C2: Asymmetric Quality Window\n\n")
        f.write("## Objective\n")
        f.write("Test whether asymmetric windows around boom improve quality prediction.\n")
        f.write("Hypothesis: frames after the boom may be more informative.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(sim_ids)} simulations\n")
        f.write("- Model: HistGradientBoostingRegressor\n")
        f.write("- 5-fold cross-validation\n\n")

        f.write("## Results\n\n")
        f.write("| Config | Window (before/after) | MAE | RMSE | Spearman r |\n")
        f.write("|--------|----------------------|-----|------|------------|\n")
        for config_name, r in sorted(results.items(), key=lambda x: x[1]['mae']):
            f.write(f"| {config_name} | {r['window_before']}/{r['window_after']} | {r['mae']:.4f} +/- {r['mae_std']:.4f} | {r['rmse']:.4f} | {r['spearman_r']:.3f} |\n")

        f.write(f"\n## Best Configuration\n\n**{best_config}**\n\n")

        f.write("## Analysis\n\n")

        # Compare symmetric 50/50 (current default) with best
        default_mae = results.get('Symmetric 50/50', {}).get('mae', float('inf'))
        best_mae = results[best_config]['mae']

        if best_mae < default_mae - 0.005:
            f.write(f"**{best_config}** improves quality prediction by {default_mae - best_mae:.4f} MAE over the default symmetric 50/50 window.\n\n")
            f.write("**Recommendation:** Consider switching to this window configuration.\n")
        elif best_mae > default_mae + 0.005:
            f.write("The default symmetric 50/50 window performs best or near-best.\n\n")
            f.write("**Recommendation:** Keep the current default window.\n")
        else:
            f.write("Window configuration has minimal impact on quality prediction.\n\n")
            f.write("**Recommendation:** Keep the current default for simplicity.\n")

        # Additional insights
        f.write("\n### Window Size Analysis\n\n")
        narrow_mae = results.get('Narrow 10/10', {}).get('mae', float('nan'))
        wide_mae = results.get('Wide 100/100', {}).get('mae', float('nan'))
        sym50_mae = results.get('Symmetric 50/50', {}).get('mae', float('nan'))

        if not np.isnan(narrow_mae) and not np.isnan(wide_mae):
            if narrow_mae < wide_mae:
                f.write("Narrower windows perform better than wider windows.\n")
                f.write("This suggests quality is determined close to the boom moment.\n")
            else:
                f.write("Wider windows perform better than narrower windows.\n")
                f.write("This suggests quality depends on context further from the boom.\n")

        f.write("\n### Asymmetry Analysis\n\n")
        after_heavy = results.get('After-heavy 25/75', {}).get('mae', float('nan'))
        before_heavy = results.get('Before-heavy 75/25', {}).get('mae', float('nan'))

        if not np.isnan(after_heavy) and not np.isnan(before_heavy):
            if after_heavy < before_heavy:
                f.write("After-heavy windows outperform before-heavy windows.\n")
                f.write("Frames **after** the boom are more informative for quality prediction.\n")
            elif before_heavy < after_heavy:
                f.write("Before-heavy windows outperform after-heavy windows.\n")
                f.write("Frames **before** the boom are more informative for quality prediction.\n")
            else:
                f.write("No clear preference between before/after-heavy windows.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
