#!/usr/bin/env python3
"""
Sweep quality model hyperparameters.

Unlike combiner sweeps (which are FREE), this script requires retraining
for each configuration since quality_window and jitter_std affect model training.

Parameters tested:
- quality_window: Window around boom for quality features (15, 25, 35, 50)
- jitter_std: Training noise for boom estimate (0, 3, 5, 10)

Usage:
    uv run python scripts/sweep_quality_params.py data
    uv run python scripts/sweep_quality_params.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from itertools import product
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='Sweep quality model hyperparameters')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_quality_params")

    logger.info("SWEEP: Quality Model Hyperparameters")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: PRODUCTION_CONFIG (no caustic)")
    logger.info("")
    logger.info("NOTE: This sweep requires RETRAINING for each config (not free!)")

    # Load dataset and features
    logger.info("")
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    cache = FeatureCache(config=PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            logger.info(f"Extracting features ({loaded}/{len(sim_ids)} cached)...")
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            logger.info(f"Using cached features ({loaded} sims)")
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        logger.info("Extracting features...")
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    # Create evaluator
    evaluator = CachedEvaluator(dataset, cache)
    logger.info(f"Evaluating on {len(evaluator.sim_ids)} simulations")

    # Parameter grid
    quality_windows = [15, 25, 35, 50]
    jitter_stds = [0, 3, 5, 10]

    # Fixed combiner config (best from previous experiments)
    combiner = ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        threshold=0.70,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"SWEEPING {len(quality_windows) * len(jitter_stds)} CONFIGURATIONS")
    logger.info("=" * 70)
    logger.info(f"quality_window: {quality_windows}")
    logger.info(f"jitter_std: {jitter_stds}")
    logger.info(f"Fixed combiner: {combiner}")

    results = []
    total_configs = len(quality_windows) * len(jitter_stds)
    config_idx = 0

    for quality_window, jitter_std in product(quality_windows, jitter_stds):
        config_idx += 1
        logger.info("")
        logger.info(f"Config {config_idx}/{total_configs}: quality_window={quality_window}, jitter_std={jitter_std}")

        start_time = time.time()

        # Create pipeline with these quality parameters
        def pipeline_factory():
            return BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=combiner,
                quality_window=quality_window,
                jitter_std=jitter_std,
            )

        # Run cross-validation
        result = evaluator.cross_validate_selective(
            pipeline_factory,
            k=5,
            seeds=args.seeds,
            verbose=False,
        )

        train_time = time.time() - start_time

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        rmse_std = result.std_metrics.get('selective_rmse', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        results.append({
            'quality_window': quality_window,
            'jitter_std': jitter_std,
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_std': rmse_std,
            'coverage': cov,
            'coverage_std': cov_std,
            'train_time': train_time,
        })

        logger.info(f"  MAE {mae:.2f} +/- {mae_std:.2f}, "
                   f"RMSE {rmse:.2f} +/- {rmse_std:.2f}, "
                   f"coverage {cov:.1%}, "
                   f"time {train_time:.0f}s")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: RESULTS SORTED BY MAE")
    logger.info("=" * 70)

    results_by_mae = sorted(results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info(f"{'Window':<8} {'Jitter':<8} {'MAE +/- std':<18} {'RMSE +/- std':<18} {'Coverage':<12}")
    logger.info("-" * 70)

    for r in results_by_mae:
        logger.info(f"{r['quality_window']:<8} {r['jitter_std']:<8} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}      "
                   f"{r['rmse']:.2f} +/- {r['rmse_std']:.2f}      "
                   f"{r['coverage']:.1%}")

    # Best result
    if results_by_mae and not np.isnan(results_by_mae[0]['mae']):
        best = results_by_mae[0]
        logger.info("")
        logger.info(f"BEST: quality_window={best['quality_window']}, jitter_std={best['jitter_std']}")
        logger.info(f"      MAE {best['mae']:.2f} +/- {best['mae_std']:.2f} at {best['coverage']:.1%} coverage")

    # Save results
    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / 'results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"\nResults saved to {run_dir / 'results.csv'}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
