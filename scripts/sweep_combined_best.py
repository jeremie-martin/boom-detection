#!/usr/bin/env python3
"""
Test combined best parameters from previous sweeps.

Combines:
1. quality_window=35, jitter_std=10 (best from quality params sweep)
2. Sigmoid transform with various scales (best from extended sweep)
3. score_function alternatives (min, product)

This requires RETRAINING since quality params changed.

Usage:
    uv run python scripts/sweep_combined_best.py data
    uv run python scripts/sweep_combined_best.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='Test combined best parameters')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_combined_best")

    logger.info("SWEEP: Combined Best Parameters")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: PRODUCTION_CONFIG (no caustic)")
    logger.info("")
    logger.info("TESTING COMBINATION OF:")
    logger.info("  - quality_window=35, jitter_std=10 (best from quality sweep)")
    logger.info("  - Sigmoid transform with scales 5, 10, 15")
    logger.info("  - score_function alternatives (weighted, min)")
    logger.info("")
    logger.info("NOTE: This requires RETRAINING for quality params")

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

    # Best quality parameters from previous sweep
    quality_window = 35
    jitter_std = 10

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"QUALITY PARAMS: window={quality_window}, jitter={jitter_std}")
    logger.info("=" * 70)

    # Combiners to test with new quality params
    combiners = []

    # Sigmoid transform (best from extended sweep)
    combiners += ThresholdCombiner.grid(
        agreement_transform=['sigmoid'],
        disagreement_scale=[5.0, 10.0, 15.0],
        threshold=[0.65, 0.70, 0.75],
        primary_model=['cnn', 'median'],
    )

    # sqrt baseline for comparison
    combiners += ThresholdCombiner.grid(
        agreement_transform=['sqrt'],
        disagreement_scale=[15.0],
        threshold=[0.65, 0.70, 0.75],
        primary_model=['cnn', 'median'],
    )

    # min score function (interesting alternative)
    combiners += ThresholdCombiner.grid(
        agreement_transform=['sqrt'],
        disagreement_scale=[15.0],
        threshold=[0.60, 0.65, 0.70],
        score_function=['min'],
    )

    logger.info(f"Testing {len(combiners)} combiner configurations")
    logger.info("")

    # Create pipeline factory with best quality params
    def pipeline_factory():
        return BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),
            combiner=ThresholdCombiner(),  # Will be replaced
            quality_window=quality_window,
            jitter_std=jitter_std,
        )

    # Create CombinerExperiment - trains models ONCE with new quality params
    logger.info("Creating CombinerExperiment (training models with new quality params)...")
    start_time = time.time()
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=pipeline_factory,
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    train_time = time.time() - start_time
    logger.info(f"CombinerExperiment created in {train_time:.1f}s")
    logger.info("Models trained - now sweeping combiners (this is FREE)")

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"SWEEPING {len(combiners)} COMBINER CONFIGURATIONS")
    logger.info("=" * 70)

    results = []
    start_sweep = time.time()

    for i, combiner in enumerate(combiners, 1):
        result = experiment.evaluate(combiner, verbose=False)

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        rmse_std = result.std_metrics.get('selective_rmse', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        results.append({
            'formula': combiner.agreement_transform,
            'scale': combiner.disagreement_scale,
            'threshold': combiner.threshold,
            'primary': combiner.primary_model,
            'score_func': combiner.score_function,
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_std': rmse_std,
            'coverage': cov,
            'coverage_std': cov_std,
        })

        if i % 10 == 0 or i == len(combiners):
            logger.info(f"Progress: {i}/{len(combiners)} configurations evaluated")

    sweep_time = time.time() - start_sweep
    logger.info(f"Sweep completed in {sweep_time:.1f}s ({sweep_time/len(combiners):.2f}s per config)")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 10 RESULTS (sorted by MAE)")
    logger.info("=" * 70)

    results_by_mae = sorted(results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info(f"{'Rank':<5} {'Formula':<8} {'Scale':<6} {'Thresh':<7} {'Primary':<8} {'ScoreFn':<9} {'MAE +/- std':<16} {'Coverage':<10}")
    logger.info("-" * 85)

    for rank, r in enumerate(results_by_mae[:10], 1):
        logger.info(f"{rank:<5} {r['formula']:<8} {r['scale']:<6.0f} {r['threshold']:<7.2f} {r['primary']:<8} {r['score_func']:<9} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}   {r['coverage']:.1%}")

    # Compare to baseline
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: Best combined vs previous baselines")
    logger.info("=" * 70)

    if results_by_mae and not np.isnan(results_by_mae[0]['mae']):
        best = results_by_mae[0]
        logger.info(f"Best combined: {best['formula']}/s={best['scale']:.0f}/t={best['threshold']:.2f}")
        logger.info(f"  MAE {best['mae']:.2f} +/- {best['mae_std']:.2f} at {best['coverage']:.1%}")
        logger.info("")
        logger.info("Previous baselines (with default quality params):")
        logger.info("  sqrt/s=15/t=0.70: MAE ~2.78 at 11.5%")
        logger.info("  quality_window=35/jitter=10: MAE 2.70 at 8.1%")

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
