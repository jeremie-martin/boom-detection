#!/usr/bin/env python3
"""
Sweep 2-model ThresholdCombiner parameters.

Uses CNN + HGB models. The combiner considers both model agreement and
predicted quality to decide whether to accept a simulation.

The key insight is that TESTING COMBINERS IS FREE - models are trained once,
then combiners can be swapped instantly. This allows comprehensive parameter
exploration at minimal cost.

Usage:
    uv run python scripts/sweep_2model.py data
    uv run python scripts/sweep_2model.py data --seeds 42 43 44 45 46
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
    parser = argparse.ArgumentParser(description='Sweep 2-model ThresholdCombiner parameters')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_2model")

    logger.info("SWEEP: 2-Model ThresholdCombiner Parameters")
    logger.info(f"Frame models: CNN + HGB")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: PRODUCTION_CONFIG (no caustic)")

    # Load dataset and features
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

    # Create CombinerExperiment - trains models ONCE
    logger.info("Creating CombinerExperiment (training models once)...")
    start_time = time.time()
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),  # 2-model pipeline
            combiner=ThresholdCombiner(),  # Default combiner, will be replaced
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    train_time = time.time() - start_time
    logger.info(f"CombinerExperiment created in {train_time:.1f}s")
    logger.info("Models trained - now sweeping combiners (this is FREE)")

    # Generate all combiner configurations using grid
    combiners = ThresholdCombiner.grid(
        agreement_transform=['sqrt', 'linear', 'sigmoid', 'quadratic'],
        disagreement_scale=[3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        threshold=[0.50, 0.55, 0.60, 0.65, 0.70],
    )

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
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_std': rmse_std,
            'coverage': cov,
            'coverage_std': cov_std,
        })

        if i % 20 == 0 or i == len(combiners):
            logger.info(f"Progress: {i}/{len(combiners)} configurations evaluated")

    sweep_time = time.time() - start_sweep
    logger.info(f"Sweep completed in {sweep_time:.1f}s ({sweep_time/len(combiners):.2f}s per config)")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: TOP 10 CONFIGURATIONS BY MAE")
    logger.info("=" * 70)

    # Sort by MAE
    results_by_mae = sorted(results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info(f"{'Rank':<5} {'Formula':<10} {'Scale':<7} {'Thresh':<7} {'MAE +/- std':<16} {'RMSE +/- std':<16} {'Coverage':<12}")
    logger.info("-" * 80)

    for rank, r in enumerate(results_by_mae[:10], 1):
        logger.info(f"{rank:<5} {r['formula']:<10} {r['scale']:<7.0f} {r['threshold']:<7.2f} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}   "
                   f"{r['rmse']:.2f} +/- {r['rmse_std']:.2f}   "
                   f"{r['coverage']:.1%}")

    # Best at different coverage levels
    logger.info("")
    logger.info("=" * 70)
    logger.info("BEST MAE AT DIFFERENT COVERAGE LEVELS")
    logger.info("=" * 70)

    coverage_ranges = [
        (0.00, 0.10, '0-10%'),
        (0.10, 0.20, '10-20%'),
        (0.20, 0.30, '20-30%'),
        (0.30, 0.50, '30-50%'),
    ]

    for low, high, label in coverage_ranges:
        in_range = [r for r in results if low <= r['coverage'] < high]
        if in_range:
            best = min(in_range, key=lambda x: x['mae'])
            logger.info(f"Coverage {label}: {best['formula']}/s={best['scale']:.0f}/t={best['threshold']:.2f} "
                       f"-> MAE {best['mae']:.2f} +/- {best['mae_std']:.2f}, "
                       f"RMSE {best['rmse']:.2f} at {best['coverage']:.1%}")
        else:
            logger.info(f"Coverage {label}: No configurations in this range")

    # Save results
    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        with open(run_dir / 'results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Results saved to {run_dir / 'results.csv'}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
