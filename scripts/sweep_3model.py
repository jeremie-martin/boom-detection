#!/usr/bin/env python3
"""
Sweep 3-model pipeline parameters.

Uses CNN + HGB + LSTM models with `std` disagreement metric.
The key finding is that 3-model with `std` metric WORKS, achieving MAE 3.27 at 12.2%.

The `std` metric (standard deviation) is more robust than `range` (max-min) when
the LSTM model occasionally disagrees with CNN/HGB.

The key insight is that TESTING COMBINERS IS FREE - models are trained once,
then combiners can be swapped instantly.

Usage:
    uv run python scripts/sweep_3model.py data
    uv run python scripts/sweep_3model.py data --seeds 42 43 44 45 46
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
    parser = argparse.ArgumentParser(description='Sweep 3-model pipeline parameters')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_3model")

    logger.info("SWEEP: 3-Model Pipeline Parameters")
    logger.info(f"Frame models: CNN + HGB + LSTM")
    logger.info(f"Disagreement metric: std (robust to outliers)")
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
    logger.info("Creating CombinerExperiment (training 3 models once)...")
    start_time = time.time()
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),  # 3-model pipeline
            combiner=ThresholdCombiner(
                disagreement_metric='std',  # Key: use std instead of range
            ),
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    train_time = time.time() - start_time
    logger.info(f"CombinerExperiment created in {train_time:.1f}s")
    logger.info("Models trained - now sweeping combiners (this is FREE)")

    # Generate combiner configurations for std metric
    combiners_std = ThresholdCombiner.grid(
        disagreement_metric=['std'],
        agreement_transform=['sqrt'],
        disagreement_scale=[3.0, 5.0, 8.0, 10.0, 15.0],
        threshold=[0.50, 0.55, 0.60, 0.65, 0.70],
    )

    # Also test range metric for comparison
    combiners_range = ThresholdCombiner.grid(
        disagreement_metric=['range'],
        agreement_transform=['sqrt'],
        disagreement_scale=[5.0, 10.0, 15.0, 20.0],
        threshold=[0.50, 0.55, 0.60, 0.65, 0.70],
    )

    all_combiners = combiners_std + combiners_range

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"SWEEPING {len(all_combiners)} COMBINER CONFIGURATIONS")
    logger.info(f"  - {len(combiners_std)} with std metric (recommended)")
    logger.info(f"  - {len(combiners_range)} with range metric (for comparison)")
    logger.info("=" * 70)

    results = []
    start_sweep = time.time()

    for i, combiner in enumerate(all_combiners, 1):
        result = experiment.evaluate(combiner, verbose=False)

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        rmse_std = result.std_metrics.get('selective_rmse', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        results.append({
            'metric': combiner.disagreement_metric,
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

        if i % 10 == 0 or i == len(all_combiners):
            logger.info(f"Progress: {i}/{len(all_combiners)} configurations evaluated")

    sweep_time = time.time() - start_sweep
    logger.info(f"Sweep completed in {sweep_time:.1f}s ({sweep_time/len(all_combiners):.2f}s per config)")

    # Summary: std metric results
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS: std METRIC (RECOMMENDED)")
    logger.info("=" * 70)

    std_results = [r for r in results if r['metric'] == 'std']
    std_by_mae = sorted(std_results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info(f"{'Rank':<5} {'Scale':<7} {'Thresh':<7} {'MAE +/- std':<16} {'RMSE +/- std':<16} {'Coverage':<12}")
    logger.info("-" * 70)

    for rank, r in enumerate(std_by_mae[:10], 1):
        logger.info(f"{rank:<5} {r['scale']:<7.0f} {r['threshold']:<7.2f} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}   "
                   f"{r['rmse']:.2f} +/- {r['rmse_std']:.2f}   "
                   f"{r['coverage']:.1%}")

    # Summary: range metric results
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS: range METRIC (COMPARISON)")
    logger.info("=" * 70)

    range_results = [r for r in results if r['metric'] == 'range']
    range_by_mae = sorted(range_results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info(f"{'Rank':<5} {'Scale':<7} {'Thresh':<7} {'MAE +/- std':<16} {'RMSE +/- std':<16} {'Coverage':<12}")
    logger.info("-" * 70)

    for rank, r in enumerate(range_by_mae[:10], 1):
        logger.info(f"{rank:<5} {r['scale']:<7.0f} {r['threshold']:<7.2f} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}   "
                   f"{r['rmse']:.2f} +/- {r['rmse_std']:.2f}   "
                   f"{r['coverage']:.1%}")

    # Comparison summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: std vs range")
    logger.info("=" * 70)

    if std_by_mae and range_by_mae:
        best_std = std_by_mae[0]
        best_range = range_by_mae[0]

        logger.info("")
        logger.info(f"Best std:   scale={best_std['scale']:.0f}/t={best_std['threshold']:.2f} "
                   f"-> MAE {best_std['mae']:.2f} +/- {best_std['mae_std']:.2f} at {best_std['coverage']:.1%}")
        logger.info(f"Best range: scale={best_range['scale']:.0f}/t={best_range['threshold']:.2f} "
                   f"-> MAE {best_range['mae']:.2f} +/- {best_range['mae_std']:.2f} at {best_range['coverage']:.1%}")

        if best_std['mae'] < best_range['mae']:
            logger.info("CONCLUSION: std metric is better (lower MAE)")
        elif best_range['mae'] < best_std['mae']:
            logger.info("CONCLUSION: range metric is better (lower MAE)")
        else:
            logger.info("CONCLUSION: Both metrics perform similarly")

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
