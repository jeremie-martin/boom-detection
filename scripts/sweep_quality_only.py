#!/usr/bin/env python3
"""
Sweep QualityGatedCombiner thresholds.

This is the simplest approach: accept simulations where predicted quality >= threshold.
No model agreement is considered.

The key insight is that TESTING COMBINERS IS FREE - models are trained once,
then combiners can be swapped instantly. This script demonstrates that principle.

Usage:
    uv run python scripts/sweep_quality_only.py data
    uv run python scripts/sweep_quality_only.py data --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import QualityGatedCombiner, ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='Sweep QualityGatedCombiner thresholds')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_quality_only")

    logger.info("SWEEP: QualityGatedCombiner Thresholds")
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
            combiner=ThresholdCombiner(),  # Default combiner, will be replaced
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    train_time = time.time() - start_time
    logger.info(f"CombinerExperiment created in {train_time:.1f}s")
    logger.info("Models trained - now sweeping combiners (this is FREE)")

    # Sweep QualityGatedCombiner thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    combiners = [QualityGatedCombiner(threshold=t) for t in thresholds]

    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEPING QUALITYGATEDCOMBINER THRESHOLDS")
    logger.info("=" * 70)

    results = []
    for combiner in combiners:
        result = experiment.evaluate(combiner, verbose=False)

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        rmse_std = result.std_metrics.get('selective_rmse', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        results.append({
            'threshold': combiner.threshold,
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_std': rmse_std,
            'coverage': cov,
            'coverage_std': cov_std,
        })

        logger.info(f"threshold={combiner.threshold:.2f}: "
                   f"MAE {mae:.2f} +/- {mae_std:.2f}, "
                   f"RMSE {rmse:.2f} +/- {rmse_std:.2f}, "
                   f"coverage {cov:.1%} +/- {cov_std:.1%}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Sort by MAE
    results_by_mae = sorted(results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info("Results sorted by MAE:")
    logger.info(f"{'Threshold':<12} {'MAE +/- std':<18} {'RMSE +/- std':<18} {'Coverage':<15}")
    logger.info("-" * 65)

    for r in results_by_mae:
        logger.info(f"{r['threshold']:<12.2f} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}      "
                   f"{r['rmse']:.2f} +/- {r['rmse_std']:.2f}      "
                   f"{r['coverage']:.1%} +/- {r['coverage_std']:.1%}")

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
