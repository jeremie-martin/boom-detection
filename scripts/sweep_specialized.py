#!/usr/bin/env python3
"""
Sweep specialized HGB model (hgb_0.5) with QualityGatedCombiner.

Compares two model configurations:
1. Baseline: ('cnn', 'hgb') - 2 regular models, uses median for prediction
2. Specialized: ('cnn', 'hgb', 'hgb_0.5') - includes specialized HGB trained on
   high-quality samples only, uses hgb_0.5 for prediction

The key insight is that TESTING COMBINERS IS FREE - models are trained once,
then combiners can be swapped instantly using CombinerExperiment.

With the unified FrameModelConfig design, a specialized model is just a regular
model with a quality_threshold > 0 for training data filtering.

Usage:
    uv run python scripts/sweep_specialized.py data
    uv run python scripts/sweep_specialized.py data --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import (
    FrameModelConfig,
    QualityGatedCombiner,
    QualityGatedModelCombiner,
)
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='Sweep specialized HGB (hgb_0.5) with QualityGatedCombiner')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_specialized")

    logger.info("SWEEP: Specialized HGB (hgb_0.5) with QualityGatedCombiner")
    logger.info("Using unified FrameModelConfig design")
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

    # Sweep thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    # =========================================================================
    # PART 1: BASELINE - Train 2-model (cnn, hgb) once, sweep combiners
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 1: BASELINE (cnn + hgb)")
    logger.info("=" * 70)

    logger.info("Creating CombinerExperiment for baseline (training 2 models once)...")
    start_time = time.time()
    baseline_experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),
            combiner=QualityGatedCombiner(threshold=0.5),  # Default, will be replaced
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    baseline_train_time = time.time() - start_time
    logger.info(f"Baseline models trained in {baseline_train_time:.1f}s")
    logger.info("Now sweeping thresholds (this is FREE)...")

    baseline_results = []
    for thresh in thresholds:
        combiner = QualityGatedCombiner(threshold=thresh)
        result = baseline_experiment.evaluate(combiner, verbose=False)

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        rmse_std = result.std_metrics.get('selective_rmse', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        baseline_results.append({
            'type': 'baseline',
            'threshold': thresh,
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_std': rmse_std,
            'coverage': cov,
            'coverage_std': cov_std,
        })

        logger.info(f"  threshold={thresh:.2f}: "
                   f"MAE {mae:.2f} +/- {mae_std:.2f}, "
                   f"RMSE {rmse:.2f} +/- {rmse_std:.2f}, "
                   f"coverage {cov:.1%}")

    # =========================================================================
    # PART 2: SPECIALIZED - Train 3-model (cnn, hgb, hgb_0.5), sweep combiners
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: SPECIALIZED (cnn + hgb + hgb_0.5)")
    logger.info("=" * 70)

    logger.info("Creating CombinerExperiment for specialized (training 3 models once)...")
    start_time = time.time()

    # Use FrameModelConfig for the specialized HGB model
    specialized_experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=(
                'cnn',
                'hgb',
                FrameModelConfig('hgb', 0.5),  # hgb_0.5: trained on quality >= 0.5
            ),
            combiner=QualityGatedModelCombiner(threshold=0.5, primary_model='hgb_0.5'),
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    specialized_train_time = time.time() - start_time
    logger.info(f"Specialized models trained in {specialized_train_time:.1f}s")
    logger.info("Now sweeping thresholds (this is FREE)...")

    specialized_results = []
    for thresh in thresholds:
        # Use QualityGatedModelCombiner with hgb_0.5 as primary model
        combiner = QualityGatedModelCombiner(threshold=thresh, primary_model='hgb_0.5')
        result = specialized_experiment.evaluate(combiner, verbose=False)

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        rmse_std = result.std_metrics.get('selective_rmse', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        specialized_results.append({
            'type': 'hgb_0.5',
            'threshold': thresh,
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'rmse_std': rmse_std,
            'coverage': cov,
            'coverage_std': cov_std,
        })

        logger.info(f"  threshold={thresh:.2f}: "
                   f"MAE {mae:.2f} +/- {mae_std:.2f}, "
                   f"RMSE {rmse:.2f} +/- {rmse_std:.2f}, "
                   f"coverage {cov:.1%}")

    # =========================================================================
    # Summary comparison
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON: Baseline vs Specialized (hgb_0.5)")
    logger.info("=" * 70)

    logger.info("")
    logger.info(f"{'Threshold':<12} {'Baseline MAE':<18} {'hgb_0.5 MAE':<18} {'Improvement':<12}")
    logger.info("-" * 60)

    for base, spec in zip(baseline_results, specialized_results):
        improvement = base['mae'] - spec['mae']
        better = "BETTER" if improvement > 0 else "worse" if improvement < 0 else "same"
        logger.info(f"{base['threshold']:<12.2f} "
                   f"{base['mae']:.2f} +/- {base['mae_std']:.2f}   "
                   f"{spec['mae']:.2f} +/- {spec['mae_std']:.2f}   "
                   f"{improvement:+.2f} ({better})")

    # Combined results
    all_results = baseline_results + specialized_results

    # Save results
    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        with open(run_dir / 'results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        logger.info(f"Results saved to {run_dir / 'results.csv'}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Baseline training: {baseline_train_time:.1f}s")
    logger.info(f"Specialized training: {specialized_train_time:.1f}s")
    logger.info(f"Total thresholds swept: {2 * len(thresholds)} (FREE after training)")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
