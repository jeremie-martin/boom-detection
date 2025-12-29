#!/usr/bin/env python3
"""
Sweep LSTM+HGB 2-model pipeline parameters.

LSTM is documented as the "best individual model" (MAE 18.3 vs CNN 20.2).
This script tests whether LSTM+HGB outperforms CNN+HGB.

Usage:
    uv run python scripts/sweep_lstm_hgb.py data
    uv run python scripts/sweep_lstm_hgb.py data --seeds 42 43 44
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
    parser = argparse.ArgumentParser(description='Sweep LSTM+HGB 2-model pipeline parameters')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_lstm_hgb")

    logger.info("SWEEP: LSTM+HGB 2-Model Pipeline")
    logger.info(f"Frame models: LSTM + HGB (dropping CNN)")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: PRODUCTION_CONFIG (no caustic)")
    logger.info("")
    logger.info("Hypothesis: LSTM is best individual model, may improve 2-model pipeline")

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

    # Create CombinerExperiment - trains LSTM + HGB models ONCE
    logger.info("")
    logger.info("Creating CombinerExperiment (training LSTM + HGB models once)...")
    start_time = time.time()
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('lstm', 'hgb'),  # LSTM + HGB instead of CNN + HGB
            combiner=ThresholdCombiner(primary_model='lstm'),  # Default to LSTM as primary
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    train_time = time.time() - start_time
    logger.info(f"CombinerExperiment created in {train_time:.1f}s")
    logger.info("Models trained - now sweeping combiners (this is FREE)")

    # Generate combiner configurations
    # Test both lstm and hgb as primary model
    combiners = []

    # LSTM as primary
    combiners += ThresholdCombiner.grid(
        primary_model=['lstm'],
        agreement_transform=['sqrt', 'linear', 'sigmoid'],
        disagreement_scale=[5.0, 10.0, 15.0, 20.0],
        threshold=[0.55, 0.60, 0.65, 0.70, 0.75],
    )

    # HGB as primary
    combiners += ThresholdCombiner.grid(
        primary_model=['hgb'],
        agreement_transform=['sqrt'],
        disagreement_scale=[10.0, 15.0, 20.0],
        threshold=[0.60, 0.65, 0.70],
    )

    # Median
    combiners += ThresholdCombiner.grid(
        primary_model=['median'],
        agreement_transform=['sqrt'],
        disagreement_scale=[10.0, 15.0],
        threshold=[0.60, 0.70],
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
            'primary': combiner.primary_model,
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

    # Summary by primary model
    logger.info("")
    logger.info("=" * 70)
    logger.info("BEST RESULTS BY PRIMARY MODEL")
    logger.info("=" * 70)

    for primary in ['lstm', 'hgb', 'median']:
        primary_results = [r for r in results if r['primary'] == primary and not np.isnan(r['mae'])]
        if primary_results:
            best = min(primary_results, key=lambda x: x['mae'])
            logger.info(f"\nprimary={primary}:")
            logger.info(f"  Best: {best['formula']}/s={best['scale']:.0f}/t={best['threshold']:.2f}")
            logger.info(f"  MAE {best['mae']:.2f} +/- {best['mae_std']:.2f} at {best['coverage']:.1%}")

    # Overall top 10
    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 10 OVERALL (LSTM+HGB)")
    logger.info("=" * 70)

    results_by_mae = sorted(results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info(f"{'Rank':<5} {'Primary':<8} {'Formula':<10} {'Scale':<7} {'Thresh':<7} {'MAE +/- std':<16} {'Coverage':<12}")
    logger.info("-" * 75)

    for rank, r in enumerate(results_by_mae[:10], 1):
        logger.info(f"{rank:<5} {r['primary']:<8} {r['formula']:<10} {r['scale']:<7.0f} {r['threshold']:<7.2f} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}   {r['coverage']:.1%}")

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
