#!/usr/bin/env python3
"""
C6: Test 3-model pipeline with optimal quality parameters.

Documentation notes quality_window=50 and jitter_std=0 are optimal for 175 sims,
but 3-model results were obtained with defaults (window=25, jitter=5).
This tests whether combining optimal settings improves performance.

Usage:
    uv run python scripts/c6_combined_optimization.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='C6: Test combined optimization')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output directory (default: results)')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c6_combined_optimization")

    logger.info("=" * 70)
    logger.info("C6: Combined Optimization Experiment")
    logger.info("=" * 70)
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Testing 3-model with optimal quality params vs defaults")

    # Load dataset
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

    # Best combiner config
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    results = []

    # Configuration 1: Default quality params (window=25, jitter=5)
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONFIG 1: Default quality params (window=25, jitter=5)")
    logger.info("=" * 70)

    start_time = time.time()
    result_default = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
            quality_window=25,
            jitter_std=5.0,
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    default_time = time.time() - start_time

    mae_default = result_default.mean_metrics.get('selective_mae', float('nan'))
    mae_std_default = result_default.std_metrics.get('selective_mae', 0)
    cov_default = result_default.mean_metrics.get('coverage', 0)

    results.append({
        'config': 'default (w=25, j=5)',
        'mae': mae_default,
        'mae_std': mae_std_default,
        'coverage': cov_default,
    })

    logger.info(f"MAE: {mae_default:.2f} +/- {mae_std_default:.2f}")
    logger.info(f"Coverage: {cov_default:.1%}")
    logger.info(f"Time: {default_time:.1f}s")

    # Configuration 2: Optimal quality params (window=50, jitter=0)
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONFIG 2: Optimal quality params (window=50, jitter=0)")
    logger.info("=" * 70)

    start_time = time.time()
    result_optimal = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
            quality_window=50,
            jitter_std=0.0,
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    optimal_time = time.time() - start_time

    mae_optimal = result_optimal.mean_metrics.get('selective_mae', float('nan'))
    mae_std_optimal = result_optimal.std_metrics.get('selective_mae', 0)
    cov_optimal = result_optimal.mean_metrics.get('coverage', 0)

    results.append({
        'config': 'optimal (w=50, j=0)',
        'mae': mae_optimal,
        'mae_std': mae_std_optimal,
        'coverage': cov_optimal,
    })

    logger.info(f"MAE: {mae_optimal:.2f} +/- {mae_std_optimal:.2f}")
    logger.info(f"Coverage: {cov_optimal:.1%}")
    logger.info(f"Time: {optimal_time:.1f}s")

    # Configuration 3: Mixed (window=50, jitter=5)
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONFIG 3: Mixed (window=50, jitter=5)")
    logger.info("=" * 70)

    start_time = time.time()
    result_mixed = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
            quality_window=50,
            jitter_std=5.0,
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    mixed_time = time.time() - start_time

    mae_mixed = result_mixed.mean_metrics.get('selective_mae', float('nan'))
    mae_std_mixed = result_mixed.std_metrics.get('selective_mae', 0)
    cov_mixed = result_mixed.mean_metrics.get('coverage', 0)

    results.append({
        'config': 'mixed (w=50, j=5)',
        'mae': mae_mixed,
        'mae_std': mae_std_mixed,
        'coverage': cov_mixed,
    })

    logger.info(f"MAE: {mae_mixed:.2f} +/- {mae_std_mixed:.2f}")
    logger.info(f"Coverage: {cov_mixed:.1%}")
    logger.info(f"Time: {mixed_time:.1f}s")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"{'Config':<25} {'MAE':<15} {'Coverage':<12}")
    logger.info("-" * 52)
    for r in sorted(results, key=lambda x: x['mae']):
        logger.info(f"{r['config']:<25} {r['mae']:.2f} +/- {r['mae_std']:.2f}   {r['coverage']:.1%}")

    # Comparison to baseline
    baseline_mae = 2.97
    logger.info("")
    logger.info(f"Documented baseline: MAE 2.97 at 10.9% coverage")
    best = min(results, key=lambda x: x['mae'])
    diff = best['mae'] - baseline_mae
    logger.info(f"Best result: {best['config']} -> MAE {best['mae']:.2f} ({diff:+.2f} vs baseline)")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'c6_combined_optimization.md'

    with open(report_path, 'w') as f:
        f.write("# C6: Combined Optimization Experiment\n\n")
        f.write("## Objective\n")
        f.write("Test whether combining 3-model pipeline with optimal quality parameters\n")
        f.write("(quality_window=50, jitter_std=0) improves on the documented baseline.\n\n")
        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Models: CNN + HGB + LSTM (3-model pipeline)\n")
        f.write(f"- Combiner: std/s=15/t=0.70\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n\n")
        f.write("## Results\n\n")
        f.write("| Configuration | MAE | MAE std | Coverage |\n")
        f.write("|---------------|-----|---------|----------|\n")
        for r in sorted(results, key=lambda x: x['mae']):
            f.write(f"| {r['config']} | {r['mae']:.2f} | {r['mae_std']:.2f} | {r['coverage']:.1%} |\n")
        f.write(f"| **Documented baseline** | **2.97** | **0.06** | **10.9%** |\n\n")

        f.write("## Analysis\n\n")
        if best['mae'] < baseline_mae - 0.1:
            f.write(f"Combined optimization **improves** performance by {baseline_mae - best['mae']:.2f} frames.\n")
            f.write(f"The optimal quality parameters help the 3-model pipeline.\n\n")
            f.write(f"**Recommendation:** Use {best['config']} configuration.\n")
        elif best['mae'] > baseline_mae + 0.1:
            f.write(f"Combined optimization **hurts** performance by {best['mae'] - baseline_mae:.2f} frames.\n")
            f.write("This is unexpected - the configurations may interact negatively.\n\n")
            f.write("**Recommendation:** Stick with default quality parameters.\n")
        else:
            f.write(f"Combined optimization has **negligible effect** on MAE.\n")
            f.write("The quality parameters don't significantly impact 3-model performance.\n\n")
            f.write("**Recommendation:** Either configuration works - defaults are simpler.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
