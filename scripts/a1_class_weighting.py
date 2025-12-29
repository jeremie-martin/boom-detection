#!/usr/bin/env python3
"""
A1: Test class weighting for HGB training.

Compares 3-model pipeline performance with and without class weights.
Class imbalance exists because boom frame divides sequence into before/after
sections of varying lengths.

Usage:
    uv run python scripts/a1_class_weighting.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description='A1: Test HGB class weighting')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output directory (default: results)')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("a1_class_weighting")

    logger.info("=" * 70)
    logger.info("A1: HGB Class Weighting Experiment")
    logger.info("=" * 70)
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: PRODUCTION_CONFIG (no caustic)")

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

    # The combiner configuration (best known: std/s=15/t=0.70)
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    # Test with class weighting disabled (current behavior - no weights)
    logger.info("")
    logger.info("=" * 70)
    logger.info("BASELINE: No class weighting (current behavior)")
    logger.info("=" * 70)

    start_time = time.time()
    result_no_weights = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    baseline_time = time.time() - start_time

    mae_baseline = result_no_weights.mean_metrics.get('selective_mae', float('nan'))
    mae_std_baseline = result_no_weights.std_metrics.get('selective_mae', 0)
    cov_baseline = result_no_weights.mean_metrics.get('coverage', 0)

    logger.info(f"Baseline MAE: {mae_baseline:.2f} +/- {mae_std_baseline:.2f}")
    logger.info(f"Baseline Coverage: {cov_baseline:.1%}")
    logger.info(f"Time: {baseline_time:.1f}s")

    # Test with class weighting enabled
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST: With class weighting")
    logger.info("=" * 70)

    start_time = time.time()
    result_with_weights = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
            use_class_weights=True,  # This is the new parameter we'll add
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    weighted_time = time.time() - start_time

    mae_weighted = result_with_weights.mean_metrics.get('selective_mae', float('nan'))
    mae_std_weighted = result_with_weights.std_metrics.get('selective_mae', 0)
    cov_weighted = result_with_weights.mean_metrics.get('coverage', 0)

    logger.info(f"Weighted MAE: {mae_weighted:.2f} +/- {mae_std_weighted:.2f}")
    logger.info(f"Weighted Coverage: {cov_weighted:.1%}")
    logger.info(f"Time: {weighted_time:.1f}s")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    diff = mae_weighted - mae_baseline
    logger.info(f"Baseline MAE: {mae_baseline:.2f} +/- {mae_std_baseline:.2f} at {cov_baseline:.1%}")
    logger.info(f"Weighted MAE: {mae_weighted:.2f} +/- {mae_std_weighted:.2f} at {cov_weighted:.1%}")
    logger.info(f"Difference: {diff:+.2f} frames ({'BETTER' if diff < 0 else 'WORSE' if diff > 0 else 'SAME'})")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'a1_class_weighting.md'

    with open(report_path, 'w') as f:
        f.write("# A1: HGB Class Weighting Experiment\n\n")
        f.write("## Objective\n")
        f.write("Test whether adding class weights to HistGradientBoostingClassifier training\n")
        f.write("improves boom detection performance. Class imbalance exists because boom frame\n")
        f.write("divides each simulation into before/after sections of varying lengths.\n\n")
        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Models: CNN + HGB + LSTM (3-model pipeline)\n")
        f.write(f"- Combiner: std/s=15/t=0.70\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n\n")
        f.write("## Results\n\n")
        f.write("| Configuration | MAE | MAE std | Coverage |\n")
        f.write("|---------------|-----|---------|----------|\n")
        f.write(f"| Baseline (no weights) | {mae_baseline:.2f} | {mae_std_baseline:.2f} | {cov_baseline:.1%} |\n")
        f.write(f"| With class weights | {mae_weighted:.2f} | {mae_std_weighted:.2f} | {cov_weighted:.1%} |\n\n")
        f.write("## Analysis\n\n")
        if abs(diff) < 0.1:
            f.write("Class weighting has **negligible effect** on performance.\n")
            f.write("The class imbalance does not appear to harm HGB training significantly.\n")
            f.write("This makes sense because the boom detection task is about finding a transition\n")
            f.write("point, and both classes provide useful signal for this.\n\n")
            f.write("**Recommendation:** Do not add class weighting - it adds complexity without benefit.\n")
        elif diff < -0.5:
            f.write(f"Class weighting **improves** performance by {-diff:.2f} frames.\n")
            f.write("The class imbalance was hurting HGB's ability to identify boom frames.\n\n")
            f.write("**Recommendation:** Keep class weighting enabled.\n")
        else:
            f.write(f"Class weighting **hurts** performance by {diff:.2f} frames.\n")
            f.write("The unweighted model actually performs better, possibly because\n")
            f.write("the majority class (before-boom frames) provides useful regularization.\n\n")
            f.write("**Recommendation:** Do not add class weighting.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
