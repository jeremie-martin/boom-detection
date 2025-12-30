#!/usr/bin/env python3
"""
A2: Test feature normalization for neural networks.

Compares 3-model pipeline performance with and without feature normalization.
Normalization standardizes features to zero mean and unit variance, which can
help neural network training convergence.

Usage:
    uv run python scripts/a2_feature_normalization.py data --seeds 42 43 44
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
    parser = argparse.ArgumentParser(description='A2: Test feature normalization')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='Output directory (default: results)')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("a2_feature_normalization")

    logger.info("=" * 70)
    logger.info("A2: Feature Normalization Experiment")
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

    # Test without normalization (current behavior)
    logger.info("")
    logger.info("=" * 70)
    logger.info("BASELINE: No feature normalization (current behavior)")
    logger.info("=" * 70)

    start_time = time.time()
    result_no_norm = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
            normalize_features=False,
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    baseline_time = time.time() - start_time

    mae_baseline = result_no_norm.mean_metrics.get('selective_mae', float('nan'))
    mae_std_baseline = result_no_norm.std_metrics.get('selective_mae', 0)
    cov_baseline = result_no_norm.mean_metrics.get('coverage', 0)

    logger.info(f"Baseline MAE: {mae_baseline:.2f} +/- {mae_std_baseline:.2f}")
    logger.info(f"Baseline Coverage: {cov_baseline:.1%}")
    logger.info(f"Time: {baseline_time:.1f}s")

    # Test with normalization enabled
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST: With feature normalization")
    logger.info("=" * 70)

    start_time = time.time()
    result_with_norm = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=combiner,
            normalize_features=True,
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    norm_time = time.time() - start_time

    mae_norm = result_with_norm.mean_metrics.get('selective_mae', float('nan'))
    mae_std_norm = result_with_norm.std_metrics.get('selective_mae', 0)
    cov_norm = result_with_norm.mean_metrics.get('coverage', 0)

    logger.info(f"Normalized MAE: {mae_norm:.2f} +/- {mae_std_norm:.2f}")
    logger.info(f"Normalized Coverage: {cov_norm:.1%}")
    logger.info(f"Time: {norm_time:.1f}s")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    diff = mae_norm - mae_baseline
    logger.info(f"Baseline MAE: {mae_baseline:.2f} +/- {mae_std_baseline:.2f} at {cov_baseline:.1%}")
    logger.info(f"Normalized MAE: {mae_norm:.2f} +/- {mae_std_norm:.2f} at {cov_norm:.1%}")
    logger.info(f"Difference: {diff:+.2f} frames ({'BETTER' if diff < 0 else 'WORSE' if diff > 0 else 'SAME'})")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'a2_feature_normalization.md'

    with open(report_path, 'w') as f:
        f.write("# A2: Feature Normalization Experiment\n\n")
        f.write("## Objective\n")
        f.write("Test whether normalizing features to zero mean and unit variance improves\n")
        f.write("neural network (CNN, LSTM) training and prediction performance.\n\n")
        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Models: CNN + HGB + LSTM (3-model pipeline)\n")
        f.write(f"- Combiner: std/s=15/t=0.70\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n\n")
        f.write("## Implementation\n")
        f.write("- Normalization statistics computed on training data only\n")
        f.write("- Same statistics applied during inference\n")
        f.write("- Only applies to CNN and LSTM (not HGB, which is tree-based)\n\n")
        f.write("## Results\n\n")
        f.write("| Configuration | MAE | MAE std | Coverage |\n")
        f.write("|---------------|-----|---------|----------|\n")
        f.write(f"| Baseline (no normalization) | {mae_baseline:.2f} | {mae_std_baseline:.2f} | {cov_baseline:.1%} |\n")
        f.write(f"| With normalization | {mae_norm:.2f} | {mae_std_norm:.2f} | {cov_norm:.1%} |\n\n")
        f.write("## Analysis\n\n")

        if abs(diff) < 0.2:
            f.write("Feature normalization has **negligible effect** on performance.\n")
            f.write("This suggests the features are already reasonably scaled, or the\n")
            f.write("AdamW optimizer with weight decay is robust to feature scaling.\n\n")
            f.write("**Recommendation:** Optional - normalization adds slight overhead without benefit.\n")
        elif diff < -0.3:
            f.write(f"Feature normalization **improves** performance by {-diff:.2f} frames.\n")
            f.write("Neural networks benefit from standardized input features for faster\n")
            f.write("and more stable training convergence.\n\n")
            f.write("**Recommendation:** Enable feature normalization.\n")
        else:
            f.write(f"Feature normalization **hurts** performance by {diff:.2f} frames.\n")
            f.write("This is unexpected - possibly the raw feature scales carry useful\n")
            f.write("information that normalization removes.\n\n")
            f.write("**Recommendation:** Do not enable feature normalization.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
