#!/usr/bin/env python3
"""
E3: Quality Feature Selection.

Test different n_quality_features values:
- [20, 30, 50, 75, 100, all]

Usage:
    uv run python scripts/e3_quality_features.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='E3: Quality Feature Selection')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/e3_quality_features'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("e3_quality_features")

    logger.info("=" * 70)
    logger.info("E3: Quality Feature Selection")
    logger.info("=" * 70)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    cache = FeatureCache(config=PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    evaluator = CachedEvaluator(dataset, cache)
    logger.info(f"Evaluating on {len(evaluator.sim_ids)} simulations")

    # Get total feature count
    n_total_features = cache[sim_ids[0]].shape[1]
    logger.info(f"Total features available: {n_total_features}")

    # Values to test (500 means "all" since it's > n_total_features)
    n_features_list = [20, 30, 50, 75, 100, 500]

    # Standard combiner
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    all_results = {}

    # Test n_quality_features
    logger.info("\nTesting n_quality_features variations...")
    for n_features in n_features_list:
        actual_n = min(n_features, n_total_features)
        config_name = f"n={actual_n}" if n_features <= n_total_features else "all"
        logger.info(f"  Testing: {config_name} ({actual_n} features)")

        result = evaluator.cross_validate_selective(
            lambda nf=n_features: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb', 'lstm'),
                combiner=combiner,
                n_quality_features=nf,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        coverage = result.mean_metrics.get('coverage', 0)

        all_results[config_name] = {
            'n_features': actual_n,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': coverage,
        }
        logger.info(f"    MAE: {mae:.2f} ± {mae_std:.2f}, Coverage: {coverage:.1%}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Configuration':<15} {'MAE':<15} {'Coverage':<12}")
    logger.info("-" * 42)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{name:<15} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['coverage']:.1%}")

    best_name = min(all_results, key=lambda x: all_results[x]['mae'] if not np.isnan(all_results[x]['mae']) else float('inf'))
    logger.info(f"\nBest: {best_name}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("config,n_features,mae,mae_std,coverage\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['n_features']},{r['mae']:.2f},{r['mae_std']:.2f},{r['coverage']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# E3: Quality Feature Selection\n\n")
        f.write("## Objective\n")
        f.write("Test different n_quality_features values for quality prediction.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write(f"- Total features available: {n_total_features}\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Parameters Tested\n")
        f.write(f"- n_quality_features: [20, 30, 50, 75, 100, all ({n_total_features})]\n\n")

        f.write("## Results\n\n")
        f.write("| n_features | MAE | Coverage |\n")
        f.write("|------------|-----|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['coverage']:.1%} |\n")

        best = all_results[best_name]
        baseline_mae = 2.97  # Known baseline (n_quality_features=50)

        f.write(f"\n## Best Configuration\n\n")
        f.write(f"**{best_name}**: MAE {best['mae']:.2f} ± {best['mae_std']:.2f}\n\n")

        f.write("## Analysis\n\n")

        # Compare to baseline (n=50 is baseline)
        diff = best['mae'] - baseline_mae

        if abs(diff) < 0.2:
            f.write(f"**n_quality_features changes have minimal impact** (within 0.2 of baseline).\n")
            f.write("Recommendation: Keep current n_quality_features=50.\n")
        elif diff < 0:
            f.write(f"**{best_name} improves over baseline** by {-diff:.2f} frames.\n")
            f.write("Recommendation: Consider updating n_quality_features.\n")
        else:
            f.write(f"**Baseline (n=50) outperforms** best sweep result by {diff:.2f} frames.\n")
            f.write("Recommendation: Keep current n_quality_features=50.\n")

        # Analyze trend
        f.write("\n### Trend Analysis\n\n")
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['n_features'])
        maes = [r['mae'] for _, r in sorted_results]
        if maes[-1] < maes[0]:
            f.write("More features generally improve MAE.\n")
        elif maes[-1] > maes[0]:
            f.write("Fewer features generally work better (feature selection helps).\n")
        else:
            f.write("No clear trend with number of features.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
