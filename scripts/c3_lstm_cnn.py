#!/usr/bin/env python3
"""
C3: LSTM + CNN Combination (Drop HGB).

Test 2-model pipeline with just neural networks.
Compare with the current best 3-model pipeline.

Usage:
    uv run python scripts/c3_lstm_cnn.py data --seeds 42 43 44
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
    parser = argparse.ArgumentParser(description='C3: LSTM + CNN Combination')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/c3_lstm_cnn'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c3_lstm_cnn")

    logger.info("=" * 70)
    logger.info("C3: LSTM + CNN Combination (Drop HGB)")
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

    # Configurations to test
    configs = [
        # Baseline 3-model (for comparison)
        ('3-model (std/s=15/t=0.70)', {
            'frame_models': ('cnn', 'hgb', 'lstm'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=15.0,
                disagreement_metric='std',
                threshold=0.70,
            ),
        }),
        # LSTM + CNN with std metric
        ('LSTM+CNN (std/s=10)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=10.0,
                disagreement_metric='std',
                threshold=0.70,
            ),
        }),
        ('LSTM+CNN (std/s=15)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=15.0,
                disagreement_metric='std',
                threshold=0.70,
            ),
        }),
        ('LSTM+CNN (std/s=20)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=20.0,
                disagreement_metric='std',
                threshold=0.70,
            ),
        }),
        # LSTM + CNN with range metric
        ('LSTM+CNN (range/s=10)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=10.0,
                disagreement_metric='range',
                threshold=0.70,
            ),
        }),
        ('LSTM+CNN (range/s=15)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=15.0,
                disagreement_metric='range',
                threshold=0.70,
            ),
        }),
        ('LSTM+CNN (range/s=20)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=20.0,
                disagreement_metric='range',
                threshold=0.70,
            ),
        }),
        # Test different thresholds with best scale
        ('LSTM+CNN (std/s=15/t=0.65)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=15.0,
                disagreement_metric='std',
                threshold=0.65,
            ),
        }),
        ('LSTM+CNN (std/s=15/t=0.75)', {
            'frame_models': ('lstm', 'cnn'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=15.0,
                disagreement_metric='std',
                threshold=0.75,
            ),
        }),
    ]

    all_results = {}

    for config_name, config in configs:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {config_name}")
        logger.info("=" * 70)

        result = evaluator.cross_validate_selective(
            lambda cfg=config: BoomDetectionPipeline(
                frame_models=cfg['frame_models'],
                combiner=cfg['combiner'],
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        coverage = result.mean_metrics.get('coverage', 0)
        coverage_std = result.std_metrics.get('coverage', 0)

        all_results[config_name] = {
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'coverage': coverage,
            'coverage_std': coverage_std,
        }

        logger.info(f"  MAE: {mae:.2f} +/- {mae_std:.2f}")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  Coverage: {coverage:.1%} +/- {coverage_std:.1%}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Configuration':<30} {'MAE':<15} {'RMSE':<10} {'Coverage':<12}")
    logger.info("-" * 67)
    for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{config_name:<30} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['rmse']:.2f}     {r['coverage']:.1%}")

    # Find best LSTM+CNN config
    lstm_cnn_results = {k: v for k, v in all_results.items() if 'LSTM+CNN' in k}
    if lstm_cnn_results:
        best_lstm_cnn = min(lstm_cnn_results, key=lambda x: lstm_cnn_results[x]['mae'] if not np.isnan(lstm_cnn_results[x]['mae']) else float('inf'))
        logger.info(f"\nBest LSTM+CNN: {best_lstm_cnn}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("config,mae,mae_std,rmse,coverage,coverage_std\n")
        for config_name, r in all_results.items():
            f.write(f"{config_name},{r['mae']:.2f},{r['mae_std']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# C3: LSTM + CNN Combination (Drop HGB)\n\n")
        f.write("## Objective\n")
        f.write("Test whether dropping HGB and using only neural networks (LSTM + CNN) is competitive.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE | RMSE | Coverage |\n")
        f.write("|---------------|-----|------|----------|\n")
        for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {config_name} | {r['mae']:.2f} +/- {r['mae_std']:.2f} | {r['rmse']:.2f} | {r['coverage']:.1%} |\n")

        f.write("\n## Analysis\n\n")

        # Compare with 3-model baseline
        baseline_mae = all_results.get('3-model (std/s=15/t=0.70)', {}).get('mae', float('nan'))

        if lstm_cnn_results:
            best_lstm_cnn = min(lstm_cnn_results, key=lambda x: lstm_cnn_results[x]['mae'] if not np.isnan(lstm_cnn_results[x]['mae']) else float('inf'))
            best_mae = lstm_cnn_results[best_lstm_cnn]['mae']

            f.write(f"**Best LSTM+CNN configuration:** {best_lstm_cnn}\n")
            f.write(f"- MAE: {best_mae:.2f}\n")
            f.write(f"- Coverage: {lstm_cnn_results[best_lstm_cnn]['coverage']:.1%}\n\n")

            if not np.isnan(baseline_mae) and not np.isnan(best_mae):
                diff = best_mae - baseline_mae
                if diff < -0.2:
                    f.write(f"**LSTM+CNN outperforms 3-model baseline** by {-diff:.2f} frames!\n\n")
                    f.write("**Recommendation:** Consider LSTM+CNN as a simpler alternative.\n")
                elif diff > 0.5:
                    f.write(f"**LSTM+CNN underperforms 3-model baseline** by {diff:.2f} frames.\n\n")
                    f.write("**Recommendation:** Keep the 3-model pipeline.\n")
                else:
                    f.write(f"**LSTM+CNN is comparable to 3-model** (diff: {diff:+.2f} frames).\n\n")
                    f.write("**Recommendation:** LSTM+CNN is a viable alternative - fewer models to train.\n")

        f.write("\n### Metric Comparison (std vs range)\n\n")

        # Compare std vs range
        std_results = {k: v for k, v in lstm_cnn_results.items() if 'std' in k}
        range_results = {k: v for k, v in lstm_cnn_results.items() if 'range' in k}

        if std_results and range_results:
            best_std_mae = min(v['mae'] for v in std_results.values() if not np.isnan(v['mae']))
            best_range_mae = min(v['mae'] for v in range_results.values() if not np.isnan(v['mae']))

            if best_std_mae < best_range_mae:
                f.write(f"- `std` metric is better ({best_std_mae:.2f} vs {best_range_mae:.2f})\n")
            else:
                f.write(f"- `range` metric is better ({best_range_mae:.2f} vs {best_std_mae:.2f})\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
