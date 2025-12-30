#!/usr/bin/env python3
"""
C5: Disagreement Metric MAD (Median Absolute Deviation).

Test MAD as an alternative disagreement metric to std and range.
MAD is more robust to outliers than std.

Usage:
    uv run python scripts/c5_mad_metric.py data --seeds 42 43 44
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
    parser = argparse.ArgumentParser(description='C5: Disagreement Metric MAD')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/c5_mad_metric'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c5_mad_metric")

    logger.info("=" * 70)
    logger.info("C5: Disagreement Metric MAD")
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

    # Train models once with CombinerExperiment
    logger.info("\nTraining 3-model pipeline (CNN + HGB + LSTM)...")
    experiment = evaluator.create_combiner_experiment(
        lambda: BoomDetectionPipeline(frame_models=('cnn', 'hgb', 'lstm')),
        seeds=args.seeds,
    )
    logger.info("Models trained. Starting combiner sweep...")

    # Configurations to test
    # MAD typically produces smaller values than std < range
    # So we test smaller scales for MAD
    metrics = ['mad', 'std', 'range']
    scales_by_metric = {
        'mad': [2, 3, 5, 8, 10],  # MAD typically smaller
        'std': [5, 8, 10, 15, 20],  # std intermediate
        'range': [10, 15, 20, 25, 30],  # range typically larger
    }
    thresholds = [0.65, 0.70, 0.75]

    all_results = {}

    for metric in metrics:
        scales = scales_by_metric[metric]
        for scale in scales:
            for threshold in thresholds:
                config_name = f"{metric}/s={scale}/t={threshold}"

                combiner = ThresholdCombiner(
                    primary_model='cnn',
                    agreement_transform='sqrt',
                    disagreement_scale=float(scale),
                    disagreement_metric=metric,
                    threshold=threshold,
                )

                result = experiment.evaluate(combiner)

                mae = result.mean_metrics.get('selective_mae', float('nan'))
                mae_std = result.std_metrics.get('selective_mae', 0)
                rmse = result.mean_metrics.get('selective_rmse', float('nan'))
                coverage = result.mean_metrics.get('coverage', 0)
                coverage_std = result.std_metrics.get('coverage', 0)

                all_results[config_name] = {
                    'metric': metric,
                    'scale': scale,
                    'threshold': threshold,
                    'mae': mae,
                    'mae_std': mae_std,
                    'rmse': rmse,
                    'coverage': coverage,
                    'coverage_std': coverage_std,
                }

                logger.info(f"{config_name}: MAE {mae:.2f} ± {mae_std:.2f}, Coverage {coverage:.1%}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Configuration':<25} {'MAE':<15} {'RMSE':<10} {'Coverage':<12}")
    logger.info("-" * 62)
    for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{config_name:<25} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['rmse']:.2f}     {r['coverage']:.1%}")

    # Best per metric at threshold 0.70
    logger.info("\nBest per metric (threshold=0.70):")
    for metric in metrics:
        metric_results = {k: v for k, v in all_results.items() if v['metric'] == metric and v['threshold'] == 0.70}
        if metric_results:
            best = min(metric_results, key=lambda x: metric_results[x]['mae'] if not np.isnan(metric_results[x]['mae']) else float('inf'))
            logger.info(f"  {metric.upper()}: {best} - MAE {metric_results[best]['mae']:.2f}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("config,metric,scale,threshold,mae,mae_std,rmse,coverage,coverage_std\n")
        for config_name, r in all_results.items():
            f.write(f"{config_name},{r['metric']},{r['scale']},{r['threshold']},{r['mae']:.2f},{r['mae_std']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# C5: Disagreement Metric MAD\n\n")
        f.write("## Objective\n")
        f.write("Test MAD (median absolute deviation) as an alternative to std and range.\n")
        f.write("MAD is more robust to outliers than std.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE | RMSE | Coverage |\n")
        f.write("|---------------|-----|------|----------|\n")
        for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {config_name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['rmse']:.2f} | {r['coverage']:.1%} |\n")

        f.write("\n## Best per Metric\n\n")
        for metric in metrics:
            metric_results = {k: v for k, v in all_results.items() if v['metric'] == metric}
            if metric_results:
                best_key = min(metric_results, key=lambda x: metric_results[x]['mae'] if not np.isnan(metric_results[x]['mae']) else float('inf'))
                best = metric_results[best_key]
                f.write(f"- **{metric.upper()}**: {best_key} - MAE {best['mae']:.2f}, Coverage {best['coverage']:.1%}\n")

        f.write("\n## Analysis\n\n")

        # Compare best of each metric at t=0.70
        best_per_metric = {}
        for metric in metrics:
            metric_results = {k: v for k, v in all_results.items() if v['metric'] == metric and v['threshold'] == 0.70}
            if metric_results:
                best_key = min(metric_results, key=lambda x: metric_results[x]['mae'] if not np.isnan(metric_results[x]['mae']) else float('inf'))
                best_per_metric[metric] = metric_results[best_key]

        if len(best_per_metric) == 3:
            f.write("### Comparison at threshold=0.70\n\n")
            f.write("| Metric | Best Scale | MAE | Coverage |\n")
            f.write("|--------|------------|-----|----------|\n")
            for metric in metrics:
                r = best_per_metric[metric]
                f.write(f"| {metric} | {r['scale']} | {r['mae']:.2f} | {r['coverage']:.1%} |\n")

            # Find overall best
            best_metric = min(best_per_metric, key=lambda x: best_per_metric[x]['mae'])
            best_result = best_per_metric[best_metric]

            f.write(f"\n**Best metric**: `{best_metric}` with scale={best_result['scale']}\n\n")

            # Compare with baseline
            baseline_mae = 2.97
            diff = best_result['mae'] - baseline_mae
            if diff < -0.1:
                f.write(f"MAD/std/range alternatives **improve** over baseline by {-diff:.2f} frames.\n")
            elif diff > 0.1:
                f.write(f"Best result is {diff:.2f} frames worse than baseline (MAE 2.97).\n")
            else:
                f.write(f"Performance is comparable to baseline (MAE 2.97, diff: {diff:+.2f}).\n")

        # Recommendations
        f.write("\n### Recommendations\n\n")
        mad_results = {k: v for k, v in all_results.items() if v['metric'] == 'mad'}
        std_results = {k: v for k, v in all_results.items() if v['metric'] == 'std'}

        if mad_results and std_results:
            best_mad = min(mad_results.values(), key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))
            best_std = min(std_results.values(), key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

            if best_mad['mae'] < best_std['mae'] - 0.1:
                f.write("- **MAD outperforms std** - consider switching to MAD metric\n")
            elif best_std['mae'] < best_mad['mae'] - 0.1:
                f.write("- **std outperforms MAD** - keep using std metric\n")
            else:
                f.write("- **MAD and std are comparable** - std is fine, MAD offers no clear advantage\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
