#!/usr/bin/env python3
"""
D4: Ensemble Prediction Methods.

Test different methods for combining model predictions:
- primary_model='cnn' (default)
- primary_model='hgb'
- primary_model='lstm'
- primary_model='median' (median of all predictions)

Usage:
    uv run python scripts/d4_ensemble_prediction.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, CombinerExperiment
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='D4: Ensemble Prediction Methods')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/d4_ensemble_prediction'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("d4_ensemble_prediction")

    logger.info("=" * 70)
    logger.info("D4: Ensemble Prediction Methods")
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

    # Train 3-model once and test different primary_model options
    logger.info("")
    logger.info("Training 3-model pipeline (CNN + HGB + LSTM)...")

    # Create combiner experiment
    experiment = evaluator.create_combiner_experiment(
        lambda: BoomDetectionPipeline(frame_models=('cnn', 'hgb', 'lstm')),
        seeds=args.seeds,
    )

    # Primary model options to test
    primary_models = ['cnn', 'hgb', 'lstm', 'median']

    all_results = {}

    for primary in primary_models:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: primary_model='{primary}'")
        logger.info("=" * 70)

        combiner = ThresholdCombiner(
            primary_model=primary,
            agreement_transform='sqrt',
            disagreement_scale=15.0,
            disagreement_metric='std',
            threshold=0.70,
        )

        result = experiment.evaluate(combiner)

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        rmse = result.mean_metrics.get('selective_rmse', float('nan'))
        coverage = result.mean_metrics.get('coverage', 0)
        coverage_std = result.std_metrics.get('coverage', 0)

        all_results[primary] = {
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

    logger.info(f"\n{'Primary Model':<15} {'MAE':<15} {'RMSE':<10} {'Coverage':<12}")
    logger.info("-" * 52)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{name:<15} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['rmse']:.2f}     {r['coverage']:.1%}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("primary_model,mae,mae_std,rmse,coverage,coverage_std\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['mae']:.2f},{r['mae_std']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# D4: Ensemble Prediction Methods\n\n")
        f.write("## Objective\n")
        f.write("Test different methods for combining model predictions after acceptance.\n\n")

        f.write("## Background\n")
        f.write("The ThresholdCombiner decides acceptance based on quality + model agreement.\n")
        f.write("Once accepted, it returns a prediction using the `primary_model` setting:\n")
        f.write("- **cnn**: Return CNN's prediction (default)\n")
        f.write("- **hgb**: Return HGB's prediction\n")
        f.write("- **lstm**: Return LSTM's prediction\n")
        f.write("- **median**: Return median of all model predictions\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Results\n\n")
        f.write("| Primary Model | MAE | RMSE | Coverage |\n")
        f.write("|---------------|-----|------|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['rmse']:.2f} | {r['coverage']:.1%} |\n")

        # Find best
        best_name = min(all_results, key=lambda x: all_results[x]['mae'] if not np.isnan(all_results[x]['mae']) else float('inf'))
        best = all_results[best_name]
        baseline = all_results['cnn']

        f.write(f"\n## Best Configuration\n\n")
        f.write(f"**{best_name}**: MAE {best['mae']:.2f} ± {best['mae_std']:.2f}\n\n")

        f.write("## Analysis\n\n")

        diff = best['mae'] - baseline['mae']
        if abs(diff) < 0.1:
            f.write(f"**Best option matches baseline (CNN).**\n\n")
            f.write("All primary_model options perform similarly.\n")
            f.write("Recommendation: Keep CNN as primary model.\n")
        elif diff < 0:
            f.write(f"**{best_name} outperforms CNN** by {-diff:.2f} frames.\n\n")
            f.write(f"Recommendation: Switch to primary_model='{best_name}'.\n")
        else:
            f.write(f"**CNN outperforms {best_name}** by {diff:.2f} frames.\n\n")
            f.write(f"Recommendation: Keep CNN as primary model.\n")

        # Check variance across methods
        mae_values = [r['mae'] for r in all_results.values() if not np.isnan(r['mae'])]
        if mae_values:
            mae_range = max(mae_values) - min(mae_values)
            f.write(f"\n### Sensitivity Analysis\n\n")
            f.write(f"MAE range across methods: {mae_range:.2f} frames\n\n")
            if mae_range < 0.3:
                f.write("Primary model choice has **minimal impact** on performance.\n")
                f.write("This suggests models agree well on accepted samples.\n")
            elif mae_range < 0.5:
                f.write("Primary model choice has **moderate impact** on performance.\n")
            else:
                f.write("Primary model choice has **significant impact** on performance.\n")

        f.write("\n### Why Models Differ\n\n")
        f.write("Even with high agreement (std metric), models can differ:\n")
        f.write("1. CNN: Good at local patterns, may miss global context\n")
        f.write("2. HGB: Frame-by-frame, no temporal modeling\n")
        f.write("3. LSTM: Long-range dependencies, but sometimes outliers\n")
        f.write("4. Median: Robust to outliers, compromise between models\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
