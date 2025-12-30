#!/usr/bin/env python3
"""
C4: Single Model Baselines with Quality Gating.

Establish individual model + quality-gating baselines.
Test CNN-only, LSTM-only, HGB-only with QualityGatedCombiner.

Usage:
    uv run python scripts/c4_single_models.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from boom_detection.combine import QualityGatedCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='C4: Single Model Baselines')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/c4_single_models'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c4_single_models")

    logger.info("=" * 70)
    logger.info("C4: Single Model Baselines with Quality Gating")
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

    # Models and thresholds to test
    models = ['cnn', 'hgb', 'lstm']
    thresholds = [0.60, 0.65, 0.70, 0.75]

    all_results = {}

    for model in models:
        for threshold in thresholds:
            config_name = f"{model.upper()}-only (t={threshold})"

            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Testing: {config_name}")
            logger.info("=" * 70)

            result = evaluator.cross_validate_selective(
                lambda m=model, t=threshold: BoomDetectionPipeline(
                    frame_models=(m,),
                    combiner=QualityGatedCombiner(threshold=t),
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
                'model': model,
                'threshold': threshold,
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

    logger.info(f"\n{'Configuration':<25} {'MAE':<15} {'RMSE':<10} {'Coverage':<12}")
    logger.info("-" * 62)
    for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{config_name:<25} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['rmse']:.2f}     {r['coverage']:.1%}")

    # Best per model
    for model in models:
        model_results = {k: v for k, v in all_results.items() if v['model'] == model}
        if model_results:
            best = min(model_results, key=lambda x: model_results[x]['mae'] if not np.isnan(model_results[x]['mae']) else float('inf'))
            logger.info(f"\nBest {model.upper()}: {best}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("config,model,threshold,mae,mae_std,rmse,coverage,coverage_std\n")
        for config_name, r in all_results.items():
            f.write(f"{config_name},{r['model']},{r['threshold']},{r['mae']:.2f},{r['mae_std']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# C4: Single Model Baselines with Quality Gating\n\n")
        f.write("## Objective\n")
        f.write("Establish individual model + quality-gating baselines.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE | RMSE | Coverage |\n")
        f.write("|---------------|-----|------|----------|\n")
        for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {config_name} | {r['mae']:.2f} +/- {r['mae_std']:.2f} | {r['rmse']:.2f} | {r['coverage']:.1%} |\n")

        f.write("\n## Best per Model\n\n")
        for model in models:
            model_results = {k: v for k, v in all_results.items() if v['model'] == model}
            if model_results:
                best_key = min(model_results, key=lambda x: model_results[x]['mae'] if not np.isnan(model_results[x]['mae']) else float('inf'))
                best = model_results[best_key]
                f.write(f"- **{model.upper()}**: {best_key} - MAE {best['mae']:.2f}, Coverage {best['coverage']:.1%}\n")

        f.write("\n## Analysis\n\n")

        # Find overall best single model
        valid_results = {k: v for k, v in all_results.items() if not np.isnan(v['mae'])}
        if valid_results:
            overall_best_key = min(valid_results, key=lambda x: valid_results[x]['mae'])
            overall_best = valid_results[overall_best_key]

            f.write(f"**Best single model configuration:** {overall_best_key}\n")
            f.write(f"- MAE: {overall_best['mae']:.2f}\n")
            f.write(f"- Coverage: {overall_best['coverage']:.1%}\n\n")

            # Compare with 3-model baseline
            baseline_mae = 2.97
            diff = overall_best['mae'] - baseline_mae
            f.write(f"**Comparison with 3-model baseline (MAE 2.97):**\n")
            f.write(f"- Single model is {diff:+.2f} frames worse\n")

            if diff < 0.5:
                f.write(f"\n**Single model is competitive** - only {diff:.2f} frames worse.\n")
                f.write("Consider single model for simplicity if MAE difference is acceptable.\n")
            else:
                f.write(f"\n**3-model significantly outperforms** all single models.\n")
                f.write("The ensemble adds meaningful value.\n")

        f.write("\n### Model Ranking\n\n")
        # Rank models at threshold 0.70
        t70_results = {k: v for k, v in all_results.items() if v['threshold'] == 0.70}
        for rank, (k, v) in enumerate(sorted(t70_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')), 1):
            f.write(f"{rank}. {k}: MAE {v['mae']:.2f}\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
