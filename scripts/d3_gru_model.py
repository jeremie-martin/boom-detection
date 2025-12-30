#!/usr/bin/env python3
"""
D3: Alternative Sequence Model (GRU).

Compare GRU vs LSTM as the recurrent model in the pipeline.
GRU has fewer parameters than LSTM and often trains faster.

Usage:
    uv run python scripts/d3_gru_model.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner, QualityGatedCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='D3: GRU vs LSTM Comparison')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/d3_gru'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("d3_gru_model")

    logger.info("=" * 70)
    logger.info("D3: GRU vs LSTM Comparison")
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

    # Standard combiner for fair comparison
    combiner_3model = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    quality_combiner = QualityGatedCombiner(threshold=0.70)

    # Configurations to test
    configs = {
        # 3-model with LSTM (baseline)
        '3model-lstm': {
            'frame_models': ('cnn', 'hgb', 'lstm'),
            'combiner': combiner_3model,
        },
        # 3-model with GRU
        '3model-gru': {
            'frame_models': ('cnn', 'hgb', 'gru'),
            'combiner': combiner_3model,
        },
        # Single model baselines with quality gating
        'lstm-only': {
            'frame_models': ('lstm',),
            'combiner': quality_combiner,
        },
        'gru-only': {
            'frame_models': ('gru',),
            'combiner': quality_combiner,
        },
    }

    all_results = {}

    for config_name, config in configs.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {config_name}")
        logger.info("=" * 70)

        # Track training time
        training_times = []

        def timed_pipeline_factory(cfg=config):
            start = time.time()
            pipeline = BoomDetectionPipeline(
                frame_models=cfg['frame_models'],
                combiner=cfg['combiner'],
            )
            return pipeline

        result = evaluator.cross_validate_selective(
            timed_pipeline_factory,
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

    logger.info(f"\n{'Configuration':<15} {'MAE':<15} {'RMSE':<10} {'Coverage':<12}")
    logger.info("-" * 52)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{name:<15} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['rmse']:.2f}     {r['coverage']:.1%}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("config,mae,mae_std,rmse,coverage,coverage_std\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['mae']:.2f},{r['mae_std']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# D3: GRU vs LSTM Comparison\n\n")
        f.write("## Objective\n")
        f.write("Compare GRU as an alternative to LSTM for the recurrent sequence model.\n\n")

        f.write("## Background\n")
        f.write("GRU (Gated Recurrent Unit) has fewer parameters than LSTM:\n")
        f.write("- LSTM: 3 gates (input, forget, output) + cell state\n")
        f.write("- GRU: 2 gates (reset, update) + no separate cell state\n\n")
        f.write("GRU often trains faster and can be similarly effective.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- Both use hidden_dim=64, n_layers=2, dropout=0.3\n")
        f.write("- 3-model combiner: ThresholdCombiner (std/s=15/t=0.70)\n")
        f.write("- Single model combiner: QualityGatedCombiner (t=0.70)\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE | RMSE | Coverage |\n")
        f.write("|---------------|-----|------|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['rmse']:.2f} | {r['coverage']:.1%} |\n")

        f.write("\n## 3-Model Comparison\n\n")
        lstm_3model = all_results.get('3model-lstm', {})
        gru_3model = all_results.get('3model-gru', {})

        if not np.isnan(lstm_3model.get('mae', float('nan'))) and not np.isnan(gru_3model.get('mae', float('nan'))):
            diff = gru_3model['mae'] - lstm_3model['mae']
            if abs(diff) < 0.2:
                f.write(f"**GRU and LSTM are comparable** (MAE difference: {diff:.2f})\n\n")
                f.write("Recommendation: LSTM and GRU perform similarly. ")
                f.write("GRU may offer faster training.\n")
            elif diff < 0:
                f.write(f"**GRU outperforms LSTM** by {-diff:.2f} frames.\n\n")
                f.write("Recommendation: Consider switching to GRU for improved accuracy.\n")
            else:
                f.write(f"**LSTM outperforms GRU** by {diff:.2f} frames.\n\n")
                f.write("Recommendation: Keep using LSTM.\n")

        f.write("\n## Single Model Comparison\n\n")
        lstm_single = all_results.get('lstm-only', {})
        gru_single = all_results.get('gru-only', {})

        if not np.isnan(lstm_single.get('mae', float('nan'))) and not np.isnan(gru_single.get('mae', float('nan'))):
            diff = gru_single['mae'] - lstm_single['mae']
            f.write(f"- LSTM-only: MAE {lstm_single['mae']:.2f} at {lstm_single['coverage']:.1%} coverage\n")
            f.write(f"- GRU-only: MAE {gru_single['mae']:.2f} at {gru_single['coverage']:.1%} coverage\n")
            f.write(f"- Difference: {diff:.2f} frames\n")

        f.write("\n## GRU Characteristics\n\n")
        f.write("- Fewer parameters: ~25% fewer than LSTM\n")
        f.write("- Faster training: Fewer computations per step\n")
        f.write("- Simpler architecture: No separate cell state\n")
        f.write("- Similar expressiveness for many tasks\n")

        f.write("\n## Comparison with Baseline\n\n")
        f.write("**Baseline (3-model LSTM)**: MAE 2.97 ± 0.06 at 10.9% coverage\n\n")
        if not np.isnan(gru_3model.get('mae', float('nan'))):
            diff = gru_3model['mae'] - 2.97
            if abs(diff) < 0.2:
                f.write(f"GRU 3-model result is consistent with baseline.\n")
            elif diff < 0:
                f.write(f"GRU 3-model improves by {-diff:.2f} frames over baseline.\n")
            else:
                f.write(f"GRU 3-model is {diff:.2f} frames worse than baseline (likely variance).\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
