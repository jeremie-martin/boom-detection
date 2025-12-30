#!/usr/bin/env python3
"""
C7: CNN Learning Rate Sensitivity.

Test different learning rates for the CNN model to find optimal setting.
Current default: 0.5e-3 (5e-4)

Usage:
    uv run python scripts/c7_cnn_learning_rate.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.sequence_models import CNNClassifier, SequenceTrainer
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


class CNNLearningRatePipeline(BoomDetectionPipeline):
    """Modified pipeline that allows customizing CNN learning rate."""

    def __init__(self, cnn_lr: float = 0.5e-3, **kwargs):
        self.cnn_lr = cnn_lr
        super().__init__(**kwargs)

    def _train_cnn(self, model_key: str, sim_ids, boom_frames, cache, seed):
        """Train CNN model with custom learning rate."""
        logger.debug("Training CNN '{}' with lr={}...", model_key, self.cnn_lr)
        start = time.time()
        model = CNNClassifier(
            n_features=self.n_features,
            hidden_dim=64,
            kernel_sizes=(5, 11, 21)
        )
        trainer = SequenceTrainer(
            model, lr=self.cnn_lr, epochs=30, patience=5, batch_size=4, augment=False,
            seed=seed, normalize=self.normalize_features,
        )
        trainer.fit(sim_ids, boom_frames, cache)
        self.trained_models[model_key] = model
        self.trainers[model_key] = trainer
        logger.debug("CNN '{}' trained in {:.1f}s", model_key, time.time() - start)


def main():
    parser = argparse.ArgumentParser(description='C7: CNN Learning Rate Sensitivity')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/c7_cnn_learning_rate'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("c7_cnn_learning_rate")

    logger.info("=" * 70)
    logger.info("C7: CNN Learning Rate Sensitivity")
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

    # Learning rates to test
    learning_rates = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

    # Standard combiner for fair comparison
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    all_results = {}

    for lr in learning_rates:
        lr_name = f"lr={lr:.0e}"

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: CNN {lr_name}")
        logger.info("=" * 70)

        result = evaluator.cross_validate_selective(
            lambda lr_val=lr: CNNLearningRatePipeline(
                cnn_lr=lr_val,
                frame_models=('cnn', 'hgb', 'lstm'),
                combiner=combiner,
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

        all_results[lr_name] = {
            'lr': lr,
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

    logger.info(f"\n{'Learning Rate':<15} {'MAE':<15} {'RMSE':<10} {'Coverage':<12}")
    logger.info("-" * 52)
    for lr_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{lr_name:<15} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['rmse']:.2f}     {r['coverage']:.1%}")

    best_lr = min(all_results, key=lambda x: all_results[x]['mae'] if not np.isnan(all_results[x]['mae']) else float('inf'))
    logger.info(f"\nBest: {best_lr}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("learning_rate,mae,mae_std,rmse,coverage,coverage_std\n")
        for lr_name, r in all_results.items():
            f.write(f"{r['lr']},{r['mae']:.2f},{r['mae_std']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# C7: CNN Learning Rate Sensitivity\n\n")
        f.write("## Objective\n")
        f.write("Test different learning rates for the CNN model to find optimal setting.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Results\n\n")
        f.write("| Learning Rate | MAE | RMSE | Coverage |\n")
        f.write("|---------------|-----|------|----------|\n")
        for lr_name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {lr_name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['rmse']:.2f} | {r['coverage']:.1%} |\n")

        f.write(f"\n## Best Configuration\n\n**{best_lr}**\n\n")

        f.write("## Analysis\n\n")

        # Current default is 5e-4
        default_result = all_results.get('lr=5e-04', {})
        best_result = all_results[best_lr]

        if not np.isnan(default_result.get('mae', float('nan'))):
            diff = best_result['mae'] - default_result['mae']
            if abs(diff) < 0.2:
                f.write(f"Current default (5e-4) is optimal or near-optimal.\n")
                f.write(f"**Recommendation:** Keep current learning rate.\n")
            elif diff < 0:
                f.write(f"**{best_lr}** improves over default (5e-4) by {-diff:.2f} frames.\n")
                f.write(f"**Recommendation:** Switch to {best_lr}.\n")
            else:
                f.write(f"Default (5e-4) outperforms {best_lr} by {diff:.2f} frames.\n")
                f.write(f"**Recommendation:** Keep current learning rate.\n")

        # Check for stability
        f.write("\n### Sensitivity Analysis\n\n")
        mae_values = [r['mae'] for r in all_results.values() if not np.isnan(r['mae'])]
        if mae_values:
            mae_range = max(mae_values) - min(mae_values)
            if mae_range < 0.5:
                f.write(f"CNN is **relatively insensitive** to learning rate (MAE range: {mae_range:.2f}).\n")
            elif mae_range < 1.0:
                f.write(f"CNN shows **moderate sensitivity** to learning rate (MAE range: {mae_range:.2f}).\n")
            else:
                f.write(f"CNN is **highly sensitive** to learning rate (MAE range: {mae_range:.2f}).\n")

        # Compare with baseline
        baseline_mae = 2.97
        f.write(f"\n### Comparison with Baseline\n\n")
        f.write(f"Baseline (std/s=15/t=0.70): MAE 2.97\n")
        f.write(f"Best with LR sweep: MAE {best_result['mae']:.2f}\n")
        diff = best_result['mae'] - baseline_mae
        if abs(diff) < 0.1:
            f.write(f"\nResults are consistent with baseline.\n")
        elif diff < 0:
            f.write(f"\nLR sweep found {-diff:.2f} frame improvement.\n")
        else:
            f.write(f"\nResults are {diff:.2f} frames worse (variance likely).\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
