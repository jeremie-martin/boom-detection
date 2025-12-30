#!/usr/bin/env python3
"""
E1: CNN Architecture Variations.

Test different CNN architecture configurations:
- hidden_dim: [32, 64, 128]
- kernel_sizes: [(3,7,15), (5,11,21), (7,15,31)]
- dropout: [0.2, 0.3, 0.4]

Usage:
    uv run python scripts/e1_cnn_architecture.py data --seeds 42 43 44
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


class CNNArchPipeline(BoomDetectionPipeline):
    """Pipeline with customizable CNN architecture."""

    def __init__(self, hidden_dim: int = 64, kernel_sizes: tuple = (5, 11, 21),
                 dropout: float = 0.3, **kwargs):
        self.cnn_hidden_dim = hidden_dim
        self.cnn_kernel_sizes = kernel_sizes
        self.cnn_dropout = dropout
        super().__init__(**kwargs)

    def _train_cnn(self, model_key: str, sim_ids, boom_frames, cache, seed):
        """Train CNN with custom architecture."""
        logger.debug("Training CNN '{}' (hidden={}, kernels={}, dropout={})...",
                    model_key, self.cnn_hidden_dim, self.cnn_kernel_sizes, self.cnn_dropout)
        start = time.time()
        model = CNNClassifier(
            n_features=self.n_features,
            hidden_dim=self.cnn_hidden_dim,
            kernel_sizes=self.cnn_kernel_sizes,
            dropout=self.cnn_dropout,
        )
        trainer = SequenceTrainer(
            model, lr=0.5e-3, epochs=30, patience=5, batch_size=4, augment=False,
            seed=seed, normalize=self.normalize_features,
        )
        trainer.fit(sim_ids, boom_frames, cache)
        self.trained_models[model_key] = model
        self.trainers[model_key] = trainer
        logger.debug("CNN '{}' trained in {:.1f}s", model_key, time.time() - start)


def main():
    parser = argparse.ArgumentParser(description='E1: CNN Architecture Variations')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/e1_cnn_architecture'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("e1_cnn_architecture")

    logger.info("=" * 70)
    logger.info("E1: CNN Architecture Variations")
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

    # Architectures to test
    hidden_dims = [32, 64, 128]
    kernel_sizes_list = [(3, 7, 15), (5, 11, 21), (7, 15, 31)]
    dropouts = [0.2, 0.3, 0.4]

    # Standard combiner
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    all_results = {}

    # Test hidden_dim (with defaults for others)
    logger.info("\nTesting hidden_dim variations...")
    for hidden_dim in hidden_dims:
        config_name = f"hidden={hidden_dim}"
        logger.info(f"  Testing: {config_name}")

        result = evaluator.cross_validate_selective(
            lambda hd=hidden_dim: CNNArchPipeline(
                hidden_dim=hd,
                kernel_sizes=(5, 11, 21),  # default
                dropout=0.3,  # default
                frame_models=('cnn', 'hgb', 'lstm'),
                combiner=combiner,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        coverage = result.mean_metrics.get('coverage', 0)

        all_results[config_name] = {
            'hidden_dim': hidden_dim,
            'kernel_sizes': '(5,11,21)',
            'dropout': 0.3,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': coverage,
        }
        logger.info(f"    MAE: {mae:.2f} ± {mae_std:.2f}, Coverage: {coverage:.1%}")

    # Test kernel_sizes (with hidden_dim=64)
    logger.info("\nTesting kernel_sizes variations...")
    for kernel_sizes in kernel_sizes_list:
        ks_str = ','.join(map(str, kernel_sizes))
        config_name = f"kernels=({ks_str})"
        logger.info(f"  Testing: {config_name}")

        result = evaluator.cross_validate_selective(
            lambda ks=kernel_sizes: CNNArchPipeline(
                hidden_dim=64,  # default
                kernel_sizes=ks,
                dropout=0.3,  # default
                frame_models=('cnn', 'hgb', 'lstm'),
                combiner=combiner,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        coverage = result.mean_metrics.get('coverage', 0)

        all_results[config_name] = {
            'hidden_dim': 64,
            'kernel_sizes': f'({ks_str})',
            'dropout': 0.3,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': coverage,
        }
        logger.info(f"    MAE: {mae:.2f} ± {mae_std:.2f}, Coverage: {coverage:.1%}")

    # Test dropout (with hidden_dim=64)
    logger.info("\nTesting dropout variations...")
    for dropout in dropouts:
        config_name = f"dropout={dropout}"
        logger.info(f"  Testing: {config_name}")

        result = evaluator.cross_validate_selective(
            lambda d=dropout: CNNArchPipeline(
                hidden_dim=64,  # default
                kernel_sizes=(5, 11, 21),  # default
                dropout=d,
                frame_models=('cnn', 'hgb', 'lstm'),
                combiner=combiner,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )

        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        coverage = result.mean_metrics.get('coverage', 0)

        all_results[config_name] = {
            'hidden_dim': 64,
            'kernel_sizes': '(5,11,21)',
            'dropout': dropout,
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

    logger.info(f"\n{'Configuration':<25} {'MAE':<15} {'Coverage':<12}")
    logger.info("-" * 52)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{name:<25} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['coverage']:.1%}")

    best_name = min(all_results, key=lambda x: all_results[x]['mae'] if not np.isnan(all_results[x]['mae']) else float('inf'))
    logger.info(f"\nBest: {best_name}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("config,hidden_dim,kernel_sizes,dropout,mae,mae_std,coverage\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['hidden_dim']},{r['kernel_sizes']},{r['dropout']},{r['mae']:.2f},{r['mae_std']:.2f},{r['coverage']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# E1: CNN Architecture Variations\n\n")
        f.write("## Objective\n")
        f.write("Test different CNN architecture configurations.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Parameters Tested\n")
        f.write(f"- hidden_dim: {hidden_dims}\n")
        f.write(f"- kernel_sizes: {kernel_sizes_list}\n")
        f.write(f"- dropout: {dropouts}\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE | Coverage |\n")
        f.write("|---------------|-----|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['coverage']:.1%} |\n")

        best = all_results[best_name]
        baseline = all_results.get('hidden=64', all_results.get('kernels=(5,11,21)', {}))

        f.write(f"\n## Best Configuration\n\n")
        f.write(f"**{best_name}**: MAE {best['mae']:.2f} ± {best['mae_std']:.2f}\n\n")

        f.write("## Analysis\n\n")

        # Compare to baseline
        baseline_mae = 2.97  # Known baseline
        diff = best['mae'] - baseline_mae

        if abs(diff) < 0.2:
            f.write(f"**Architecture changes have minimal impact** (within 0.2 of baseline).\n")
            f.write("Recommendation: Keep current architecture (hidden=64, kernels=(5,11,21), dropout=0.3).\n")
        elif diff < 0:
            f.write(f"**{best_name} improves over baseline** by {-diff:.2f} frames.\n")
            f.write("Recommendation: Consider updating CNN architecture.\n")
        else:
            f.write(f"**Baseline outperforms** best sweep result by {diff:.2f} frames.\n")
            f.write("Recommendation: Keep current architecture.\n")

        # Per-parameter analysis
        f.write("\n### Per-Parameter Sensitivity\n\n")

        # Hidden dim
        hd_results = {k: v for k, v in all_results.items() if 'hidden=' in k}
        if hd_results:
            mae_range = max(v['mae'] for v in hd_results.values() if not np.isnan(v['mae'])) - \
                       min(v['mae'] for v in hd_results.values() if not np.isnan(v['mae']))
            f.write(f"**hidden_dim**: MAE range = {mae_range:.2f} frames\n")

        # Kernel sizes
        ks_results = {k: v for k, v in all_results.items() if 'kernels=' in k}
        if ks_results:
            mae_range = max(v['mae'] for v in ks_results.values() if not np.isnan(v['mae'])) - \
                       min(v['mae'] for v in ks_results.values() if not np.isnan(v['mae']))
            f.write(f"**kernel_sizes**: MAE range = {mae_range:.2f} frames\n")

        # Dropout
        do_results = {k: v for k, v in all_results.items() if 'dropout=' in k}
        if do_results:
            mae_range = max(v['mae'] for v in do_results.values() if not np.isnan(v['mae'])) - \
                       min(v['mae'] for v in do_results.values() if not np.isnan(v['mae']))
            f.write(f"**dropout**: MAE range = {mae_range:.2f} frames\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
