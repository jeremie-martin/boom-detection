#!/usr/bin/env python3
"""
E2: LSTM Architecture Variations.

Test different LSTM architecture configurations:
- hidden_dim: [32, 64, 128]
- n_layers: [1, 2, 3]
- dropout: [0.2, 0.3, 0.4]

Usage:
    uv run python scripts/e2_lstm_architecture.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.sequence_models import LSTMClassifier, SequenceTrainer
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


class LSTMArchPipeline(BoomDetectionPipeline):
    """Pipeline with customizable LSTM architecture."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2,
                 dropout: float = 0.3, **kwargs):
        self.lstm_hidden_dim = hidden_dim
        self.lstm_n_layers = n_layers
        self.lstm_dropout = dropout
        super().__init__(**kwargs)

    def _train_lstm(self, model_key: str, sim_ids, boom_frames, cache, seed):
        """Train LSTM with custom architecture."""
        logger.debug("Training LSTM '{}' (hidden={}, layers={}, dropout={})...",
                    model_key, self.lstm_hidden_dim, self.lstm_n_layers, self.lstm_dropout)
        start = time.time()
        model = LSTMClassifier(
            n_features=self.n_features,
            hidden_dim=self.lstm_hidden_dim,
            n_layers=self.lstm_n_layers,
            dropout=self.lstm_dropout,
        )
        trainer = SequenceTrainer(
            model, lr=0.5e-3, epochs=30, patience=5, batch_size=4, augment=False,
            seed=seed, normalize=self.normalize_features,
        )
        trainer.fit(sim_ids, boom_frames, cache)
        self.trained_models[model_key] = model
        self.trainers[model_key] = trainer
        logger.debug("LSTM '{}' trained in {:.1f}s", model_key, time.time() - start)


def main():
    parser = argparse.ArgumentParser(description='E2: LSTM Architecture Variations')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/e2_lstm_architecture'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("e2_lstm_architecture")

    logger.info("=" * 70)
    logger.info("E2: LSTM Architecture Variations")
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
    n_layers_list = [1, 2, 3]
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
            lambda hd=hidden_dim: LSTMArchPipeline(
                hidden_dim=hd,
                n_layers=2,  # default
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
            'n_layers': 2,
            'dropout': 0.3,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': coverage,
        }
        logger.info(f"    MAE: {mae:.2f} ± {mae_std:.2f}, Coverage: {coverage:.1%}")

    # Test n_layers (with hidden_dim=64)
    logger.info("\nTesting n_layers variations...")
    for n_layers in n_layers_list:
        config_name = f"layers={n_layers}"
        logger.info(f"  Testing: {config_name}")

        result = evaluator.cross_validate_selective(
            lambda nl=n_layers: LSTMArchPipeline(
                hidden_dim=64,  # default
                n_layers=nl,
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
            'n_layers': n_layers,
            'dropout': 0.3,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': coverage,
        }
        logger.info(f"    MAE: {mae:.2f} ± {mae_std:.2f}, Coverage: {coverage:.1%}")

    # Test dropout (with hidden_dim=64, n_layers=2)
    logger.info("\nTesting dropout variations...")
    for dropout in dropouts:
        config_name = f"dropout={dropout}"
        logger.info(f"  Testing: {config_name}")

        result = evaluator.cross_validate_selective(
            lambda d=dropout: LSTMArchPipeline(
                hidden_dim=64,  # default
                n_layers=2,  # default
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
            'n_layers': 2,
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
        f.write("config,hidden_dim,n_layers,dropout,mae,mae_std,coverage\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['hidden_dim']},{r['n_layers']},{r['dropout']},{r['mae']:.2f},{r['mae_std']:.2f},{r['coverage']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# E2: LSTM Architecture Variations\n\n")
        f.write("## Objective\n")
        f.write("Test different LSTM architecture configurations.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Parameters Tested\n")
        f.write(f"- hidden_dim: {hidden_dims}\n")
        f.write(f"- n_layers: {n_layers_list}\n")
        f.write(f"- dropout: {dropouts}\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE | Coverage |\n")
        f.write("|---------------|-----|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['coverage']:.1%} |\n")

        best = all_results[best_name]
        baseline_mae = 2.97  # Known baseline

        f.write(f"\n## Best Configuration\n\n")
        f.write(f"**{best_name}**: MAE {best['mae']:.2f} ± {best['mae_std']:.2f}\n\n")

        f.write("## Analysis\n\n")

        # Compare to baseline
        diff = best['mae'] - baseline_mae

        if abs(diff) < 0.2:
            f.write(f"**LSTM architecture changes have minimal impact** (within 0.2 of baseline).\n")
            f.write("Recommendation: Keep current LSTM architecture (hidden=64, layers=2, dropout=0.3).\n")
        elif diff < 0:
            f.write(f"**{best_name} improves over baseline** by {-diff:.2f} frames.\n")
            f.write("Recommendation: Consider updating LSTM architecture.\n")
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

        # n_layers
        nl_results = {k: v for k, v in all_results.items() if 'layers=' in k}
        if nl_results:
            mae_range = max(v['mae'] for v in nl_results.values() if not np.isnan(v['mae'])) - \
                       min(v['mae'] for v in nl_results.values() if not np.isnan(v['mae']))
            f.write(f"**n_layers**: MAE range = {mae_range:.2f} frames\n")

        # Dropout
        do_results = {k: v for k, v in all_results.items() if 'dropout=' in k}
        if do_results:
            mae_range = max(v['mae'] for v in do_results.values() if not np.isnan(v['mae'])) - \
                       min(v['mae'] for v in do_results.values() if not np.isnan(v['mae']))
            f.write(f"**dropout**: MAE range = {mae_range:.2f} frames\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
