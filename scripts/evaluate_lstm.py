#!/usr/bin/env python3
"""
Evaluate LSTM model for boom frame prediction.

Compares LSTM vs CNN vs HGB individual performance using multi-seed
cross-validation. This establishes baseline metrics before integrating
LSTM into the 3-model pipeline.

Uses memory-safe two-phase pattern:
- Phase 1: Extract features (requires raw data)
- Phase 2: Evaluate from disk cache (releases raw data)

Usage:
    uv run python scripts/evaluate_lstm.py
    uv run python scripts/evaluate_lstm.py --quick  # Single seed for dev
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.evaluation import CachedEvaluator
from boom_detection.frame_models import FrameLevelClassifier
from boom_detection.sequence_models import (
    CNNClassifier,
    LSTMClassifier,
    TransformerClassifier,
    SequenceTrainer,
)
from boom_detection.logging_config import log_memory_usage


def create_cnn_factory(n_features: int):
    """Factory for CNN model (same config as production pipeline)."""
    def factory():
        model = CNNClassifier(
            n_features=n_features,
            hidden_dim=64,
            kernel_sizes=(5, 11, 21),
        )
        return SequenceTrainer(
            model,
            lr=0.5e-3,
            epochs=30,
            patience=5,
            batch_size=4,
            augment=False,
        )
    return factory


def create_lstm_factory(n_features: int):
    """Factory for LSTM model."""
    def factory():
        model = LSTMClassifier(
            n_features=n_features,
            hidden_dim=128,
            n_layers=2,
            dropout=0.3,
        )
        return SequenceTrainer(
            model,
            lr=0.5e-3,
            epochs=30,
            patience=5,
            batch_size=4,
            augment=False,
        )
    return factory


def create_transformer_factory(n_features: int):
    """Factory for Transformer model."""
    def factory():
        model = TransformerClassifier(
            n_features=n_features,
            hidden_dim=128,
            n_heads=4,
            n_layers=2,
            dropout=0.3,
        )
        return SequenceTrainer(
            model,
            lr=0.5e-3,
            epochs=30,
            patience=5,
            batch_size=4,
            augment=False,
        )
    return factory


def evaluate_models(
    dataset,
    cache: FeatureCache,
    seeds: list[int],
    include_transformer: bool = False,
) -> dict:
    """Evaluate all models and return results."""
    evaluator = CachedEvaluator(dataset, cache)
    n_features = cache[dataset.annotations[0].id].shape[1]
    results = {}

    print(f"\n{'Model':<15} {'MAE':<20} {'Within 5':<15} {'Within 10':<15}")
    print("-" * 65)

    # HGB baseline
    print("Evaluating HGB...", end=" ", flush=True)
    start = time.time()
    hgb_result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
        verbose=False,
    )
    results['hgb'] = {
        'mae': hgb_result.mean_metrics['mae'],
        'mae_std': hgb_result.std_metrics['mae'],
        'within_5': hgb_result.mean_metrics.get('within_5', 0),
        'within_10': hgb_result.mean_metrics.get('within_10', 0),
        'time': time.time() - start,
    }
    mae_str = f"{results['hgb']['mae']:.2f} ± {results['hgb']['mae_std']:.2f}"
    print(f"\r{'HGB':<15} {mae_str:<20} {results['hgb']['within_5']:.1%}           "
          f"{results['hgb']['within_10']:.1%}           ({results['hgb']['time']:.1f}s)")

    # CNN
    print("Evaluating CNN...", end=" ", flush=True)
    start = time.time()
    cnn_result = evaluator.cross_validate(
        create_cnn_factory(n_features),
        seeds=seeds,
        verbose=False,
    )
    results['cnn'] = {
        'mae': cnn_result.mean_metrics['mae'],
        'mae_std': cnn_result.std_metrics['mae'],
        'within_5': cnn_result.mean_metrics.get('within_5', 0),
        'within_10': cnn_result.mean_metrics.get('within_10', 0),
        'time': time.time() - start,
    }
    mae_str = f"{results['cnn']['mae']:.2f} ± {results['cnn']['mae_std']:.2f}"
    print(f"\r{'CNN':<15} {mae_str:<20} {results['cnn']['within_5']:.1%}           "
          f"{results['cnn']['within_10']:.1%}           ({results['cnn']['time']:.1f}s)")

    # LSTM
    print("Evaluating LSTM...", end=" ", flush=True)
    start = time.time()
    lstm_result = evaluator.cross_validate(
        create_lstm_factory(n_features),
        seeds=seeds,
        verbose=False,
    )
    results['lstm'] = {
        'mae': lstm_result.mean_metrics['mae'],
        'mae_std': lstm_result.std_metrics['mae'],
        'within_5': lstm_result.mean_metrics.get('within_5', 0),
        'within_10': lstm_result.mean_metrics.get('within_10', 0),
        'time': time.time() - start,
    }
    mae_str = f"{results['lstm']['mae']:.2f} ± {results['lstm']['mae_std']:.2f}"
    print(f"\r{'LSTM':<15} {mae_str:<20} {results['lstm']['within_5']:.1%}           "
          f"{results['lstm']['within_10']:.1%}           ({results['lstm']['time']:.1f}s)")

    # Transformer (optional)
    if include_transformer:
        print("Evaluating Transformer...", end=" ", flush=True)
        start = time.time()
        transformer_result = evaluator.cross_validate(
            create_transformer_factory(n_features),
            seeds=seeds,
            verbose=False,
        )
        results['transformer'] = {
            'mae': transformer_result.mean_metrics['mae'],
            'mae_std': transformer_result.std_metrics['mae'],
            'within_5': transformer_result.mean_metrics.get('within_5', 0),
            'within_10': transformer_result.mean_metrics.get('within_10', 0),
            'time': time.time() - start,
        }
        mae_str = f"{results['transformer']['mae']:.2f} ± {results['transformer']['mae_std']:.2f}"
        print(f"\r{'Transformer':<15} {mae_str:<20} {results['transformer']['within_5']:.1%}           "
              f"{results['transformer']['within_10']:.1%}           ({results['transformer']['time']:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTM model')
    parser.add_argument('--data', type=Path, default=Path('data'),
                        help='Path to data directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick single-seed evaluation for development')
    parser.add_argument('--transformer', action='store_true',
                        help='Also evaluate Transformer model')
    args = parser.parse_args()

    seeds = [42] if args.quick else [42, 43, 44, 45, 46]
    mode = "quick (1 seed)" if args.quick else "robust (5 seeds)"

    print("=" * 70)
    print("LSTM MODEL EVALUATION")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Seeds: {seeds}")

    # Phase 1: Load data and extract features
    print("\nPhase 1: Loading data and extracting features...")
    dataset = load_dataset(args.data, verbose=False)
    print(f"Loaded {len(dataset)} simulations")
    log_memory_usage("after loading")

    config = FeatureConfig(max_pendulums=2000, include_caustic=False)
    cache = FeatureCache(config=config, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            print(f"  Extracting features ({loaded}/{len(sim_ids)} cached)...")
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            print(f"  Using cached features ({loaded} sims)")
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        print("  Extracting features...")
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    # Phase 2: Evaluate models
    print("\nPhase 2: Evaluating models...")
    results = evaluate_models(dataset, cache, seeds, include_transformer=args.transformer)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_model = min(results.keys(), key=lambda k: results[k]['mae'])
    print(f"\nBest model: {best_model.upper()} (MAE: {results[best_model]['mae']:.2f})")

    print("\nRanking by MAE:")
    for i, (model, r) in enumerate(sorted(results.items(), key=lambda x: x[1]['mae']), 1):
        print(f"  {i}. {model.upper()}: {r['mae']:.2f} ± {r['mae_std']:.2f}")

    if args.quick:
        print("\nNote: This is a quick single-seed result.")
        print("      Run without --quick for robust multi-seed evaluation.")


if __name__ == '__main__':
    main()
