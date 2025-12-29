#!/usr/bin/env python3
"""
Evaluate 3-model (CNN + HGB + LSTM) pipeline vs 2-model baseline.

Compares selective prediction performance with different model configurations
to determine if adding LSTM improves confidence estimation.

Uses memory-safe two-phase pattern:
- Phase 1: Extract features (requires raw data)
- Phase 2: Evaluate from disk cache (releases raw data)

Usage:
    uv run python scripts/evaluate_3model_pipeline.py
    uv run python scripts/evaluate_3model_pipeline.py --quick  # Single seed for dev
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.logging_config import log_memory_usage


def create_2model_factory(agreement_scale: float = 15.0):
    """Factory for 2-model (CNN + HGB) pipeline."""
    def factory():
        return BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),
            primary_model='cnn',
            accept_threshold=0.60,
            agreement_formula='sqrt',
            agreement_scale=agreement_scale,
            calibrate_quality=True,
        )
    return factory


def create_3model_factory(agreement_scale: float = 10.0):
    """Factory for 3-model (CNN + HGB + LSTM) pipeline."""
    def factory():
        return BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            primary_model='cnn',
            accept_threshold=0.60,
            agreement_formula='sqrt',
            agreement_scale=agreement_scale,
            calibrate_quality=True,
        )
    return factory


def evaluate_configurations(
    dataset,
    cache: FeatureCache,
    seeds: list[int],
    scales_2model: list[float],
    scales_3model: list[float],
) -> dict:
    """Evaluate different pipeline configurations and return results."""
    evaluator = CachedEvaluator(dataset, cache)
    results = {}

    print(f"\n{'Config':<25} {'Selective MAE':<20} {'Coverage':<15} {'Time':<10}")
    print("-" * 70)

    # 2-model configurations
    for scale in scales_2model:
        config_name = f"2-model (scale={scale:.0f})"
        print(f"Evaluating {config_name}...", end=" ", flush=True)
        start = time.time()

        result = evaluator.cross_validate_selective(
            create_2model_factory(agreement_scale=scale),
            seeds=seeds,
            verbose=False,
        )

        elapsed = time.time() - start
        results[config_name] = {
            'selective_mae': result.mean_metrics['selective_mae'],
            'selective_mae_std': result.std_metrics['selective_mae'],
            'coverage': result.mean_metrics['coverage'],
            'coverage_std': result.std_metrics['coverage'],
            'frame_models': ('cnn', 'hgb'),
            'agreement_scale': scale,
            'time': elapsed,
        }

        mae_str = f"{results[config_name]['selective_mae']:.2f} ± {results[config_name]['selective_mae_std']:.2f}"
        cov_str = f"{results[config_name]['coverage']:.1%} ± {results[config_name]['coverage_std']:.1%}"
        print(f"\r{config_name:<25} {mae_str:<20} {cov_str:<15} {elapsed:.0f}s")

    # 3-model configurations
    for scale in scales_3model:
        config_name = f"3-model (scale={scale:.0f})"
        print(f"Evaluating {config_name}...", end=" ", flush=True)
        start = time.time()

        result = evaluator.cross_validate_selective(
            create_3model_factory(agreement_scale=scale),
            seeds=seeds,
            verbose=False,
        )

        elapsed = time.time() - start
        results[config_name] = {
            'selective_mae': result.mean_metrics['selective_mae'],
            'selective_mae_std': result.std_metrics['selective_mae'],
            'coverage': result.mean_metrics['coverage'],
            'coverage_std': result.std_metrics['coverage'],
            'frame_models': ('cnn', 'hgb', 'lstm'),
            'agreement_scale': scale,
            'time': elapsed,
        }

        mae_str = f"{results[config_name]['selective_mae']:.2f} ± {results[config_name]['selective_mae_std']:.2f}"
        cov_str = f"{results[config_name]['coverage']:.1%} ± {results[config_name]['coverage_std']:.1%}"
        print(f"\r{config_name:<25} {mae_str:<20} {cov_str:<15} {elapsed:.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3-model pipeline')
    parser.add_argument('--data', type=Path, default=Path('data'),
                        help='Path to data directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick single-seed evaluation for development')
    args = parser.parse_args()

    seeds = [42] if args.quick else [42, 43, 44]
    mode = "quick (1 seed)" if args.quick else "robust (3 seeds)"

    # Scales to evaluate
    # Lower scale = more selective (rejects when disagreement is lower)
    # For 2-model: use documented configs for comparison
    # For 3-model: test range from selective to permissive
    scales_2model = [5.0, 10.0, 15.0]  # Selective, default, balanced
    scales_3model = [5.0, 8.0, 10.0, 12.0, 15.0]  # Wide range to find optimal

    print("=" * 70)
    print("3-MODEL PIPELINE EVALUATION")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Seeds: {seeds}")
    print(f"2-model scales: {scales_2model}")
    print(f"3-model scales: {scales_3model}")

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

    # Phase 2: Evaluate configurations
    print("\nPhase 2: Evaluating configurations...")
    results = evaluate_configurations(dataset, cache, seeds, scales_2model, scales_3model)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best configuration at similar coverage levels
    print("\nAll configurations (sorted by selective MAE):")
    for config_name, r in sorted(results.items(), key=lambda x: x[1]['selective_mae']):
        mae_str = f"{r['selective_mae']:.2f} ± {r['selective_mae_std']:.2f}"
        cov_str = f"{r['coverage']:.1%}"
        print(f"  {config_name:<25} MAE: {mae_str:<15} Coverage: {cov_str}")

    # Compare 2-model vs best 3-model
    baseline = results.get("2-model (scale=15)")
    if baseline:
        print(f"\n2-model baseline: MAE {baseline['selective_mae']:.2f} at {baseline['coverage']:.1%} coverage")

    best_3model = min(
        [(k, v) for k, v in results.items() if '3-model' in k],
        key=lambda x: x[1]['selective_mae'],
        default=None,
    )
    if best_3model:
        name, r = best_3model
        print(f"Best 3-model:     MAE {r['selective_mae']:.2f} at {r['coverage']:.1%} coverage ({name})")

        if baseline:
            mae_diff = r['selective_mae'] - baseline['selective_mae']
            cov_diff = r['coverage'] - baseline['coverage']
            direction = "better" if mae_diff < 0 else "worse"
            print(f"\nDifference: MAE {mae_diff:+.2f} frames ({direction}), coverage {cov_diff:+.1%}")

    if args.quick:
        print("\nNote: This is a quick single-seed result.")
        print("      Run without --quick for robust multi-seed evaluation.")


if __name__ == '__main__':
    main()
