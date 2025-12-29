#!/usr/bin/env python3
"""
Experiment: Multiple Quality Thresholds for Specialized Models.

Trains HGB at different quality thresholds (0.4, 0.5, 0.6, 0.7)
and tests combiner configurations to find the optimal setup.

Memory-efficient version: runs one configuration at a time with cleanup.

Usage:
    uv run python scripts/experiment_multi_threshold.py data
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.logging_config import log_memory_usage
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.combine import (
    SpecializedModelCombiner,
    QualityGatedCombiner,
)


def format_result(result, name: str) -> str:
    """Format a result for printing."""
    mae = result.mean_metrics.get('selective_mae', float('nan'))
    mae_std = result.std_metrics.get('selective_mae', 0)
    cov = result.mean_metrics.get('coverage', 0)
    return f"{name:<55} MAE {mae:5.2f} ± {mae_std:4.2f}  at {cov:5.1%} coverage"


def main():
    parser = argparse.ArgumentParser(description='Multi-threshold specialized model experiment')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.4, 0.5, 0.6, 0.7],
                        help='Quality thresholds for training (default: 0.4 0.5 0.6 0.7)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Save results to directory')
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT: MULTI-THRESHOLD SPECIALIZED HGB MODELS")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print(f"Training thresholds: {args.thresholds}")
    print("\nUsing PRODUCTION_CONFIG for reproducibility")

    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    # Build feature cache using PRODUCTION_CONFIG
    print("Extracting features...")
    cache = FeatureCache(PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            print(f"  Using cached features ({loaded} sims)")
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    # Create evaluator
    evaluator = CachedEvaluator(dataset, cache)
    print(f"Evaluating on {len(dataset.annotations)} simulations")

    results = {}

    # =========================================================================
    # Part 1: Baseline (no specialized models)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: BASELINES (no specialized models)")
    print("=" * 70)

    # QualityGatedCombiner baselines
    for accept_thresh in [0.5, 0.6, 0.7]:
        result = evaluator.cross_validate_selective(
            lambda t=accept_thresh: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=QualityGatedCombiner(threshold=t),
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"baseline_quality_{accept_thresh}"
        results[key] = result
        print(format_result(result, f"QualityGated(accept={accept_thresh}) [no specialized]"))
        gc.collect()

    log_memory_usage("after baselines")

    # =========================================================================
    # Part 2: Test each specialized model threshold ONE AT A TIME
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SPECIALIZED HGB MODELS (one at a time)")
    print("=" * 70)

    for train_thresh in args.thresholds:
        spec_key = f"hgb_{train_thresh}"
        # Create config for just this one specialized model
        specialized_configs = {spec_key: ('hgb', train_thresh)}

        print(f"\n--- Training threshold: {train_thresh} ---")
        print(f"  Specialized model trained on quality >= {train_thresh}")

        for accept_thresh in [0.5, 0.6, 0.7]:
            result = evaluator.cross_validate_selective(
                lambda k=spec_key, t=accept_thresh, sc=specialized_configs: BoomDetectionPipeline(
                    frame_models=('cnn', 'hgb'),
                    combiner=SpecializedModelCombiner(
                        threshold=t,
                        specialized_model=k,
                    ),
                    specialized_configs=sc,
                ),
                k=5,
                seeds=args.seeds,
                verbose=False,
            )
            key = f"{spec_key}_accept_{accept_thresh}"
            results[key] = result
            print(format_result(result, f"Specialized({spec_key}) accept={accept_thresh}"))

            # Aggressive cleanup after each run
            gc.collect()

        log_memory_usage(f"after train_thresh={train_thresh}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: BEST CONFIGURATIONS BY COVERAGE")
    print("=" * 70)

    # Group by coverage range
    coverage_groups = {
        '5-15%': (0.05, 0.15),
        '15-25%': (0.15, 0.25),
        '25-40%': (0.25, 0.40),
        '40%+': (0.40, 1.0),
    }

    for group_name, (min_cov, max_cov) in coverage_groups.items():
        group_results = [
            (k, v) for k, v in results.items()
            if min_cov <= v.mean_metrics.get('coverage', 0) < max_cov
        ]
        if group_results:
            print(f"\n{group_name} coverage:")
            group_results.sort(key=lambda x: x[1].mean_metrics.get('selective_mae', 999))
            for name, result in group_results[:5]:
                print(format_result(result, f"  {name}"))

    # Save results
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        def serialize_result(result):
            return {
                'mean_metrics': result.mean_metrics,
                'std_metrics': result.std_metrics,
            }

        with open(args.output / 'results.json', 'w') as f:
            json.dump({k: serialize_result(v) for k, v in results.items()}, f, indent=2)

        # Save config
        config = {
            'thresholds': args.thresholds,
            'seeds': args.seeds,
        }
        with open(args.output / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
