#!/usr/bin/env python3
"""
Experiment: Specialized Model for High-Quality Predictions.

Tests whether training a model only on high-quality data improves final predictions.
Uses the unified CachedEvaluator framework for reproducibility.

Hypothesis:
- Models trained on all data learn noisy patterns from ambiguous booms
- A specialized model trained only on well-defined booms (quality >= 0.70)
  should make more accurate predictions

Key features:
- Frame models (CNN, HGB) compute agreement and quality as usual
- Specialized model (CNN, HGB, or LSTM) is used ONLY for final prediction
- The specialized model is NOT included in agreement calculation

Usage:
    uv run python scripts/experiment_specialized_model.py data
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
    ThresholdCombiner,
    QualityGatedCombiner,
)


def format_result(result, name: str) -> str:
    """Format a result for printing."""
    mae = result.mean_metrics.get('selective_mae', float('nan'))
    mae_std = result.std_metrics.get('selective_mae', 0)
    cov = result.mean_metrics.get('coverage', 0)
    return f"{name:<55} MAE {mae:5.2f} ± {mae_std:4.2f}  at {cov:5.1%} coverage"


def main():
    parser = argparse.ArgumentParser(description='Specialized model experiment')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Save results to directory')
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT: SPECIALIZED HIGH-QUALITY MODEL")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print("Hypothesis: Training only on high-quality data produces better predictions")
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
    # Part 1: Baseline without specialized model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: BASELINES (no specialized model)")
    print("=" * 70)

    # Best known config: QualityGatedModel with HGB
    print("\nEvaluating baseline configurations...")

    # Quality-gated baseline (uses median by default)
    for thresh in [0.65, 0.70, 0.75]:
        result = evaluator.cross_validate_selective(
            lambda t=thresh: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=QualityGatedCombiner(threshold=t),
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"baseline_quality_gated_{thresh}"
        results[key] = result
        print(format_result(result, f"QualityGated({thresh}) [baseline, no specialized]"))

    # ThresholdCombiner baseline
    for scale in [5, 15]:
        result = evaluator.cross_validate_selective(
            lambda s=scale: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=ThresholdCombiner(
                    agreement_transform='sqrt',
                    disagreement_scale=s,
                    threshold=0.60,
                    primary_model='cnn',
                ),
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"baseline_threshold_scale{scale}"
        results[key] = result
        print(format_result(result, f"ThresholdCombiner(sqrt, scale={scale})"))

    # =========================================================================
    # Part 2: Specialized HGB model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SPECIALIZED HGB (trained on quality >= 0.70 only)")
    print("=" * 70)

    # Quality-gated with specialized HGB
    print("\nWith QualityGatedCombiner:")
    for thresh in [0.65, 0.70, 0.75]:
        result = evaluator.cross_validate_selective(
            lambda t=thresh: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=QualityGatedCombiner(threshold=t),
                specialized_model='hgb',
                specialized_quality_threshold=0.70,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"specialized_hgb_quality_{thresh}"
        results[key] = result
        print(format_result(result, f"QualityGated({thresh}) + specialized HGB"))

    # ThresholdCombiner with specialized HGB
    print("\nWith ThresholdCombiner:")
    for scale in [5, 10, 15]:
        result = evaluator.cross_validate_selective(
            lambda s=scale: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=ThresholdCombiner(
                    agreement_transform='sqrt',
                    disagreement_scale=s,
                    threshold=0.60,
                    primary_model='cnn',  # CNN for agreement, specialized HGB for prediction
                ),
                specialized_model='hgb',
                specialized_quality_threshold=0.70,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"specialized_hgb_threshold_scale{scale}"
        results[key] = result
        print(format_result(result, f"ThresholdCombiner(scale={scale}) + specialized HGB"))

    # =========================================================================
    # Part 3: Specialized CNN model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: SPECIALIZED CNN (trained on quality >= 0.70 only)")
    print("=" * 70)

    # Quality-gated with specialized CNN
    for thresh in [0.65, 0.70, 0.75]:
        result = evaluator.cross_validate_selective(
            lambda t=thresh: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=QualityGatedCombiner(threshold=t),
                specialized_model='cnn',
                specialized_quality_threshold=0.70,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"specialized_cnn_quality_{thresh}"
        results[key] = result
        print(format_result(result, f"QualityGated({thresh}) + specialized CNN"))

    # =========================================================================
    # Part 4: Specialized LSTM model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: SPECIALIZED LSTM (trained on quality >= 0.70 only)")
    print("=" * 70)

    # Quality-gated with specialized LSTM
    for thresh in [0.65, 0.70, 0.75]:
        result = evaluator.cross_validate_selective(
            lambda t=thresh: BoomDetectionPipeline(
                frame_models=('cnn', 'hgb'),
                combiner=QualityGatedCombiner(threshold=t),
                specialized_model='lstm',
                specialized_quality_threshold=0.70,
            ),
            k=5,
            seeds=args.seeds,
            verbose=False,
        )
        key = f"specialized_lstm_quality_{thresh}"
        results[key] = result
        print(format_result(result, f"QualityGated({thresh}) + specialized LSTM"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: BEST CONFIGURATIONS")
    print("=" * 70)

    # Find best results at similar coverage
    print("\nAt ~12% coverage (quality threshold 0.70):")
    coverage_12 = [(k, v) for k, v in results.items()
                   if 0.10 <= v.mean_metrics.get('coverage', 0) <= 0.15]
    coverage_12.sort(key=lambda x: x[1].mean_metrics.get('selective_mae', 999))
    for name, result in coverage_12[:5]:
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

        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
