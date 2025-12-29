#!/usr/bin/env python3
"""
Experiment 6: Combined Best Configuration.

Tests combinations of best findings:
- Best primary model (HGB for quality-gated)
- Best weights (0.4/0.6)
- Best thresholds
- Compare ThresholdCombiner vs QualityGatedModelCombiner

Usage:
    uv run python scripts/experiment_combined_best.py data
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.logging_config import log_memory_usage
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.combine import (
    ThresholdCombiner,
    QualityGatedModelCombiner,
)


def format_result(result, name: str) -> str:
    """Format a result for printing."""
    mae = result.mean_metrics.get('selective_mae', float('nan'))
    mae_std = result.std_metrics.get('selective_mae', 0)
    cov = result.mean_metrics.get('coverage', 0)
    return f"{name:<55} MAE {mae:5.2f} ± {mae_std:4.2f}  at {cov:5.1%} coverage"


def main():
    parser = argparse.ArgumentParser(description='Combined best configuration experiment')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 6: COMBINED BEST CONFIGURATION")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print("Combining findings from previous experiments to find optimal config.")

    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    # Build feature cache
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

    # Create 2-model experiment
    print("\nTraining 2-model pipeline...")
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),
            combiner=ThresholdCombiner(),
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON AT SIMILAR COVERAGE")
    print("=" * 70)

    # ~12-14% coverage comparison
    print("\n--- LOW COVERAGE (~12-14%) ---")

    result = experiment.evaluate(ThresholdCombiner(
        primary_model='cnn',
        disagreement_scale=5.0,
        threshold=0.60,
    ))
    print(format_result(result, "ThresholdCombiner(cnn, scale=5, thresh=0.60)"))

    result = experiment.evaluate(ThresholdCombiner(
        primary_model='hgb',
        disagreement_scale=5.0,
        threshold=0.60,
    ))
    print(format_result(result, "ThresholdCombiner(hgb, scale=5, thresh=0.60)"))

    result = experiment.evaluate(QualityGatedModelCombiner(threshold=0.70, primary_model='cnn'))
    print(format_result(result, "QualityGatedModel(thresh=0.70, primary=cnn)"))

    result = experiment.evaluate(QualityGatedModelCombiner(threshold=0.70, primary_model='hgb'))
    print(format_result(result, "QualityGatedModel(thresh=0.70, primary=hgb)"))

    result = experiment.evaluate(QualityGatedModelCombiner(threshold=0.70, primary_model='median'))
    print(format_result(result, "QualityGatedModel(thresh=0.70, primary=median)"))

    # ~30% coverage comparison
    print("\n--- MODERATE COVERAGE (~30%) ---")

    result = experiment.evaluate(ThresholdCombiner(
        primary_model='cnn',
        disagreement_scale=15.0,
        threshold=0.60,
    ))
    print(format_result(result, "ThresholdCombiner(cnn, scale=15, thresh=0.60)"))

    result = experiment.evaluate(ThresholdCombiner(
        primary_model='hgb',
        disagreement_scale=15.0,
        threshold=0.60,
    ))
    print(format_result(result, "ThresholdCombiner(hgb, scale=15, thresh=0.60)"))

    result = experiment.evaluate(QualityGatedModelCombiner(threshold=0.65, primary_model='cnn'))
    print(format_result(result, "QualityGatedModel(thresh=0.65, primary=cnn)"))

    result = experiment.evaluate(QualityGatedModelCombiner(threshold=0.65, primary_model='hgb'))
    print(format_result(result, "QualityGatedModel(thresh=0.65, primary=hgb)"))

    # Fine-tune the best config
    print("\n" + "=" * 70)
    print("FINE-TUNING BEST CONFIG")
    print("=" * 70)

    print("\n--- QualityGatedModel with HGB primary ---")
    for thresh in [0.68, 0.69, 0.70, 0.71, 0.72]:
        result = experiment.evaluate(QualityGatedModelCombiner(threshold=thresh, primary_model='hgb'))
        print(format_result(result, f"QualityGatedModel(thresh={thresh}, primary=hgb)"))

    print("\n--- ThresholdCombiner with HGB primary ---")
    for scale in [4, 5, 6, 7]:
        result = experiment.evaluate(ThresholdCombiner(
            primary_model='hgb',
            disagreement_scale=scale,
            threshold=0.60,
        ))
        print(format_result(result, f"ThresholdCombiner(hgb, scale={scale}, thresh=0.60)"))

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    print("\n--- Simplest (quality-only, no agreement needed) ---")
    result = experiment.evaluate(QualityGatedModelCombiner(threshold=0.70, primary_model='hgb'))
    print(format_result(result, "QualityGatedModel(thresh=0.70, primary=hgb)"))

    print("\n--- Default (balanced, uses agreement) ---")
    result = experiment.evaluate(ThresholdCombiner(
        primary_model='cnn',
        disagreement_scale=15.0,
        threshold=0.60,
    ))
    print(format_result(result, "ThresholdCombiner(cnn, scale=15, thresh=0.60)"))

    print("\n--- Most selective (uses agreement) ---")
    result = experiment.evaluate(ThresholdCombiner(
        primary_model='cnn',
        disagreement_scale=5.0,
        threshold=0.60,
    ))
    print(format_result(result, "ThresholdCombiner(cnn, scale=5, thresh=0.60)"))

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
