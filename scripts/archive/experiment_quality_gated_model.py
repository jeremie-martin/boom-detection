#!/usr/bin/env python3
"""
Experiment 5: Quality-Gated with Model Selection.

Tests whether using a specific model (CNN, LSTM, HGB) for prediction
helps when using quality-only gating.

Key question: Is agreement useful for PREDICTION even when not used for ACCEPTANCE?

Usage:
    uv run python scripts/experiment_quality_gated_model.py data
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
    QualityGatedCombiner,
    QualityGatedModelCombiner,
)


def format_result(result, name: str) -> str:
    """Format a result for printing."""
    mae = result.mean_metrics.get('selective_mae', float('nan'))
    mae_std = result.std_metrics.get('selective_mae', 0)
    cov = result.mean_metrics.get('coverage', 0)
    return f"{name:<45} MAE {mae:5.2f} ± {mae_std:4.2f}  at {cov:5.1%} coverage"


def main():
    parser = argparse.ArgumentParser(description='Quality-gated model selection experiment')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 5: QUALITY-GATED WITH MODEL SELECTION")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print("Question: Does using a specific model for prediction help")
    print("          when using quality-only for acceptance?")

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

    # Create evaluator
    evaluator = CachedEvaluator(dataset, cache)
    print(f"Evaluating on {len(dataset.annotations)} simulations")

    # -------------------------------------------
    # Part 1: 2-model pipeline with QualityGatedModelCombiner
    # -------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: 2-MODEL PIPELINE (CNN + HGB)")
    print("=" * 70)

    print("\nTraining 2-model pipeline...")
    experiment_2model = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),
            combiner=ThresholdCombiner(),
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )

    print("\n" + "-" * 70)
    print("Testing quality-gated with different primary models")
    print("-" * 70)

    # Baseline: QualityGatedCombiner (uses median)
    for thresh in [0.65, 0.70, 0.75]:
        result = experiment_2model.evaluate(QualityGatedCombiner(threshold=thresh))
        print(format_result(result, f"  QualityGated(thresh={thresh}) [median]"))

    print()

    # QualityGatedModelCombiner with different primary models
    for primary in ['cnn', 'hgb', 'median']:
        for thresh in [0.65, 0.70, 0.75]:
            combiner = QualityGatedModelCombiner(threshold=thresh, primary_model=primary)
            result = experiment_2model.evaluate(combiner)
            print(format_result(result, f"  QualityGatedModel(thresh={thresh}, primary={primary})"))
        print()

    # -------------------------------------------
    # Part 2: 3-model pipeline with QualityGatedModelCombiner
    # -------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: 3-MODEL PIPELINE (CNN + HGB + LSTM)")
    print("=" * 70)

    print("\nTraining 3-model pipeline...")
    experiment_3model = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=ThresholdCombiner(),
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )

    print("\n" + "-" * 70)
    print("Testing quality-gated with different primary models (3-model)")
    print("-" * 70)

    # QualityGatedModelCombiner with LSTM
    for primary in ['cnn', 'lstm', 'hgb', 'median']:
        for thresh in [0.65, 0.70, 0.75]:
            combiner = QualityGatedModelCombiner(threshold=thresh, primary_model=primary)
            result = experiment_3model.evaluate(combiner)
            print(format_result(result, f"  QualityGatedModel(thresh={thresh}, primary={primary})"))
        print()

    # -------------------------------------------
    # Part 3: Compare best configs
    # -------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: BEST CONFIGURATIONS")
    print("=" * 70)

    print("\n2-model ThresholdCombiner (baseline):")
    result = experiment_2model.evaluate(ThresholdCombiner(
        disagreement_scale=5.0,
        threshold=0.60,
    ))
    print(format_result(result, "  ThresholdCombiner(scale=5, thresh=0.60)"))

    print("\n3-model ThresholdCombiner:")
    result = experiment_3model.evaluate(ThresholdCombiner(
        disagreement_metric='std',
        disagreement_scale=5.0,
        threshold=0.60,
    ))
    print(format_result(result, "  ThresholdCombiner(std, scale=5, thresh=0.60)"))

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
