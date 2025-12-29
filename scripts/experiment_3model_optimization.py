#!/usr/bin/env python3
"""
3-Model Pipeline Optimization Experiments.

This script investigates optimization opportunities for the 3-model (CNN+HGB+LSTM) pipeline
that was unlocked by using `std` instead of `range` for disagreement metric.

Experiments:
1. Primary model selection - which model should provide the final prediction?
2. Weight optimization - what agreement/quality weights work best for 3-model?
3. Scale fine-tuning - what's the optimal disagreement scale?
4. Threshold sweep - what's the optimal acceptance threshold?

Usage:
    uv run python scripts/experiment_3model_optimization.py data
    uv run python scripts/experiment_3model_optimization.py data --output runs/3model_optimization
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.logging_config import log_memory_usage
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.combine import ThresholdCombiner


def format_result(result, name: str) -> str:
    """Format a result for printing."""
    mae = result.mean_metrics.get('selective_mae', float('nan'))
    mae_std = result.std_metrics.get('selective_mae', 0)
    cov = result.mean_metrics.get('coverage', 0)
    return f"{name:<50} MAE {mae:5.2f} ± {mae_std:4.2f}  at {cov:5.1%} coverage"


def run_primary_model_experiment(experiment, results: dict):
    """Experiment 1: Which model should provide the final prediction?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: PRIMARY MODEL SELECTION")
    print("=" * 70)
    print("Testing which model should provide the final prediction when accepted.")
    print("Hypothesis: LSTM is best individual model (MAE 18.3), might be better as primary.\n")

    # Test each primary model with the best known config (std/scale=5/threshold=0.60)
    for primary in ['cnn', 'lstm', 'hgb']:
        combiner = ThresholdCombiner(
            primary_model=primary,
            disagreement_metric='std',
            disagreement_scale=5.0,
            threshold=0.60,
        )
        result = experiment.evaluate(combiner)
        key = f"primary_{primary}"
        results[key] = result
        print(format_result(result, f"  primary_model='{primary}'"))

    # Also test median aggregation (no primary model preference)
    combiner = ThresholdCombiner(
        primary_model=None,  # Will use median
        disagreement_metric='std',
        disagreement_scale=5.0,
        threshold=0.60,
    )
    result = experiment.evaluate(combiner)
    results['primary_median'] = result
    print(format_result(result, "  primary_model=None (median)"))


def run_weight_optimization_experiment(experiment, results: dict):
    """Experiment 2: What agreement/quality weights work best for 3-model?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: WEIGHT OPTIMIZATION FOR 3-MODEL")
    print("=" * 70)
    print("Testing different agreement/quality weight combinations.")
    print("2-model found 0.2/0.8 beats 0.4/0.6. Does this hold for 3-model?\n")

    # Fine-grained weight sweep
    for agree_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        quality_w = 1.0 - agree_w
        combiner = ThresholdCombiner(
            disagreement_metric='std',
            disagreement_scale=5.0,
            threshold=0.60,
            agreement_weight=agree_w,
            quality_weight=quality_w,
        )
        result = experiment.evaluate(combiner)
        key = f"weights_{agree_w:.1f}_{quality_w:.1f}"
        results[key] = result
        print(format_result(result, f"  agree={agree_w:.1f} quality={quality_w:.1f}"))


def run_scale_finetuning_experiment(experiment, results: dict):
    """Experiment 3: What's the optimal disagreement scale for 3-model + std?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SCALE FINE-TUNING FOR 3-MODEL")
    print("=" * 70)
    print("Testing fine-grained scale values around the best known (5.0).")
    print("Lower scale = more selective. Maybe scale=4 or scale=3 is better?\n")

    # Fine-grained scale sweep
    for scale in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
        combiner = ThresholdCombiner(
            disagreement_metric='std',
            disagreement_scale=scale,
            threshold=0.60,
        )
        result = experiment.evaluate(combiner)
        key = f"scale_{scale}"
        results[key] = result
        print(format_result(result, f"  scale={scale}"))


def run_threshold_sweep_experiment(experiment, results: dict):
    """Experiment 4: What's the optimal acceptance threshold?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: THRESHOLD SWEEP")
    print("=" * 70)
    print("Testing different acceptance thresholds for 3-model + std.")
    print("Higher threshold = more selective.\n")

    # Threshold sweep with best scale
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        combiner = ThresholdCombiner(
            disagreement_metric='std',
            disagreement_scale=5.0,
            threshold=threshold,
        )
        result = experiment.evaluate(combiner)
        key = f"threshold_{threshold}"
        results[key] = result
        print(format_result(result, f"  threshold={threshold}"))


def analyze_results(results: dict):
    """Analyze and summarize all results."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    # Group results by experiment
    primary_results = {k: v for k, v in results.items() if k.startswith('primary_')}
    weight_results = {k: v for k, v in results.items() if k.startswith('weights_')}
    scale_results = {k: v for k, v in results.items() if k.startswith('scale_')}
    threshold_results = {k: v for k, v in results.items() if k.startswith('threshold_')}

    def find_best(result_dict):
        """Find best result (lowest MAE)."""
        best_key = None
        best_mae = float('inf')
        for key, result in result_dict.items():
            mae = result.mean_metrics.get('selective_mae', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_key = key
        return best_key, best_mae

    # Experiment 1: Primary model
    print("\n1. PRIMARY MODEL SELECTION:")
    if primary_results:
        best_key, best_mae = find_best(primary_results)
        print(f"   Best: {best_key} (MAE {best_mae:.2f})")
        for key, result in sorted(primary_results.items()):
            mae = result.mean_metrics.get('selective_mae', float('nan'))
            cov = result.mean_metrics.get('coverage', 0)
            marker = " <-- BEST" if key == best_key else ""
            print(f"   - {key}: MAE {mae:.2f} at {cov:.1%}{marker}")

    # Experiment 2: Weight optimization
    print("\n2. WEIGHT OPTIMIZATION:")
    if weight_results:
        best_key, best_mae = find_best(weight_results)
        print(f"   Best: {best_key} (MAE {best_mae:.2f})")
        for key, result in sorted(weight_results.items()):
            mae = result.mean_metrics.get('selective_mae', float('nan'))
            cov = result.mean_metrics.get('coverage', 0)
            marker = " <-- BEST" if key == best_key else ""
            print(f"   - {key}: MAE {mae:.2f} at {cov:.1%}{marker}")

    # Experiment 3: Scale tuning
    print("\n3. SCALE FINE-TUNING:")
    if scale_results:
        best_key, best_mae = find_best(scale_results)
        print(f"   Best: {best_key} (MAE {best_mae:.2f})")
        # Sort by scale value
        sorted_keys = sorted(scale_results.keys(), key=lambda k: float(k.split('_')[1]))
        for key in sorted_keys:
            result = scale_results[key]
            mae = result.mean_metrics.get('selective_mae', float('nan'))
            cov = result.mean_metrics.get('coverage', 0)
            marker = " <-- BEST" if key == best_key else ""
            print(f"   - {key}: MAE {mae:.2f} at {cov:.1%}{marker}")

    # Experiment 4: Threshold sweep
    print("\n4. THRESHOLD SWEEP:")
    if threshold_results:
        best_key, best_mae = find_best(threshold_results)
        print(f"   Best: {best_key} (MAE {best_mae:.2f})")
        for key, result in sorted(threshold_results.items()):
            mae = result.mean_metrics.get('selective_mae', float('nan'))
            cov = result.mean_metrics.get('coverage', 0)
            marker = " <-- BEST" if key == best_key else ""
            print(f"   - {key}: MAE {mae:.2f} at {cov:.1%}{marker}")

    # Overall best
    print("\n" + "-" * 70)
    print("OVERALL BEST CONFIGURATION:")
    best_overall_key, best_overall_mae = find_best(results)
    best_result = results[best_overall_key]
    print(f"   {best_overall_key}")
    print(f"   MAE: {best_overall_mae:.2f} ± {best_result.std_metrics.get('selective_mae', 0):.2f}")
    print(f"   Coverage: {best_result.mean_metrics.get('coverage', 0):.1%}")


def main():
    parser = argparse.ArgumentParser(description='3-Model optimization experiments')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Save results to directory')
    args = parser.parse_args()

    print("=" * 70)
    print("3-MODEL PIPELINE OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print("Models: CNN + HGB + LSTM")
    print("Base config: disagreement_metric='std'")

    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    # Build feature cache
    print("Extracting features...")
    config = FeatureConfig(max_pendulums=2000, include_caustic=False)
    cache = FeatureCache(config, cache_dir='.feature_cache/no_caustic')

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

    # Create 3-model combiner experiment
    print("\nTraining 3-model pipeline (CNN + HGB + LSTM)...")
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=ThresholdCombiner(disagreement_metric='std'),
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )

    results = {}

    # Run all experiments
    run_primary_model_experiment(experiment, results)
    run_weight_optimization_experiment(experiment, results)
    run_scale_finetuning_experiment(experiment, results)
    run_threshold_sweep_experiment(experiment, results)

    # Analyze results
    analyze_results(results)

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
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
