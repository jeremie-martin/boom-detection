#!/usr/bin/env python3
"""
Comprehensive combiner ablation experiments.

This script investigates three key questions:

1. **std vs range**: Does std-based disagreement work better for 3-model?
   - The 3-model paradox: range inflates with more models, but correlation is better
   - Test if std resolves this by not inflating with outliers

2. **Simpler baselines**: How much does the combined score help?
   - Quality-only gating
   - Agreement-only gating
   - Majority vote

3. **Weight optimization**: Are 0.4/0.6 weights optimal?
   - Sweep different weight combinations
   - Test quality-only (1.0/0.0) and agreement-only (0.0/1.0)

Usage:
    uv run python scripts/experiment_combiner_ablations.py data
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.logging_config import log_memory_usage
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.combine import (
    ThresholdCombiner,
    QualityGatedCombiner,
    AgreementGatedCombiner,
    MajorityVoteCombiner,
    MedianCombiner,
)


def format_result(result, name: str) -> str:
    """Format a result for printing."""
    mae = result.mean_metrics.get('selective_mae', float('nan'))
    mae_std = result.std_metrics.get('selective_mae', 0)
    cov = result.mean_metrics.get('coverage', 0)
    return f"{name:<45} MAE {mae:5.2f} ± {mae_std:4.2f}  at {cov:5.1%} coverage"


def run_2model_experiments(evaluator: CachedEvaluator, seeds: list[int]):
    """Run experiments on 2-model pipeline (CNN + HGB)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: 2-MODEL PIPELINE (CNN + HGB)")
    print("=" * 70)

    # Create experiment
    print("\nTraining 2-model pipeline...")
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),
            combiner=ThresholdCombiner(),  # Will be replaced during evaluation
        ),
        k=5,
        seeds=seeds,
        verbose=True,
    )

    results = {}

    # -------------------------------------------
    # 1. std vs range disagreement metric
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("1A. std vs range disagreement metric")
    print("-" * 70)

    # For 2-model, range = 2 * std (when there are 2 values)
    # So we need to scale appropriately
    for metric, scales in [
        ('range', [5, 10, 15, 20]),
        ('std', [2.5, 5, 7.5, 10]),  # ~0.5x the range scales
    ]:
        for scale in scales:
            combiner = ThresholdCombiner(
                disagreement_metric=metric,
                disagreement_scale=scale,
                threshold=0.60,
            )
            result = experiment.evaluate(combiner)
            key = f"{metric}_scale{scale}"
            results[key] = result
            print(format_result(result, f"  {metric} scale={scale}"))

    # -------------------------------------------
    # 2. Simpler baselines
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("1B. Simpler baselines vs ThresholdCombiner")
    print("-" * 70)

    # Baseline: No rejection (always accept)
    baseline = MedianCombiner()
    result = experiment.evaluate(baseline)
    results['median_always'] = result
    print(format_result(result, "  MedianCombiner (no rejection)"))

    # Quality-only gating
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        combiner = QualityGatedCombiner(threshold=thresh)
        result = experiment.evaluate(combiner)
        key = f"quality_only_{thresh}"
        results[key] = result
        print(format_result(result, f"  QualityGated(thresh={thresh})"))

    # Agreement-only gating
    for max_dis in [5, 10, 15, 20]:
        combiner = AgreementGatedCombiner(max_disagreement=max_dis)
        result = experiment.evaluate(combiner)
        key = f"agreement_only_{max_dis}"
        results[key] = result
        print(format_result(result, f"  AgreementGated(max={max_dis})"))

    # Majority vote
    for tol in [3, 5, 10]:
        combiner = MajorityVoteCombiner(tolerance=tol, min_agreement_ratio=0.5)
        result = experiment.evaluate(combiner)
        key = f"majority_tol{tol}"
        results[key] = result
        print(format_result(result, f"  MajorityVote(tol={tol})"))

    # ThresholdCombiner (default) for comparison
    combiner = ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        threshold=0.60,
    )
    result = experiment.evaluate(combiner)
    results['threshold_default'] = result
    print(format_result(result, "  ThresholdCombiner(default)"))

    # -------------------------------------------
    # 3. Weight optimization
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("1C. Agreement/Quality weight optimization")
    print("-" * 70)

    # Test different weight combinations
    for agree_w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        quality_w = 1.0 - agree_w
        combiner = ThresholdCombiner(
            agreement_transform='sqrt',
            disagreement_scale=15.0,
            threshold=0.60,
            agreement_weight=agree_w,
            quality_weight=quality_w,
        )
        result = experiment.evaluate(combiner)
        key = f"weights_{agree_w:.1f}_{quality_w:.1f}"
        results[key] = result
        print(format_result(result, f"  agree={agree_w:.1f} quality={quality_w:.1f}"))

    return results


def run_3model_experiments(evaluator: CachedEvaluator, seeds: list[int]):
    """Run experiments on 3-model pipeline (CNN + HGB + LSTM)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 3-MODEL PIPELINE (CNN + HGB + LSTM)")
    print("=" * 70)

    # Create experiment
    print("\nTraining 3-model pipeline...")
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb', 'lstm'),
            combiner=ThresholdCombiner(),
        ),
        k=5,
        seeds=seeds,
        verbose=True,
    )

    results = {}

    # -------------------------------------------
    # std vs range for 3-model
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("2A. std vs range for 3-model (testing the paradox)")
    print("-" * 70)

    # For 3-model, range inflates more than std
    # Test if std gives better selectivity
    for metric, scales in [
        ('range', [10, 15, 20, 25, 30]),  # Range is larger for 3 models
        ('std', [5, 7.5, 10, 12.5, 15]),  # std is roughly 0.5x range
    ]:
        for scale in scales:
            combiner = ThresholdCombiner(
                disagreement_metric=metric,
                disagreement_scale=scale,
                threshold=0.60,
            )
            result = experiment.evaluate(combiner)
            key = f"3model_{metric}_scale{scale}"
            results[key] = result
            print(format_result(result, f"  {metric} scale={scale}"))

    # -------------------------------------------
    # Best configs from 2-model, applied to 3-model
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("2B. Applying 2-model best configs to 3-model")
    print("-" * 70)

    # sqrt/scale=5/threshold=0.60 was best for 2-model (MAE 3.4 at 14%)
    combiner = ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=5.0,
        threshold=0.60,
        disagreement_metric='range',
    )
    result = experiment.evaluate(combiner)
    results['3model_2model_best_range'] = result
    print(format_result(result, "  sqrt/range/scale=5 (2-model best)"))

    # Same but with std metric
    combiner = ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=2.5,  # scale down for std
        threshold=0.60,
        disagreement_metric='std',
    )
    result = experiment.evaluate(combiner)
    results['3model_2model_best_std'] = result
    print(format_result(result, "  sqrt/std/scale=2.5 (2-model best, std)"))

    return results


def analyze_results(results_2model: dict, results_3model: dict):
    """Analyze and summarize results."""
    print("\n" + "=" * 70)
    print("ANALYSIS AND INSIGHTS")
    print("=" * 70)

    # -------------------------------------------
    # Insight 1: std vs range
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("INSIGHT 1: std vs range disagreement")
    print("-" * 70)

    # Find best range and std configs for 2-model at similar coverage
    range_results = [(k, v) for k, v in results_2model.items() if k.startswith('range_')]
    std_results = [(k, v) for k, v in results_2model.items() if k.startswith('std_')]

    print("\n2-model comparison:")
    for name, result in range_results:
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        print(f"  {name}: MAE {mae:.2f} at {cov:.1%}")

    for name, result in std_results:
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        print(f"  {name}: MAE {mae:.2f} at {cov:.1%}")

    if results_3model:
        print("\n3-model comparison (testing the paradox):")
        range_3m = [(k, v) for k, v in results_3model.items() if '3model_range' in k]
        std_3m = [(k, v) for k, v in results_3model.items() if '3model_std' in k]

        for name, result in range_3m:
            mae = result.mean_metrics.get('selective_mae', float('nan'))
            cov = result.mean_metrics.get('coverage', 0)
            print(f"  {name}: MAE {mae:.2f} at {cov:.1%}")

        for name, result in std_3m:
            mae = result.mean_metrics.get('selective_mae', float('nan'))
            cov = result.mean_metrics.get('coverage', 0)
            print(f"  {name}: MAE {mae:.2f} at {cov:.1%}")

    # -------------------------------------------
    # Insight 2: Simple baselines
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("INSIGHT 2: How much does combined score help?")
    print("-" * 70)

    baseline_keys = ['median_always', 'threshold_default']
    quality_keys = [k for k in results_2model.keys() if k.startswith('quality_only')]
    agreement_keys = [k for k in results_2model.keys() if k.startswith('agreement_only')]
    majority_keys = [k for k in results_2model.keys() if k.startswith('majority')]

    print("\nNo rejection baseline:")
    if 'median_always' in results_2model:
        result = results_2model['median_always']
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        print(f"  MedianCombiner: MAE {mae:.2f} at 100% coverage")

    print("\nQuality-only gating:")
    for key in sorted(quality_keys):
        result = results_2model[key]
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        print(f"  {key}: MAE {mae:.2f} at {cov:.1%}")

    print("\nAgreement-only gating:")
    for key in sorted(agreement_keys):
        result = results_2model[key]
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        print(f"  {key}: MAE {mae:.2f} at {cov:.1%}")

    print("\nMajority vote:")
    for key in sorted(majority_keys):
        result = results_2model[key]
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        print(f"  {key}: MAE {mae:.2f} at {cov:.1%}")

    print("\nThresholdCombiner (combined score):")
    if 'threshold_default' in results_2model:
        result = results_2model['threshold_default']
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        print(f"  Default (sqrt/15/0.60): MAE {mae:.2f} at {cov:.1%}")

    # -------------------------------------------
    # Insight 3: Weight optimization
    # -------------------------------------------
    print("\n" + "-" * 70)
    print("INSIGHT 3: Optimal agreement/quality weights")
    print("-" * 70)

    weight_keys = [k for k in results_2model.keys() if k.startswith('weights_')]
    weight_results = []
    for key in sorted(weight_keys):
        result = results_2model[key]
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        cov = result.mean_metrics.get('coverage', 0)
        agree_w = float(key.split('_')[1])
        quality_w = float(key.split('_')[2])
        weight_results.append((agree_w, quality_w, mae, cov))
        print(f"  agree={agree_w:.1f} quality={quality_w:.1f}: MAE {mae:.2f} at {cov:.1%}")

    # Find best at similar coverage
    target_coverage = 0.30
    similar_cov = [(a, q, m, c) for a, q, m, c in weight_results if abs(c - target_coverage) < 0.10]
    if similar_cov:
        best = min(similar_cov, key=lambda x: x[2])
        print(f"\n  Best at ~30% coverage: agree={best[0]:.1f} quality={best[1]:.1f} "
              f"(MAE {best[2]:.2f} at {best[3]:.1%})")


def main():
    parser = argparse.ArgumentParser(description='Combiner ablation experiments')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (default: 42 43 44)')
    parser.add_argument('--skip-3model', action='store_true',
                        help='Skip 3-model experiments (faster)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Save results to directory')
    args = parser.parse_args()

    print("=" * 70)
    print("COMBINER ABLATION EXPERIMENTS")
    print("=" * 70)
    print(f"\nSeeds: {args.seeds}")
    print(f"Skip 3-model: {args.skip_3model}")

    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    # Build feature cache (using PRODUCTION_CONFIG for consistency)
    print("Extracting features...")
    cache = FeatureCache(PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')
    cache.extract_all(dataset, verbose=False)
    dataset.release_simulation_data()
    gc.collect()
    log_memory_usage("after feature extraction")

    # Create evaluator
    evaluator = CachedEvaluator(dataset, cache)
    print(f"Evaluating on {len(dataset.annotations)} simulations")

    # Run 2-model experiments
    results_2model = run_2model_experiments(evaluator, args.seeds)

    # Run 3-model experiments
    results_3model = {}
    if not args.skip_3model:
        results_3model = run_3model_experiments(evaluator, args.seeds)

    # Analyze results
    analyze_results(results_2model, results_3model)

    # Save results
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        def serialize_result(result):
            return {
                'mean_metrics': result.mean_metrics,
                'std_metrics': result.std_metrics,
            }

        with open(args.output / '2model_results.json', 'w') as f:
            json.dump({k: serialize_result(v) for k, v in results_2model.items()}, f, indent=2)

        if results_3model:
            with open(args.output / '3model_results.json', 'w') as f:
                json.dump({k: serialize_result(v) for k, v in results_3model.items()}, f, indent=2)

        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
