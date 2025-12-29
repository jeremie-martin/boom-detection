#!/usr/bin/env python3
"""
Validate that FormulaExperiment produces IDENTICAL results to cross_validate_selective.

This script proves the key guarantee: formula iteration produces the same results
as retraining the full pipeline.

Expected output: MAE 3.38 ± 0.83 at 13.7% coverage (sqrt/scale=5)
"""
from __future__ import annotations

import gc
import sys

import numpy as np


def main():
    # Import heavy modules inside main
    from boom_detection.deploy_pipeline import BoomDetectionPipeline
    from boom_detection.features import FeatureCache, FeatureConfig
    from boom_detection.evaluation import CachedEvaluator
    from boom_detection.loader import load_dataset
    from boom_detection.logging_config import log_memory_usage

    print("=" * 70)
    print("VALIDATING FormulaExperiment IDENTITY GUARANTEE")
    print("=" * 70)

    # Load dataset and cache
    # IMPORTANT: Use the same feature config as evaluate_3model_pipeline.py!
    print("\nLoading dataset...")
    dataset = load_dataset('data', verbose=False)
    log_memory_usage("after loading")

    config = FeatureConfig(max_pendulums=2000, include_caustic=False)
    cache = FeatureCache(config=config, cache_dir='.feature_cache/no_caustic')

    # Try to load from disk first, extract if needed
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
    n_sims = len(evaluator.sim_ids)
    print(f"\nEvaluating on {n_sims} simulations")

    # Define test configuration
    test_config = {
        'acceptance_formula': 'sqrt',
        'acceptance_params': {'scale': 5.0, 'threshold': 0.60},
    }
    seeds = [42, 43, 44]

    print(f"\nTest configuration:")
    print(f"  acceptance_formula: {test_config['acceptance_formula']}")
    print(f"  acceptance_params: {test_config['acceptance_params']}")
    print(f"  seeds: {seeds}")

    # =========================================================================
    # METHOD 1: FormulaExperiment (fast formula iteration)
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: FormulaExperiment")
    print("=" * 70)

    # Create formula experiment (trains models once)
    print("\nTraining models and caching predictions...")
    experiment = evaluator.create_formula_experiment(
        predictor_fn=lambda: BoomDetectionPipeline(
            acceptance_formula='sqrt',  # Initial value, will be overridden
            acceptance_params={'scale': 15.0, 'threshold': 0.60},
        ),
        k=5,
        seeds=seeds,
        verbose=True,
    )

    # Evaluate with test config
    print("\nEvaluating with FormulaExperiment...")
    formula_result = experiment.evaluate(
        acceptance_formula=test_config['acceptance_formula'],
        acceptance_params=test_config['acceptance_params'],
        verbose=True,
    )

    formula_mae = formula_result.mean_metrics.get('selective_mae', float('nan'))
    formula_mae_std = formula_result.std_metrics.get('selective_mae', 0)
    formula_cov = formula_result.mean_metrics.get('coverage', 0)
    formula_cov_std = formula_result.std_metrics.get('coverage', 0)

    print(f"\nFormulaExperiment Result:")
    print(f"  Selective MAE: {formula_mae:.2f} ± {formula_mae_std:.2f}")
    print(f"  Coverage: {formula_cov:.1%} ± {formula_cov_std:.1%}")

    # =========================================================================
    # METHOD 2: cross_validate_selective (full pipeline per config)
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: cross_validate_selective (ground truth)")
    print("=" * 70)

    print("\nRunning full cross_validate_selective with same config...")
    direct_result = evaluator.cross_validate_selective(
        predictor_fn=lambda: BoomDetectionPipeline(
            acceptance_formula=test_config['acceptance_formula'],
            acceptance_params=test_config['acceptance_params'],
        ),
        k=5,
        seeds=seeds,
        verbose=True,
    )

    direct_mae = direct_result.mean_metrics.get('selective_mae', float('nan'))
    direct_mae_std = direct_result.std_metrics.get('selective_mae', 0)
    direct_cov = direct_result.mean_metrics.get('coverage', 0)
    direct_cov_std = direct_result.std_metrics.get('coverage', 0)

    print(f"\ncross_validate_selective Result:")
    print(f"  Selective MAE: {direct_mae:.2f} ± {direct_mae_std:.2f}")
    print(f"  Coverage: {direct_cov:.1%} ± {direct_cov_std:.1%}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Check identity
    mae_diff = abs(formula_mae - direct_mae)
    cov_diff = abs(formula_cov - direct_cov)

    print(f"\n{'Metric':<20} {'FormulaExperiment':<20} {'cross_validate':<20} {'Difference':<15}")
    print("-" * 75)
    print(f"{'Selective MAE':<20} {formula_mae:.4f}             {direct_mae:.4f}             {mae_diff:.6f}")
    print(f"{'Coverage':<20} {formula_cov:.4f}             {direct_cov:.4f}             {cov_diff:.6f}")

    # Verify identity
    TOLERANCE = 1e-10
    mae_identical = mae_diff < TOLERANCE
    cov_identical = cov_diff < TOLERANCE

    print(f"\n{'='*70}")
    if mae_identical and cov_identical:
        print("✓ IDENTITY VERIFIED: Results are IDENTICAL (within tolerance)")
        print(f"  Tolerance: {TOLERANCE}")
    else:
        print("✗ IDENTITY FAILED: Results differ!")
        print(f"  MAE difference: {mae_diff}")
        print(f"  Coverage difference: {cov_diff}")
        sys.exit(1)
    print("=" * 70)

    # =========================================================================
    # BONUS: Demonstrate fast formula iteration
    # =========================================================================
    print("\n" + "=" * 70)
    print("BONUS: Fast Formula Sweep (no retraining!)")
    print("=" * 70)

    sweep_results = experiment.sweep(
        formulas=['sqrt'],
        param_grid={
            'scale': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
            'threshold': [0.60],
        },
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Formula':<8} {'Scale':<8} {'Threshold':<10} {'MAE ± std':<15} {'Coverage':<12}")
    print("-" * 60)

    for (formula, scale, threshold), result in sorted(sweep_results.items()):
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        cov = result.mean_metrics.get('coverage', 0)
        print(f"{formula:<8} {scale:<8.0f} {threshold:<10.2f} {mae:.2f} ± {mae_std:.2f}       {cov:.1%}")

    print("\n✓ All operations completed successfully!")


if __name__ == '__main__':
    main()
