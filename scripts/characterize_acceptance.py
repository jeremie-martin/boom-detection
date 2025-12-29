#!/usr/bin/env python3
"""
Comprehensive acceptance formula characterization.

Uses FormulaExperiment.sweep() to efficiently explore the acceptance parameter space
and validate documented results.

Usage:
    # Validate documented results
    uv run python scripts/characterize_acceptance.py data --validate

    # Run comprehensive sweep
    uv run python scripts/characterize_acceptance.py data --sweep

    # Run both
    uv run python scripts/characterize_acceptance.py data --validate --sweep --output runs/characterization
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


# Documented results from README.md (to validate against)
DOCUMENTED_RESULTS = {
    ('sqrt', 5.0, 0.60): {'mae': 3.4, 'mae_std': 0.8, 'coverage': 0.14},
    ('sqrt', 15.0, 0.60): {'mae': 4.9, 'mae_std': 1.3, 'coverage': 0.30},
    ('linear', 10.0, 0.60): {'mae': 7.3, 'mae_std': 1.7, 'coverage': 0.39},
}


def validate_documented_results(experiment) -> bool:
    """Validate that we can reproduce documented results."""
    print("\n" + "=" * 70)
    print("VALIDATING DOCUMENTED RESULTS")
    print("=" * 70)

    all_passed = True

    for (formula, scale, threshold), expected in DOCUMENTED_RESULTS.items():
        result = experiment.evaluate(
            acceptance_formula=formula,
            acceptance_params={'scale': scale, 'threshold': threshold},
        )

        actual_mae = result.mean_metrics.get('selective_mae', float('nan'))
        actual_mae_std = result.std_metrics.get('selective_mae', 0)
        actual_cov = result.mean_metrics.get('coverage', 0)

        # Check if within reasonable tolerance
        mae_close = abs(actual_mae - expected['mae']) < 0.5
        cov_close = abs(actual_cov - expected['coverage']) < 0.05

        status = "PASS" if mae_close and cov_close else "FAIL"
        if not (mae_close and cov_close):
            all_passed = False

        print(f"\n{formula}/scale={scale:.0f}/threshold={threshold:.2f}:")
        print(f"  Expected: MAE {expected['mae']:.1f} ± {expected['mae_std']:.1f} at {expected['coverage']:.0%}")
        print(f"  Actual:   MAE {actual_mae:.2f} ± {actual_mae_std:.2f} at {actual_cov:.1%}")
        print(f"  Status:   [{status}]")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL DOCUMENTED RESULTS VALIDATED")
    else:
        print("SOME RESULTS DIFFER FROM DOCUMENTED VALUES")
    print("=" * 70)

    return all_passed


def run_comprehensive_sweep(experiment, output_dir: Path | None = None) -> dict:
    """Run comprehensive parameter sweep across all formulas."""
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE SWEEP")
    print("=" * 70)

    # Define sweep parameters
    formulas = ['sqrt', 'linear', 'sigmoid', 'quadratic']
    param_grid = {
        'scale': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
        'threshold': [0.50, 0.55, 0.60, 0.65, 0.70],
    }

    results = experiment.sweep(
        formulas=formulas,
        param_grid=param_grid,
        verbose=True,
    )

    # Organize results for analysis
    organized = []
    for (formula, scale, threshold), result in results.items():
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        cov = result.mean_metrics.get('coverage', 0)
        cov_std = result.std_metrics.get('coverage', 0)

        organized.append({
            'formula': formula,
            'scale': scale,
            'threshold': threshold,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': cov,
            'coverage_std': cov_std,
        })

    # Sort by MAE
    organized.sort(key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    # Print best configurations
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY MAE")
    print("=" * 70)
    print(f"{'Rank':<5} {'Formula':<10} {'Scale':<8} {'Thresh':<8} {'MAE ± std':<15} {'Coverage':<12}")
    print("-" * 65)

    for i, cfg in enumerate(organized[:10]):
        print(f"{i+1:<5} {cfg['formula']:<10} {cfg['scale']:<8.0f} {cfg['threshold']:<8.2f} "
              f"{cfg['mae']:.2f} ± {cfg['mae_std']:.2f}     {cfg['coverage']:.1%}")

    # Find best for each coverage range
    print("\n" + "=" * 70)
    print("BEST MAE AT DIFFERENT COVERAGE LEVELS")
    print("=" * 70)

    coverage_ranges = [(0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.50)]
    for low, high in coverage_ranges:
        in_range = [c for c in organized if low <= c['coverage'] < high]
        if in_range:
            best = min(in_range, key=lambda x: x['mae'])
            print(f"Coverage {low:.0%}-{high:.0%}: {best['formula']}/scale={best['scale']:.0f} "
                  f"-> MAE {best['mae']:.2f} at {best['coverage']:.1%}")

    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results as JSON
        with open(output_dir / 'sweep_results.json', 'w') as f:
            json.dump(organized, f, indent=2)

        # Save as CSV for easy analysis
        import csv
        with open(output_dir / 'sweep_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=organized[0].keys())
            writer.writeheader()
            writer.writerows(organized)

        print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Acceptance formula characterization')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--validate', action='store_true', help='Validate documented results')
    parser.add_argument('--sweep', action='store_true', help='Run comprehensive parameter sweep')
    parser.add_argument('--output', type=Path, default=None, help='Output directory for results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation')
    args = parser.parse_args()

    if not args.validate and not args.sweep:
        parser.error("Must specify at least one of --validate or --sweep")

    # Import heavy modules
    from boom_detection.deploy_pipeline import BoomDetectionPipeline
    from boom_detection.features import FeatureCache, FeatureConfig
    from boom_detection.evaluation import CachedEvaluator
    from boom_detection.loader import load_dataset
    from boom_detection.logging_config import log_memory_usage

    print("=" * 70)
    print("ACCEPTANCE FORMULA CHARACTERIZATION")
    print("=" * 70)

    # Load dataset and features
    print("\nLoading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
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

    # Create evaluator and formula experiment
    evaluator = CachedEvaluator(dataset, cache)
    print(f"\nEvaluating on {len(evaluator.sim_ids)} simulations with seeds {args.seeds}")

    print("\nCreating FormulaExperiment (training models once)...")
    start_time = time.time()
    experiment = evaluator.create_formula_experiment(
        predictor_fn=lambda: BoomDetectionPipeline(
            acceptance_formula='sqrt',
            acceptance_params={'scale': 15.0, 'threshold': 0.60},
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    print(f"FormulaExperiment created in {time.time() - start_time:.1f}s")

    # Run validation
    if args.validate:
        validated = validate_documented_results(experiment)
        if not validated:
            print("\nWARNING: Some documented results could not be reproduced exactly.")
            print("This may be due to dataset changes or random variation.")

    # Run sweep
    if args.sweep:
        run_comprehensive_sweep(experiment, args.output)

    print("\n" + "=" * 70)
    print("CHARACTERIZATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
