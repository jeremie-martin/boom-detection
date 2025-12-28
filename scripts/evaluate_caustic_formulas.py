#!/usr/bin/env python3
"""
Systematic evaluation of caustic formula variants.

Evaluates each formula on each pipeline component:
- HGB classifier alone
- CNN classifier alone (via pipeline)
- Quality prediction correlation
- Full pipeline (selective)

Uses memory-safe patterns with auto_release and n_jobs limits.

Usage:
    uv run python scripts/evaluate_caustic_formulas.py
    uv run python scripts/evaluate_caustic_formulas.py --quick  # Single seed
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.evaluation import CachedEvaluator
from boom_detection.frame_models import FrameLevelClassifier
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.logging_config import logger, log_memory_usage


# Formulas to test (coverage is the baseline, legacy includes all 9 features)
FORMULAS = ['coverage', 'dispersion', 'entropy', 'coverage_dispersion', 'legacy']


def evaluate_formula_on_hgb(
    formula: str,
    dataset,
    seeds: list[int],
    cache_dir: str,
) -> dict:
    """Evaluate a caustic formula on HGB alone (loads from disk cache)."""
    config = FeatureConfig(
        max_pendulums=2000,
        include_caustic=True,
        caustic_formula=formula,
    )

    # Load from disk cache (features already extracted)
    cache = FeatureCache(config=config, cache_dir=f'{cache_dir}/{formula}')
    sim_ids = [a.id for a in dataset.annotations]
    cache.load_from_disk(sim_ids, verbose=False)

    evaluator = CachedEvaluator(dataset, cache)
    result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
    )

    del cache
    gc.collect()

    return {
        'mae': result.mean_metrics['mae'],
        'mae_std': result.std_metrics['mae'],
        'within_5': result.mean_metrics.get('within_5', 0),
        'within_10': result.mean_metrics.get('within_10', 0),
    }


def evaluate_formula_on_pipeline(
    formula: str,
    dataset,
    seeds: list[int],
    cache_dir: str,
) -> dict:
    """Evaluate a caustic formula on the full pipeline (loads from disk cache)."""
    config = FeatureConfig(
        max_pendulums=2000,
        include_caustic=True,
        caustic_formula=formula,
    )

    # Load from disk cache (features already extracted)
    cache = FeatureCache(config=config, cache_dir=f'{cache_dir}/{formula}')
    sim_ids = [a.id for a in dataset.annotations]
    cache.load_from_disk(sim_ids, verbose=False)

    evaluator = CachedEvaluator(dataset, cache)
    result = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            accept_threshold=0.60,
            agreement_formula='sqrt',
            agreement_scale=15.0,
        ),
        seeds=seeds,
    )

    del cache
    gc.collect()

    return {
        'selective_mae': result.mean_metrics['selective_mae'],
        'selective_mae_std': result.std_metrics['selective_mae'],
        'coverage': result.mean_metrics['coverage'],
        'within_5_selective': result.mean_metrics.get('within_5_selective', 0),
    }


def evaluate_no_caustic(
    dataset,
    seeds: list[int],
    cache_dir: str,
) -> dict:
    """Evaluate baseline without caustic features (loads from disk cache)."""
    config = FeatureConfig(
        max_pendulums=2000,
        include_caustic=False,
    )

    # Load from disk cache (features already extracted)
    cache = FeatureCache(config=config, cache_dir=f'{cache_dir}/no_caustic')
    sim_ids = [a.id for a in dataset.annotations]
    cache.load_from_disk(sim_ids, verbose=False)

    evaluator = CachedEvaluator(dataset, cache)

    # HGB
    hgb_result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
    )

    # Pipeline
    pipeline_result = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            accept_threshold=0.60,
            agreement_formula='sqrt',
            agreement_scale=15.0,
        ),
        seeds=seeds,
    )

    del cache
    gc.collect()

    return {
        'hgb_mae': hgb_result.mean_metrics['mae'],
        'hgb_mae_std': hgb_result.std_metrics['mae'],
        'pipeline_mae': pipeline_result.mean_metrics['selective_mae'],
        'pipeline_mae_std': pipeline_result.std_metrics['selective_mae'],
        'coverage': pipeline_result.mean_metrics['coverage'],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate caustic formula variants')
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--quick', action='store_true', help='Quick single-seed evaluation')
    parser.add_argument('--cache-dir', type=str, default='.feature_cache', help='Cache directory')
    args = parser.parse_args()

    seeds = [42] if args.quick else [42, 43, 44]

    logger.info("Loading dataset...")
    log_memory_usage("initial")
    dataset = load_dataset(args.data, verbose=False)
    log_memory_usage("after loading")

    print(f"\nEvaluating caustic formulas with {len(seeds)} seeds")
    print("=" * 70)

    # PHASE 1: Extract ALL features first (requires raw data)
    print("\nPhase 1: Extracting features for all formulas...")
    print("-" * 50)

    all_configs = [
        ('no_caustic', FeatureConfig(max_pendulums=2000, include_caustic=False)),
    ]
    for formula in FORMULAS:
        all_configs.append((formula, FeatureConfig(
            max_pendulums=2000,
            include_caustic=True,
            caustic_formula=formula,
        )))

    for name, config in all_configs:
        cache = FeatureCache(config=config, cache_dir=f'{args.cache_dir}/{name}')
        cache.extract_all(dataset, auto_release=False, n_jobs=4)
        del cache
        gc.collect()
        log_memory_usage(f"after {name} extraction")

    # CRITICAL: Release raw data now - we have all features cached to disk
    dataset.release_simulation_data()
    gc.collect()
    log_memory_usage("after releasing raw data")

    # PHASE 2: Evaluate using cached features (no raw data needed)
    print("\nPhase 2: Evaluating formulas...")

    # Baseline
    print("\n1. BASELINE (no caustic features)")
    print("-" * 50)
    baseline = evaluate_no_caustic(dataset, seeds, args.cache_dir)
    print(f"   HGB MAE: {baseline['hgb_mae']:.2f} +/- {baseline['hgb_mae_std']:.2f}")
    print(f"   Pipeline MAE: {baseline['pipeline_mae']:.2f} +/- {baseline['pipeline_mae_std']:.2f}")
    print(f"   Coverage: {baseline['coverage']:.1%}")

    # Evaluate each formula
    hgb_results = {}
    pipeline_results = {}

    print("\n2. HGB CLASSIFIER RESULTS")
    print("-" * 50)
    print(f"{'Formula':<20} {'MAE':<12} {'Within 5':<12} {'Within 10':<12}")
    print("-" * 50)

    for formula in FORMULAS:
        result = evaluate_formula_on_hgb(formula, dataset, seeds, args.cache_dir)
        hgb_results[formula] = result
        mae_str = f"{result['mae']:.2f} +/- {result['mae_std']:.2f}"
        print(f"{formula:<20} {mae_str:<12} {result['within_5']:.1%}         {result['within_10']:.1%}")

    print("\n3. FULL PIPELINE RESULTS")
    print("-" * 50)
    print(f"{'Formula':<20} {'Selective MAE':<15} {'Coverage':<12} {'Within 5':<12}")
    print("-" * 50)

    for formula in FORMULAS:
        result = evaluate_formula_on_pipeline(formula, dataset, seeds, args.cache_dir)
        pipeline_results[formula] = result
        mae_str = f"{result['selective_mae']:.2f} +/- {result['selective_mae_std']:.2f}"
        print(f"{formula:<20} {mae_str:<15} {result['coverage']:.1%}         {result['within_5_selective']:.1%}")

    log_memory_usage("after all evaluations")

    print("\n4. SUMMARY")
    print("=" * 70)

    # Find best formula for each component
    best_hgb = min(hgb_results, key=lambda x: hgb_results[x]['mae'])
    best_pipeline = min(pipeline_results, key=lambda x: pipeline_results[x]['selective_mae'])

    print(f"Best for HGB: {best_hgb} (MAE {hgb_results[best_hgb]['mae']:.2f})")
    print(f"Best for Pipeline: {best_pipeline} (Selective MAE {pipeline_results[best_pipeline]['selective_mae']:.2f})")

    print("\nComparison to baseline:")
    print(f"  HGB improvement with {best_hgb}: {baseline['hgb_mae'] - hgb_results[best_hgb]['mae']:+.2f}")
    print(f"  Pipeline improvement with {best_pipeline}: {baseline['pipeline_mae'] - pipeline_results[best_pipeline]['selective_mae']:+.2f}")

    # Key insight
    if baseline['pipeline_mae'] < pipeline_results[best_pipeline]['selective_mae']:
        print("\nKEY INSIGHT: No caustic features still best for pipeline!")
    else:
        print(f"\nKEY INSIGHT: {best_pipeline} formula improves pipeline!")


if __name__ == '__main__':
    main()
