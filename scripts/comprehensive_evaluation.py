#!/usr/bin/env python3
"""
Comprehensive Evaluation Script.

This script performs a full evaluation of the boom detection system:
1. Dataset characterization
2. Baseline evaluation (3 pipeline configs)
3. Caustic formula comparison (6 formulas)
4. HGB feature subset evaluation
5. Statistical analysis and plots

Uses memory-safe two-phase pattern:
- Phase 1: Extract all features (requires raw data)
- Phase 2: Evaluate from disk cache (releases raw data)

Usage:
    uv run python scripts/comprehensive_evaluation.py
    uv run python scripts/comprehensive_evaluation.py --quick  # 3 seeds instead of 5
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.evaluation import CachedEvaluator
from boom_detection.frame_models import FrameLevelClassifier
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.sequence_models import CNNClassifier, SequenceTrainer
from boom_detection.logging_config import logger, log_memory_usage


# =============================================================================
# Configuration
# =============================================================================

CAUSTIC_FORMULAS = ['no_caustic', 'legacy', 'coverage', 'dispersion', 'entropy', 'coverage_dispersion']

PIPELINE_CONFIGS = [
    ('default', {'agreement_formula': 'linear', 'agreement_scale': 10.0}),
    ('balanced', {'agreement_formula': 'sqrt', 'agreement_scale': 15.0}),
    ('selective', {'agreement_formula': 'sqrt', 'agreement_scale': 5.0}),
]


@dataclass
class EvalResults:
    """Container for all evaluation results."""
    dataset_stats: dict
    baseline: dict
    caustic_hgb: dict
    caustic_pipeline: dict
    hgb_subsets: dict
    statistics: dict


# =============================================================================
# Phase 1: Feature Extraction
# =============================================================================

def extract_all_features(dataset, cache_dir: str, verbose: bool = True) -> None:
    """Extract features for all caustic configurations."""
    print("\n" + "=" * 70)
    print("PHASE 1: FEATURE EXTRACTION")
    print("=" * 70)

    configs = []
    for formula in CAUSTIC_FORMULAS:
        if formula == 'no_caustic':
            configs.append((formula, FeatureConfig(max_pendulums=2000, include_caustic=False)))
        else:
            configs.append((formula, FeatureConfig(
                max_pendulums=2000,
                include_caustic=True,
                caustic_formula=formula,
            )))

    sim_ids = [a.id for a in dataset.annotations]

    for name, config in configs:
        cache_path = f'{cache_dir}/{name}'
        cache = FeatureCache(config=config, cache_dir=cache_path)

        # Try to load from disk first
        try:
            loaded = cache.load_from_disk(sim_ids, verbose=False)
            if loaded == len(sim_ids):
                if verbose:
                    print(f"  {name}: Using cached features ({loaded} sims)")
            else:
                if verbose:
                    print(f"  {name}: Extracting features ({loaded}/{len(sim_ids)} cached)...")
                cache.extract_all(dataset, auto_release=False, n_jobs=4, verbose=False)
        except (ValueError, FileNotFoundError):
            if verbose:
                print(f"  {name}: Extracting features...")
            cache.extract_all(dataset, auto_release=False, n_jobs=4, verbose=False)

        del cache
        gc.collect()

    if verbose:
        log_memory_usage("after feature extraction")


# =============================================================================
# Phase 2: Evaluation Functions
# =============================================================================

def characterize_dataset(dataset) -> dict:
    """Compute dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET CHARACTERIZATION")
    print("=" * 70)

    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    stats_dict = {
        'n_simulations': len(dataset),
        'boom_frame': {
            'min': int(boom_frames.min()),
            'max': int(boom_frames.max()),
            'mean': float(boom_frames.mean()),
            'median': float(np.median(boom_frames)),
            'std': float(boom_frames.std()),
        },
        'quality': {
            'min': float(qualities.min()),
            'max': float(qualities.max()),
            'mean': float(qualities.mean()),
            'std': float(qualities.std()),
        },
    }

    print(f"\nSimulations: {stats_dict['n_simulations']}")
    print(f"\nBoom Frame Distribution:")
    print(f"  Range: {stats_dict['boom_frame']['min']} - {stats_dict['boom_frame']['max']}")
    print(f"  Mean: {stats_dict['boom_frame']['mean']:.1f} ± {stats_dict['boom_frame']['std']:.1f}")
    print(f"  Median: {stats_dict['boom_frame']['median']:.1f}")
    print(f"\nQuality Distribution:")
    print(f"  Range: {stats_dict['quality']['min']:.2f} - {stats_dict['quality']['max']:.2f}")
    print(f"  Mean: {stats_dict['quality']['mean']:.2f} ± {stats_dict['quality']['std']:.2f}")

    return stats_dict


def evaluate_baseline(dataset, cache_dir: str, seeds: list[int]) -> dict:
    """Evaluate baseline pipeline with different configurations."""
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    print(f"Seeds: {seeds}")

    # Load no_caustic cache
    config = FeatureConfig(max_pendulums=2000, include_caustic=False)
    cache = FeatureCache(config=config, cache_dir=f'{cache_dir}/no_caustic')
    sim_ids = [a.id for a in dataset.annotations]
    cache.load_from_disk(sim_ids, verbose=False)

    evaluator = CachedEvaluator(dataset, cache)
    results = {}

    # Evaluate each pipeline config
    print(f"\n{'Config':<12} {'Selective MAE':<18} {'Coverage':<12} {'Within 5':<12} {'Within 10':<12}")
    print("-" * 70)

    for name, params in PIPELINE_CONFIGS:
        result = evaluator.cross_validate_selective(
            lambda p=params: BoomDetectionPipeline(
                accept_threshold=0.60,
                agreement_formula=p['agreement_formula'],
                agreement_scale=p['agreement_scale'],
            ),
            seeds=seeds,
        )

        results[name] = {
            'selective_mae': result.mean_metrics['selective_mae'],
            'selective_mae_std': result.std_metrics['selective_mae'],
            'coverage': result.mean_metrics['coverage'],
            'within_5': result.mean_metrics.get('within_5_selective', 0),
            'within_10': result.mean_metrics.get('within_10_selective', 0),
            'aurc': result.mean_metrics.get('aurc', 0),
        }

        mae_str = f"{results[name]['selective_mae']:.2f} ± {results[name]['selective_mae_std']:.2f}"
        print(f"{name:<12} {mae_str:<18} {results[name]['coverage']:.1%}        "
              f"{results[name]['within_5']:.1%}         {results[name]['within_10']:.1%}")

    # Also evaluate component models
    print("\n\nComponent-level baselines:")
    print("-" * 50)

    # HGB alone
    hgb_result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
    )
    results['hgb_alone'] = {
        'mae': hgb_result.mean_metrics['mae'],
        'mae_std': hgb_result.std_metrics['mae'],
        'within_5': hgb_result.mean_metrics.get('within_5', 0),
        'within_10': hgb_result.mean_metrics.get('within_10', 0),
    }
    print(f"HGB alone:    MAE {results['hgb_alone']['mae']:.2f} ± {results['hgb_alone']['mae_std']:.2f}, "
          f"Within 5: {results['hgb_alone']['within_5']:.1%}")

    del cache
    gc.collect()

    return results


def evaluate_caustic_formulas_hgb(dataset, cache_dir: str, seeds: list[int]) -> dict:
    """Evaluate each caustic formula on HGB classifier alone."""
    print("\n" + "=" * 70)
    print("CAUSTIC FORMULA COMPARISON: HGB CLASSIFIER")
    print("=" * 70)

    results = {}
    print(f"\n{'Formula':<22} {'MAE':<18} {'Within 5':<12} {'Within 10':<12}")
    print("-" * 70)

    for formula in CAUSTIC_FORMULAS:
        if formula == 'no_caustic':
            config = FeatureConfig(max_pendulums=2000, include_caustic=False)
        else:
            config = FeatureConfig(max_pendulums=2000, include_caustic=True, caustic_formula=formula)

        cache = FeatureCache(config=config, cache_dir=f'{cache_dir}/{formula}')
        sim_ids = [a.id for a in dataset.annotations]
        cache.load_from_disk(sim_ids, verbose=False)

        evaluator = CachedEvaluator(dataset, cache)
        result = evaluator.cross_validate(
            lambda: FrameLevelClassifier(model='hist_gbm'),
            seeds=seeds,
        )

        results[formula] = {
            'mae': result.mean_metrics['mae'],
            'mae_std': result.std_metrics['mae'],
            'within_5': result.mean_metrics.get('within_5', 0),
            'within_10': result.mean_metrics.get('within_10', 0),
            'n_features': cache[sim_ids[0]].shape[1],
        }

        mae_str = f"{results[formula]['mae']:.2f} ± {results[formula]['mae_std']:.2f}"
        print(f"{formula:<22} {mae_str:<18} {results[formula]['within_5']:.1%}         "
              f"{results[formula]['within_10']:.1%}")

        del cache
        gc.collect()

    return results


def evaluate_caustic_formulas_pipeline(dataset, cache_dir: str, seeds: list[int]) -> dict:
    """Evaluate each caustic formula on full pipeline (balanced config)."""
    print("\n" + "=" * 70)
    print("CAUSTIC FORMULA COMPARISON: FULL PIPELINE")
    print("=" * 70)

    results = {}
    print(f"\n{'Formula':<22} {'Selective MAE':<18} {'Coverage':<12} {'Within 5':<12}")
    print("-" * 70)

    for formula in CAUSTIC_FORMULAS:
        if formula == 'no_caustic':
            config = FeatureConfig(max_pendulums=2000, include_caustic=False)
        else:
            config = FeatureConfig(max_pendulums=2000, include_caustic=True, caustic_formula=formula)

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

        results[formula] = {
            'selective_mae': result.mean_metrics['selective_mae'],
            'selective_mae_std': result.std_metrics['selective_mae'],
            'coverage': result.mean_metrics['coverage'],
            'within_5': result.mean_metrics.get('within_5_selective', 0),
        }

        mae_str = f"{results[formula]['selective_mae']:.2f} ± {results[formula]['selective_mae_std']:.2f}"
        print(f"{formula:<22} {mae_str:<18} {results[formula]['coverage']:.1%}         "
              f"{results[formula]['within_5']:.1%}")

        del cache
        gc.collect()

    return results


def evaluate_hgb_subsets(dataset, cache_dir: str, seeds: list[int]) -> dict:
    """Evaluate HGB with different feature subsets."""
    print("\n" + "=" * 70)
    print("HGB FEATURE SUBSET EVALUATION")
    print("=" * 70)

    # Load baseline cache (no caustic, 183 features)
    config = FeatureConfig(max_pendulums=2000, include_caustic=False)
    cache = FeatureCache(config=config, cache_dir=f'{cache_dir}/no_caustic')
    sim_ids = [a.id for a in dataset.annotations]
    cache.load_from_disk(sim_ids, verbose=False)

    evaluator = CachedEvaluator(dataset, cache)

    # Get number of features from cache
    n_total = cache[sim_ids[0]].shape[1]

    results = {}

    print(f"\n{'Subset':<25} {'Features':<10} {'MAE':<18} {'Within 5':<12}")
    print("-" * 70)

    # All features (baseline)
    result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
    )
    results['all_183'] = {
        'n_features': n_total,
        'mae': result.mean_metrics['mae'],
        'mae_std': result.std_metrics['mae'],
        'within_5': result.mean_metrics.get('within_5', 0),
    }
    mae_str = f"{results['all_183']['mae']:.2f} ± {results['all_183']['mae_std']:.2f}"
    print(f"{'All features':<25} {n_total:<10} {mae_str:<18} {results['all_183']['within_5']:.1%}")

    # Note: HGB feature subset requires the pipeline's hgb_feature_subset parameter
    # For now, we just report the baseline
    print("\n(Note: HGB feature subset analysis requires custom feature selection implementation)")

    del cache
    gc.collect()

    return results


def compute_statistics(caustic_hgb: dict, caustic_pipeline: dict, baseline: dict) -> dict:
    """Compute statistical comparisons."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    stats_dict = {}

    # Compare each caustic formula to no_caustic baseline
    baseline_hgb_mae = caustic_hgb['no_caustic']['mae']
    baseline_pipeline_mae = caustic_pipeline['no_caustic']['selective_mae']

    print("\nHGB: Improvement over no_caustic baseline")
    print("-" * 50)

    for formula in CAUSTIC_FORMULAS:
        if formula == 'no_caustic':
            continue

        improvement = baseline_hgb_mae - caustic_hgb[formula]['mae']
        pct_improvement = (improvement / baseline_hgb_mae) * 100

        # Note: Without raw per-fold errors, we can't do proper paired t-test
        # Using approximate significance based on std overlap
        combined_std = np.sqrt(caustic_hgb['no_caustic']['mae_std']**2 +
                               caustic_hgb[formula]['mae_std']**2)
        z_score = abs(improvement) / combined_std if combined_std > 0 else 0
        p_approx = 2 * (1 - stats.norm.cdf(z_score))

        stats_dict[f'hgb_{formula}_vs_baseline'] = {
            'improvement': improvement,
            'pct_improvement': pct_improvement,
            'z_score': z_score,
            'p_approx': p_approx,
        }

        sig = "*" if p_approx < 0.05 else ""
        print(f"  {formula:<20}: {improvement:+.2f} ({pct_improvement:+.1f}%) p≈{p_approx:.3f} {sig}")

    print("\nPipeline: Improvement over no_caustic baseline")
    print("-" * 50)

    for formula in CAUSTIC_FORMULAS:
        if formula == 'no_caustic':
            continue

        improvement = baseline_pipeline_mae - caustic_pipeline[formula]['selective_mae']
        pct_improvement = (improvement / baseline_pipeline_mae) * 100

        combined_std = np.sqrt(caustic_pipeline['no_caustic']['selective_mae_std']**2 +
                               caustic_pipeline[formula]['selective_mae_std']**2)
        z_score = abs(improvement) / combined_std if combined_std > 0 else 0
        p_approx = 2 * (1 - stats.norm.cdf(z_score))

        stats_dict[f'pipeline_{formula}_vs_baseline'] = {
            'improvement': improvement,
            'pct_improvement': pct_improvement,
            'z_score': z_score,
            'p_approx': p_approx,
        }

        sig = "*" if p_approx < 0.05 else ""
        print(f"  {formula:<20}: {improvement:+.2f} ({pct_improvement:+.1f}%) p≈{p_approx:.3f} {sig}")

    return stats_dict


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(results: EvalResults, output_dir: Path) -> None:
    """Generate evaluation plots."""
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Baseline comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(results.baseline.keys())
    configs = [c for c in configs if c != 'hgb_alone']
    maes = [results.baseline[c]['selective_mae'] for c in configs]
    stds = [results.baseline[c]['selective_mae_std'] for c in configs]
    coverages = [results.baseline[c]['coverage'] for c in configs]

    x = np.arange(len(configs))
    bars = ax.bar(x, maes, yerr=stds, capsize=5, color=['#1f77b4', '#2ca02c', '#ff7f0e'])

    # Add coverage labels
    for i, (bar, cov) in enumerate(zip(bars, coverages)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.5,
                f'{cov:.0%}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Selective MAE (frames)')
    ax.set_title('Baseline Pipeline Configurations\n(labels show coverage)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_baseline.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'evaluation_baseline.png'}")

    # Plot 2: Caustic formula comparison (HGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    formulas = list(results.caustic_hgb.keys())
    hgb_maes = [results.caustic_hgb[f]['mae'] for f in formulas]
    hgb_stds = [results.caustic_hgb[f]['mae_std'] for f in formulas]

    x = np.arange(len(formulas))
    colors = ['#1f77b4' if f == 'no_caustic' else '#ff7f0e' for f in formulas]
    bars = ax1.bar(x, hgb_maes, yerr=hgb_stds, capsize=5, color=colors)
    ax1.axhline(y=hgb_maes[formulas.index('no_caustic')], color='red', linestyle='--',
                alpha=0.5, label='no_caustic baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(formulas, rotation=45, ha='right')
    ax1.set_ylabel('MAE (frames)')
    ax1.set_title('HGB Classifier: Caustic Formula Comparison')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()

    # Pipeline comparison
    pipeline_maes = [results.caustic_pipeline[f]['selective_mae'] for f in formulas]
    pipeline_stds = [results.caustic_pipeline[f]['selective_mae_std'] for f in formulas]

    bars = ax2.bar(x, pipeline_maes, yerr=pipeline_stds, capsize=5, color=colors)
    ax2.axhline(y=pipeline_maes[formulas.index('no_caustic')], color='red', linestyle='--',
                alpha=0.5, label='no_caustic baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(formulas, rotation=45, ha='right')
    ax2.set_ylabel('Selective MAE (frames)')
    ax2.set_title('Full Pipeline: Caustic Formula Comparison')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_caustic.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'evaluation_caustic.png'}")


# =============================================================================
# Summary and Recommendations
# =============================================================================

def print_summary(results: EvalResults) -> None:
    """Print summary and recommendations."""
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Best baseline config
    baseline_configs = {k: v for k, v in results.baseline.items() if k != 'hgb_alone'}
    best_baseline = min(baseline_configs, key=lambda x: baseline_configs[x]['selective_mae'])
    print(f"\n1. BEST BASELINE CONFIG: {best_baseline}")
    print(f"   Selective MAE: {results.baseline[best_baseline]['selective_mae']:.2f} ± "
          f"{results.baseline[best_baseline]['selective_mae_std']:.2f}")
    print(f"   Coverage: {results.baseline[best_baseline]['coverage']:.1%}")

    # Best caustic formula for HGB
    best_hgb = min(results.caustic_hgb, key=lambda x: results.caustic_hgb[x]['mae'])
    baseline_hgb = results.caustic_hgb['no_caustic']['mae']
    best_hgb_mae = results.caustic_hgb[best_hgb]['mae']
    print(f"\n2. BEST CAUSTIC FORMULA (HGB): {best_hgb}")
    print(f"   MAE: {best_hgb_mae:.2f} (baseline: {baseline_hgb:.2f})")
    if best_hgb_mae < baseline_hgb:
        print(f"   Improvement: {baseline_hgb - best_hgb_mae:.2f} frames ({(baseline_hgb - best_hgb_mae)/baseline_hgb*100:.1f}%)")
    else:
        print(f"   No improvement over baseline")

    # Best caustic formula for pipeline
    best_pipeline = min(results.caustic_pipeline, key=lambda x: results.caustic_pipeline[x]['selective_mae'])
    baseline_pipeline = results.caustic_pipeline['no_caustic']['selective_mae']
    best_pipeline_mae = results.caustic_pipeline[best_pipeline]['selective_mae']
    print(f"\n3. BEST CAUSTIC FORMULA (PIPELINE): {best_pipeline}")
    print(f"   Selective MAE: {best_pipeline_mae:.2f} (baseline: {baseline_pipeline:.2f})")
    if best_pipeline_mae < baseline_pipeline:
        print(f"   Improvement: {baseline_pipeline - best_pipeline_mae:.2f} frames")
    else:
        print(f"   No improvement over baseline")

    # Key recommendations
    print("\n4. RECOMMENDATIONS:")
    print("-" * 50)

    if best_hgb != 'no_caustic' and results.caustic_hgb[best_hgb]['mae'] < baseline_hgb - 0.5:
        print(f"   - Consider {best_hgb} formula for HGB (improves MAE)")
    else:
        print("   - Keep no_caustic for HGB (caustic features don't help)")

    if best_pipeline == 'no_caustic' or best_pipeline_mae >= baseline_pipeline:
        print("   - Keep no_caustic for pipeline (caustic features don't help)")
    else:
        print(f"   - Consider {best_pipeline} formula for pipeline")

    # Coverage vs accuracy tradeoff
    print(f"\n   - For high accuracy: Use 'selective' config (MAE {results.baseline['selective']['selective_mae']:.1f} @ {results.baseline['selective']['coverage']:.0%})")
    print(f"   - For balanced: Use 'balanced' config (MAE {results.baseline['balanced']['selective_mae']:.1f} @ {results.baseline['balanced']['coverage']:.0%})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive boom detection evaluation')
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--cache-dir', type=str, default='.feature_cache', help='Cache directory')
    parser.add_argument('--output-dir', type=str, default='plots', help='Plot output directory')
    parser.add_argument('--quick', action='store_true', help='Quick 3-seed evaluation')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    seeds = [42, 43, 44] if args.quick else [42, 43, 44, 45, 46]

    print("=" * 70)
    print("COMPREHENSIVE BOOM DETECTION EVALUATION")
    print("=" * 70)
    print(f"Seeds: {seeds} ({len(seeds)}-seed evaluation)")
    print(f"Cache: {args.cache_dir}")

    # Load dataset
    log_memory_usage("initial")
    dataset = load_dataset(args.data, verbose=False)
    log_memory_usage("after loading dataset")

    # Phase 1: Extract features
    extract_all_features(dataset, args.cache_dir)

    # Release raw data
    dataset.release_simulation_data()
    gc.collect()
    log_memory_usage("after releasing raw data")

    # Phase 2: Evaluate
    dataset_stats = characterize_dataset(dataset)
    baseline = evaluate_baseline(dataset, args.cache_dir, seeds)
    caustic_hgb = evaluate_caustic_formulas_hgb(dataset, args.cache_dir, seeds)
    caustic_pipeline = evaluate_caustic_formulas_pipeline(dataset, args.cache_dir, seeds)
    hgb_subsets = evaluate_hgb_subsets(dataset, args.cache_dir, seeds)
    statistics = compute_statistics(caustic_hgb, caustic_pipeline, baseline)

    # Collect results
    results = EvalResults(
        dataset_stats=dataset_stats,
        baseline=baseline,
        caustic_hgb=caustic_hgb,
        caustic_pipeline=caustic_pipeline,
        hgb_subsets=hgb_subsets,
        statistics=statistics,
    )

    # Generate plots
    if not args.skip_plots:
        generate_plots(results, Path(args.output_dir))

    # Print summary
    print_summary(results)

    # Save results to JSON
    output_path = Path(args.output_dir) / 'evaluation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    log_memory_usage("final")


if __name__ == '__main__':
    main()
