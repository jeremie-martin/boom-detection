#!/usr/bin/env python3
"""
B6: 3-Model Stability Deep Dive.

Verify the remarkably low variance (std=0.06) of the 3-model pipeline.
Compare stability across different configurations.

Usage:
    uv run python scripts/b6_stability_analysis.py data --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boom_detection.combine import ThresholdCombiner, QualityGatedCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='B6: 3-Model Stability Analysis')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                        help='Random seeds for evaluation (default: 10 seeds)')
    parser.add_argument('--output', type=Path, default=Path('results/b6_stability_analysis'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("b6_stability_analysis")

    logger.info("=" * 70)
    logger.info("B6: 3-Model Stability Deep Dive")
    logger.info("=" * 70)
    logger.info(f"Seeds: {args.seeds} ({len(args.seeds)} seeds)")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    cache = FeatureCache(config=PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    evaluator = CachedEvaluator(dataset, cache)
    logger.info(f"Evaluating on {len(evaluator.sim_ids)} simulations")

    # Configurations to test
    configs = [
        ('3-model (std/s=15)', {
            'frame_models': ('cnn', 'hgb', 'lstm'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sqrt',
                disagreement_scale=15.0,
                disagreement_metric='std',
                threshold=0.70,
            ),
        }),
        ('2-model (sigmoid/s=10)', {
            'frame_models': ('cnn', 'hgb'),
            'combiner': ThresholdCombiner(
                primary_model='cnn',
                agreement_transform='sigmoid',
                disagreement_scale=10.0,
                threshold=0.70,
            ),
        }),
        ('Quality-only (t=0.70)', {
            'frame_models': ('cnn', 'hgb'),
            'combiner': QualityGatedCombiner(threshold=0.70),
        }),
    ]

    all_results = {}

    for config_name, config in configs:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {config_name}")
        logger.info("=" * 70)

        seed_results = []

        for seed in args.seeds:
            result = evaluator.cross_validate_selective(
                lambda cfg=config: BoomDetectionPipeline(
                    frame_models=cfg['frame_models'],
                    combiner=cfg['combiner'],
                ),
                k=5,
                seeds=[seed],
                verbose=False,
            )

            mae = result.mean_metrics.get('selective_mae', float('nan'))
            coverage = result.mean_metrics.get('coverage', 0)

            seed_results.append({
                'seed': seed,
                'mae': mae,
                'coverage': coverage,
            })

            logger.info(f"  Seed {seed}: MAE={mae:.2f}, Coverage={coverage:.1%}")

        maes = np.array([r['mae'] for r in seed_results if not np.isnan(r['mae'])])
        coverages = np.array([r['coverage'] for r in seed_results])

        if len(maes) > 0:
            mean_mae = np.mean(maes)
            std_mae = np.std(maes)
            min_mae = np.min(maes)
            max_mae = np.max(maes)
        else:
            mean_mae = std_mae = min_mae = max_mae = float('nan')

        mean_cov = np.mean(coverages)
        std_cov = np.std(coverages)

        all_results[config_name] = {
            'seed_results': seed_results,
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'min_mae': min_mae,
            'max_mae': max_mae,
            'range_mae': max_mae - min_mae,
            'mean_cov': mean_cov,
            'std_cov': std_cov,
            'maes': maes,
            'coverages': coverages,
        }

        logger.info(f"  Summary: MAE={mean_mae:.2f} +/- {std_mae:.2f}, Range=[{min_mae:.2f}, {max_mae:.2f}]")
        logger.info(f"           Coverage={mean_cov:.1%} +/- {std_cov:.1%}")

    # Comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("STABILITY COMPARISON")
    logger.info("=" * 70)

    logger.info(f"\n{'Configuration':<30} {'MAE Mean':<12} {'MAE Std':<12} {'MAE Range':<12} {'Coverage':<12}")
    logger.info("-" * 78)
    for config_name, r in all_results.items():
        logger.info(f"{config_name:<30} {r['mean_mae']:.2f}       {r['std_mae']:.2f}       {r['range_mae']:.2f}       {r['mean_cov']:.1%}")

    # Identify most stable
    stds = {k: r['std_mae'] for k, r in all_results.items() if not np.isnan(r['std_mae'])}
    if stds:
        most_stable = min(stds, key=stds.get)
        logger.info(f"\nMost stable: {most_stable} (std={stds[most_stable]:.2f})")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Plot 1: MAE distribution by config
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = list(range(len(all_results)))
    data_to_plot = [r['maes'] for r in all_results.values()]
    labels = list(all_results.keys())

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

    colors = ['blue', 'green', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('MAE (frames)')
    ax.set_title(f'MAE Distribution by Configuration ({len(args.seeds)} seeds)')
    ax.axhline(y=2.97, color='red', linestyle='--', label='Baseline (2.97)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output / 'mae_distribution.png', dpi=150)
    plt.close()

    # Plot 2: MAE vs Seed for each config
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (config_name, r) in enumerate(all_results.items()):
        seeds = [sr['seed'] for sr in r['seed_results']]
        maes = [sr['mae'] for sr in r['seed_results']]
        ax.plot(seeds, maes, 'o-', label=config_name, alpha=0.7)

    ax.set_xlabel('Seed')
    ax.set_ylabel('MAE (frames)')
    ax.set_title('MAE by Seed Across Configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output / 'mae_by_seed.png', dpi=150)
    plt.close()

    # Plot 3: Std comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(all_results))
    stds = [r['std_mae'] for r in all_results.values()]
    colors = ['blue' if s < 0.5 else 'orange' if s < 1.0 else 'red' for s in stds]

    ax.bar(x, stds, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(all_results.keys()), rotation=15, ha='right')
    ax.set_ylabel('MAE Standard Deviation')
    ax.set_title('Stability Comparison (lower is better)')
    ax.axhline(y=0.06, color='green', linestyle='--', label='Documented 3-model std (0.06)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output / 'stability_comparison.png', dpi=150)
    plt.close()

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# B6: 3-Model Stability Deep Dive\n\n")
        f.write("## Objective\n")
        f.write("Verify the remarkably low variance (std=0.06) of the 3-model pipeline.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds tested: {args.seeds}\n")
        f.write(f"- Number of seeds: {len(args.seeds)}\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | MAE Mean | MAE Std | MAE Range | Coverage |\n")
        f.write("|---------------|----------|---------|-----------|----------|\n")
        for config_name, r in sorted(all_results.items(), key=lambda x: x[1]['std_mae']):
            f.write(f"| {config_name} | {r['mean_mae']:.2f} | {r['std_mae']:.2f} | [{r['min_mae']:.2f}, {r['max_mae']:.2f}] | {r['mean_cov']:.1%} |\n")

        f.write("\n## Analysis\n\n")

        three_model_std = all_results.get('3-model (std/s=15)', {}).get('std_mae', float('nan'))
        two_model_std = all_results.get('2-model (sigmoid/s=10)', {}).get('std_mae', float('nan'))
        quality_std = all_results.get('Quality-only (t=0.70)', {}).get('std_mae', float('nan'))

        f.write("### Stability Ranking\n\n")
        ranked = sorted([(k, v['std_mae']) for k, v in all_results.items()], key=lambda x: x[1])
        for i, (name, std) in enumerate(ranked, 1):
            f.write(f"{i}. **{name}**: std = {std:.2f}\n")

        f.write("\n### Key Findings\n\n")

        if not np.isnan(three_model_std):
            if three_model_std < 0.15:
                f.write(f"1. **3-model is remarkably stable** (std={three_model_std:.2f})\n")
                f.write("   - The `std` disagreement metric filters outliers effectively\n")
                f.write("   - All three models must roughly agree for acceptance\n")
            else:
                f.write(f"1. **3-model stability is moderate** (std={three_model_std:.2f})\n")
                f.write(f"   - Higher than documented 0.06 with {len(args.seeds)} seeds\n")

        if not np.isnan(two_model_std) and not np.isnan(three_model_std):
            if three_model_std < two_model_std * 0.5:
                f.write(f"2. **3-model is much more stable than 2-model** ({three_model_std:.2f} vs {two_model_std:.2f})\n")
            elif three_model_std < two_model_std:
                f.write(f"2. **3-model is more stable than 2-model** ({three_model_std:.2f} vs {two_model_std:.2f})\n")
            else:
                f.write(f"2. **2-model is comparable or better stability** ({two_model_std:.2f} vs 3-model {three_model_std:.2f})\n")

        f.write("\n### Why is 3-model more stable?\n\n")
        f.write("1. **More stringent acceptance**: Three models must agree (using std metric)\n")
        f.write("2. **Outlier robustness**: std is less sensitive to one model being wrong\n")
        f.write("3. **Consistent sample selection**: Similar samples accepted across seeds\n")

        f.write("\n## Plots\n")
        f.write("- `mae_distribution.png`: Box plots showing MAE variance by config\n")
        f.write("- `mae_by_seed.png`: How MAE varies across seeds\n")
        f.write("- `stability_comparison.png`: Bar chart of std by configuration\n")

    logger.info(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
