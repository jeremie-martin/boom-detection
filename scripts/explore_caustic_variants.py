#!/usr/bin/env python3
"""
Explore different caustic feature formulations to find optimal configuration.

Tests:
1. Coverage-only vs coverage×gini vs both
2. Different bin counts
3. Individual features vs combinations
4. Impact on HGB frame classifier (fast, reliable)

Uses 3-seed evaluation for reasonable reliability without excessive runtime.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.evaluation import CachedEvaluator
from boom_detection.frame_models import FrameLevelClassifier


@dataclass
class VariantResult:
    """Result from testing a caustic variant."""
    name: str
    n_features: int
    mae_mean: float
    mae_std: float
    within_5: float
    within_10: float


def test_hgb_variant(
    dataset,
    cache: FeatureCache,
    seeds: list[int],
    name: str,
) -> VariantResult:
    """Test HGB classifier with a feature cache variant."""
    evaluator = CachedEvaluator(dataset, cache)

    result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
        verbose=False,
    )

    n_features = cache[dataset.annotations[0].id].shape[1]

    return VariantResult(
        name=name,
        n_features=n_features,
        mae_mean=result.mean_metrics['mae'],
        mae_std=result.std_metrics['mae'],
        within_5=result.mean_metrics['within_5'],
        within_10=result.mean_metrics['within_10'],
    )


def analyze_feature_correlation_with_boom(
    dataset,
    cache: FeatureCache,
    feature_names: list[str],
):
    """Analyze how each feature correlates with boom frame."""
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = [a.boom_frame for a in dataset.annotations]

    print("\nFeature correlations with boom frame:")
    print("-" * 50)

    correlations = []
    for i, name in enumerate(feature_names):
        # Get feature values at boom frame for each simulation
        boom_values = []
        for sid, boom in zip(sim_ids, boom_frames):
            features = cache[sid]
            # Get feature value at boom frame
            if boom < len(features):
                boom_values.append(features[boom, i])
            else:
                boom_values.append(features[-1, i])

        # Also try mean feature value correlation
        mean_values = [cache[sid][:, i].mean() for sid in sim_ids]

        corr_at_boom, _ = spearmanr(boom_values, boom_frames)
        corr_mean, _ = spearmanr(mean_values, boom_frames)

        correlations.append((name, corr_at_boom, corr_mean))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"{'Feature':<30} {'Corr@Boom':>12} {'Corr(mean)':>12}")
    print("-" * 55)
    for name, corr_boom, corr_mean in correlations[:20]:
        print(f"{name:<30} {corr_boom:>+12.3f} {corr_mean:>+12.3f}")

    return correlations


def main():
    parser = argparse.ArgumentParser(description="Explore caustic variants")
    parser.add_argument("data_path", nargs="?", default="data")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))

    print("=" * 70)
    print("CAUSTIC FEATURE VARIANT EXPLORATION")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    print(f"Loaded {len(dataset)} simulations")
    print()

    results = []

    # =========================================================================
    # Variant 1: No caustic features at all (minimal baseline)
    # =========================================================================
    print("=" * 70)
    print("1. NO CAUSTIC FEATURES (baseline)")
    print("=" * 70)

    config = FeatureConfig(include_caustic=False)
    cache = FeatureCache(config=config)
    cache.extract_all(dataset, verbose=False)

    result = test_hgb_variant(dataset, cache, seeds, "No Caustic")
    results.append(result)
    print(f"  Features: {result.n_features}, MAE: {result.mae_mean:.2f} ± {result.mae_std:.2f}")

    # Get feature names for this config
    baseline_features = config.feature_names()

    # =========================================================================
    # Variant 2: Current full caustic (9 features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. FULL CAUSTIC (9 features: legacy + coverage)")
    print("=" * 70)

    config = FeatureConfig(include_caustic=True)
    cache = FeatureCache(config=config)
    cache.extract_all(dataset, verbose=False)

    result = test_hgb_variant(dataset, cache, seeds, "Full Caustic (9)")
    results.append(result)
    print(f"  Features: {result.n_features}, MAE: {result.mae_mean:.2f} ± {result.mae_std:.2f}")

    # Analyze feature correlations
    all_features = config.feature_names()
    caustic_indices = [i for i, name in enumerate(all_features) if 'caust' in name or 'coverage' in name]
    caustic_names = [all_features[i] for i in caustic_indices]
    print(f"\nCaustic features: {caustic_names}")

    # =========================================================================
    # Variant 3: Test coverage-only vs legacy-only
    # =========================================================================
    # To test this properly, we need to modify the feature extraction...
    # For now, let's analyze feature importance from the trained model

    print("\n" + "=" * 70)
    print("3. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Train a single model and get feature importance
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Prepare data
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = [a.boom_frame for a in dataset.annotations]

    X_all = []
    y_all = []
    for sid, boom in zip(sim_ids, boom_frames):
        features = cache[sid]
        n_frames = len(features)
        labels = (np.arange(n_frames) >= boom).astype(int)
        X_all.append(features)
        y_all.append(labels)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_

    print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':>12}")
    print("-" * 50)

    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:15], 1):
        name = all_features[idx]
        imp = importances[idx]
        marker = "**" if 'caust' in name or 'coverage' in name else ""
        print(f"{rank:<6} {name:<30} {imp:>12.4f} {marker}")

    # Show caustic-only importances
    print("\nCaustic feature importances:")
    print("-" * 50)
    for idx in caustic_indices:
        name = all_features[idx]
        imp = importances[idx]
        rank = list(sorted_idx).index(idx) + 1
        print(f"  #{rank:<3} {name:<30} {imp:.4f}")

    # =========================================================================
    # Test removing specific caustic features
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. ABLATION: WHICH CAUSTIC FEATURES HELP?")
    print("=" * 70)

    # Sort caustic features by importance
    caustic_importances = [(all_features[i], importances[i]) for i in caustic_indices]
    caustic_importances.sort(key=lambda x: x[1], reverse=True)

    print("\nCaustic features ranked by importance:")
    for name, imp in caustic_importances:
        print(f"  {name:<30} {imp:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Variant':<25} {'Features':>10} {'MAE':>15} {'Within 5':>12} {'Within 10':>12}")
    print("-" * 75)

    for r in results:
        print(f"{r.name:<25} {r.n_features:>10} {r.mae_mean:>7.2f}±{r.mae_std:<5.2f} "
              f"{r.within_5*100:>10.1f}% {r.within_10*100:>10.1f}%")

    # Calculate improvement
    baseline_mae = results[0].mae_mean
    for r in results[1:]:
        change = (baseline_mae - r.mae_mean) / baseline_mae * 100
        print(f"\n{r.name} vs No Caustic: {change:+.1f}% {'improvement' if change > 0 else 'degradation'}")


if __name__ == "__main__":
    main()
