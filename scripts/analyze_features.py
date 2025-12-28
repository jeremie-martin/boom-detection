#!/usr/bin/env python3
"""
Feature Analysis for Boom Detection

Comprehensive analysis of features including:
1. Correlation with target (boom frame proximity)
2. Feature importance (via model-based methods)
3. Feature correlation matrix
4. Phase-wise analysis (early, pre-boom, boom, post-boom)
5. Performance comparison with/without feature groups

Usage:
    uv run python scripts/analyze_features.py
    uv run python scripts/analyze_features.py --save-plots
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from boom_detection.loader import load_dataset
from boom_detection.features import (
    FeatureCache, FeatureConfig, FeatureExtractor,
    FEATURE_GROUPS, DEFAULT_CONFIG, CAUSTIC_CONFIG,
)
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    name: str
    group: str
    # Correlation with "distance to boom" (negative = good predictor)
    corr_with_boom_distance: float
    corr_pvalue: float
    # Importance from permutation importance
    importance_mean: float
    importance_std: float
    # Basic stats
    mean: float
    std: float
    min: float
    max: float
    # Phase-specific means
    early_mean: float  # First 20% of frames
    preboom_mean: float  # 50 frames before boom
    boom_mean: float  # At boom frame
    postboom_mean: float  # 50 frames after boom


def get_feature_group(feature_name: str) -> str:
    """Determine which group a feature belongs to."""
    # Check base features first
    for group, names in FEATURE_GROUPS.items():
        if feature_name in names:
            return group

    # Check derivative features (d1_*, d2_*)
    if feature_name.startswith('d1_'):
        base = feature_name[3:]
        for group, names in FEATURE_GROUPS.items():
            if base in names:
                return f'd1_{group}'
    if feature_name.startswith('d2_'):
        base = feature_name[3:]
        for group, names in FEATURE_GROUPS.items():
            if base in names:
                return f'd2_{group}'

    return 'unknown'


def compute_feature_stats(
    features: np.ndarray,
    feature_names: list[str],
    boom_frames: np.ndarray,
    sim_ids: list[str],
    cache: FeatureCache,
) -> list[FeatureStats]:
    """
    Compute comprehensive statistics for each feature.

    Args:
        features: Aggregated features (one row per simulation)
        feature_names: Names of features
        boom_frames: Boom frame for each simulation
        sim_ids: Simulation IDs
        cache: Feature cache for per-frame analysis
    """
    n_sims = len(sim_ids)
    n_features = len(feature_names)

    stats_list = []

    # For correlation, we need per-frame features aligned with boom distance
    # Collect all frame features with their boom distance
    all_frame_features = []
    all_boom_distances = []

    for i, sim_id in enumerate(sim_ids):
        sim_features = cache[sim_id]  # (frames, n_features)
        n_frames = sim_features.shape[0]
        boom = boom_frames[i]

        # Boom distance: how far is each frame from the boom
        frame_indices = np.arange(n_frames)
        boom_distances = np.abs(frame_indices - boom)

        all_frame_features.append(sim_features)
        all_boom_distances.append(boom_distances)

    # Concatenate all frames
    all_features = np.vstack(all_frame_features)  # (total_frames, n_features)
    all_distances = np.concatenate(all_boom_distances)  # (total_frames,)

    # Train a quick classifier for feature importance
    # Target: is this frame within 10 frames of boom?
    near_boom = (all_distances <= 10).astype(int)

    # Subsample for speed (max 50k frames)
    if len(near_boom) > 50000:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(near_boom), 50000, replace=False)
        X_imp = all_features[indices]
        y_imp = near_boom[indices]
    else:
        X_imp = all_features
        y_imp = near_boom

    # Train classifier for permutation importance
    print("Training classifier for feature importance...")
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    clf.fit(X_imp, y_imp)

    # Compute permutation importance (faster than full permutation)
    print("Computing permutation importance...")
    perm_imp = permutation_importance(
        clf, X_imp, y_imp,
        n_repeats=5, random_state=42, n_jobs=-1
    )

    # Compute phase-specific statistics
    print("Computing phase-specific statistics...")
    for i, name in enumerate(feature_names):
        group = get_feature_group(name)

        # Correlation with boom distance
        feat_col = all_features[:, i]
        # Use Spearman for robustness to non-linearity
        corr, pvalue = spearmanr(feat_col, all_distances)

        # Phase-specific means (compute per simulation then average)
        early_means = []
        preboom_means = []
        boom_vals = []
        postboom_means = []

        for j, sim_id in enumerate(sim_ids):
            sim_features = cache[sim_id]
            n_frames = sim_features.shape[0]
            boom = int(boom_frames[j])

            # Early: first 20%
            early_end = int(n_frames * 0.2)
            if early_end > 0:
                early_means.append(np.mean(sim_features[:early_end, i]))

            # Pre-boom: 50 frames before
            preboom_start = max(0, boom - 50)
            if preboom_start < boom:
                preboom_means.append(np.mean(sim_features[preboom_start:boom, i]))

            # Boom frame
            if 0 <= boom < n_frames:
                boom_vals.append(sim_features[boom, i])

            # Post-boom: 50 frames after
            postboom_end = min(n_frames, boom + 50)
            if boom < postboom_end:
                postboom_means.append(np.mean(sim_features[boom:postboom_end, i]))

        stats = FeatureStats(
            name=name,
            group=group,
            corr_with_boom_distance=corr,
            corr_pvalue=pvalue,
            importance_mean=perm_imp.importances_mean[i],
            importance_std=perm_imp.importances_std[i],
            mean=np.mean(feat_col),
            std=np.std(feat_col),
            min=np.min(feat_col),
            max=np.max(feat_col),
            early_mean=np.mean(early_means) if early_means else 0.0,
            preboom_mean=np.mean(preboom_means) if preboom_means else 0.0,
            boom_mean=np.mean(boom_vals) if boom_vals else 0.0,
            postboom_mean=np.mean(postboom_means) if postboom_means else 0.0,
        )
        stats_list.append(stats)

    return stats_list


def compute_feature_correlations(
    cache: FeatureCache,
    sim_ids: list[str],
    feature_names: list[str],
) -> np.ndarray:
    """Compute correlation matrix between features."""
    # Collect all frame features
    all_features = []
    for sim_id in sim_ids:
        all_features.append(cache[sim_id])

    all_features = np.vstack(all_features)

    # Subsample for speed
    if len(all_features) > 20000:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_features), 20000, replace=False)
        all_features = all_features[indices]

    # Compute correlation matrix
    n_features = len(feature_names)
    corr_matrix = np.corrcoef(all_features.T)

    return corr_matrix


def print_top_features(stats: list[FeatureStats], n: int = 20):
    """Print top features by importance."""
    print("\n" + "=" * 80)
    print("TOP FEATURES BY PERMUTATION IMPORTANCE")
    print("=" * 80)
    print(f"{'Rank':<5} {'Feature':<40} {'Group':<20} {'Importance':<12} {'Corr w/Dist':<12}")
    print("-" * 89)

    sorted_stats = sorted(stats, key=lambda s: s.importance_mean, reverse=True)

    for i, s in enumerate(sorted_stats[:n]):
        corr_str = f"{s.corr_with_boom_distance:+.3f}"
        print(f"{i+1:<5} {s.name:<40} {s.group:<20} {s.importance_mean:.4f}      {corr_str}")


def print_feature_groups_summary(stats: list[FeatureStats]):
    """Print summary by feature group."""
    print("\n" + "=" * 80)
    print("FEATURE GROUP SUMMARY (mean importance)")
    print("=" * 80)

    # Aggregate by group
    group_importances: dict[str, list[float]] = {}
    for s in stats:
        if s.group not in group_importances:
            group_importances[s.group] = []
        group_importances[s.group].append(s.importance_mean)

    # Sort by mean importance
    group_means = [(g, np.mean(imps), len(imps)) for g, imps in group_importances.items()]
    group_means.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Group':<25} {'Mean Importance':<18} {'Count':<8} {'Total Contribution':<20}")
    print("-" * 71)

    for group, mean_imp, count in group_means:
        total = mean_imp * count
        print(f"{group:<25} {mean_imp:.4f}            {count:<8} {total:.4f}")


def print_phase_analysis(stats: list[FeatureStats]):
    """Print phase-specific analysis for interesting features."""
    print("\n" + "=" * 80)
    print("PHASE ANALYSIS (features that change significantly at boom)")
    print("=" * 80)

    # Find features with big changes between phases
    changes = []
    for s in stats:
        if s.early_mean == 0 or np.isnan(s.boom_mean):
            continue
        # Relative change from early to boom
        if abs(s.early_mean) > 1e-9:
            change = (s.boom_mean - s.early_mean) / abs(s.early_mean)
        else:
            change = s.boom_mean - s.early_mean
        changes.append((s, change))

    # Sort by absolute change
    changes.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<40} {'Early':<12} {'Pre-boom':<12} {'Boom':<12} {'Post-boom':<12} {'Change':<10}")
    print("-" * 98)

    for s, change in changes[:20]:
        print(f"{s.name:<40} {s.early_mean:>10.4f}   {s.preboom_mean:>10.4f}   "
              f"{s.boom_mean:>10.4f}   {s.postboom_mean:>10.4f}   {change:>+8.1%}")


def compare_feature_sets(
    dataset,
    cache_default: FeatureCache,
    cache_caustic: FeatureCache,
):
    """Compare model performance with different feature sets."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: Default vs Caustic Features")
    print("=" * 80)

    # Run CV with default features
    print("\nRunning CV with DEFAULT features...")
    evaluator_default = CachedEvaluator(dataset, cache_default)
    result_default = evaluator_default.cross_validate_selective(
        lambda: BoomDetectionPipeline(accept_threshold=0.60),
        seeds=[42, 43, 44],
        verbose=False,
    )

    # Run CV with caustic features
    print("Running CV with CAUSTIC features...")
    evaluator_caustic = CachedEvaluator(dataset, cache_caustic)
    result_caustic = evaluator_caustic.cross_validate_selective(
        lambda: BoomDetectionPipeline(accept_threshold=0.60),
        seeds=[42, 43, 44],
        verbose=False,
    )

    print("\nResults:")
    print(f"{'Config':<20} {'Selective MAE':<18} {'Coverage':<15} {'AURC':<10}")
    print("-" * 63)

    d = result_default
    print(f"{'Default':<20} {d.mean_metrics['selective_mae']:.2f} +/- {d.std_metrics['selective_mae']:.2f}      "
          f"{d.mean_metrics['coverage']*100:.1f}%           {d.mean_metrics['aurc']:.2f}")

    d = result_caustic
    print(f"{'+ Caustic':<20} {d.mean_metrics['selective_mae']:.2f} +/- {d.std_metrics['selective_mae']:.2f}      "
          f"{d.mean_metrics['coverage']*100:.1f}%           {d.mean_metrics['aurc']:.2f}")

    # Compute improvement
    mae_default = result_default.mean_metrics['selective_mae']
    mae_caustic = result_caustic.mean_metrics['selective_mae']
    improvement = (mae_default - mae_caustic) / mae_default * 100

    print(f"\nImprovement: {improvement:+.1f}% MAE reduction")


def save_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    output_path: Path,
):
    """Save correlation heatmap as text (ASCII art style)."""
    # Group features for cleaner display
    n = len(feature_names)

    # For text output, just save the top correlations
    with open(output_path, 'w') as f:
        f.write("FEATURE CORRELATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        # Find highly correlated pairs (excluding self-correlation)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if not np.isnan(corr_matrix[i, j]):
                    pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        f.write("TOP CORRELATED FEATURE PAIRS:\n")
        f.write("-" * 60 + "\n")
        for f1, f2, corr in pairs[:50]:
            f.write(f"{corr:+.3f}  {f1} <-> {f2}\n")

        f.write("\n\nLEAST CORRELATED FEATURE PAIRS:\n")
        f.write("-" * 60 + "\n")
        for f1, f2, corr in pairs[-20:]:
            f.write(f"{corr:+.3f}  {f1} <-> {f2}\n")

    print(f"Saved correlation analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze boom detection features")
    parser.add_argument("--data-path", default="data", help="Path to data directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--save-plots", action="store_true", help="Save analysis to files")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for files")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=True, max_samples=args.max_samples)
    print(f"Loaded {len(dataset)} simulations")

    # Extract features with caustic config (includes caustic features)
    print("\nExtracting features with CAUSTIC_CONFIG...")
    config_caustic = FeatureConfig(
        max_pendulums=1000,
        include_caustic=True,
    )
    cache_caustic = FeatureCache(config=config_caustic)
    cache_caustic.extract_all(dataset, verbose=True)

    # Also extract with default config for comparison
    print("\nExtracting features with DEFAULT_CONFIG...")
    config_default = FeatureConfig(max_pendulums=1000)
    cache_default = FeatureCache(config=config_default)
    cache_default.extract_all(dataset, verbose=True)

    # Get feature info
    extractor = FeatureExtractor(config_caustic)
    feature_names = extractor.feature_names
    print(f"\nTotal features (with caustic): {len(feature_names)}")

    # Get simulation data
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])

    # Aggregate features for correlation analysis
    print("\nComputing feature statistics...")
    agg_features = np.array([cache_caustic[sid].mean(axis=0) for sid in sim_ids])

    stats = compute_feature_stats(
        agg_features, feature_names, boom_frames, sim_ids, cache_caustic
    )

    # Print analysis
    print_top_features(stats)
    print_feature_groups_summary(stats)
    print_phase_analysis(stats)

    # Compare performance
    compare_feature_sets(dataset, cache_default, cache_caustic)

    # Save analysis if requested
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Compute and save correlation matrix
        print("\nComputing feature correlations...")
        corr_matrix = compute_feature_correlations(cache_caustic, sim_ids, feature_names)
        save_correlation_heatmap(corr_matrix, feature_names, output_dir / "correlations.txt")

        # Save feature stats to CSV
        import csv
        stats_path = output_dir / "feature_stats.csv"
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'feature', 'group', 'importance', 'importance_std',
                'corr_with_distance', 'corr_pvalue',
                'mean', 'std', 'min', 'max',
                'early_mean', 'preboom_mean', 'boom_mean', 'postboom_mean'
            ])
            for s in stats:
                writer.writerow([
                    s.name, s.group, s.importance_mean, s.importance_std,
                    s.corr_with_boom_distance, s.corr_pvalue,
                    s.mean, s.std, s.min, s.max,
                    s.early_mean, s.preboom_mean, s.boom_mean, s.postboom_mean
                ])
        print(f"Saved feature stats to {stats_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
