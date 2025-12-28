#!/usr/bin/env python3
"""
Investigate causticness feature formulations.

The causticness feature is the most important for boom detection.
This script investigates:
1. Different n_bins values (angular resolution)
2. Different formulas (coverage, gini, entropy, etc.)
3. Different input angles (th1, th2, th1+th2, tip direction)
4. Feature importance comparison

The current formula is: causticness = coverage × gini
Where:
- coverage = (occupied bins) / (expected bins if random)  [birthday-corrected]
- gini = Gini coefficient of histogram (inequality measure)

High causticness means: pendulums spread across many angular bins
AND cluster unevenly (spiky histogram = some bins very full, others empty).

Usage:
    uv run python scripts/investigate_causticness.py data
    uv run python scripts/investigate_causticness.py data --quick
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy as scipy_entropy, spearmanr

from boom_detection.loader import load_dataset, TH1, TH2, X2, Y2


@dataclass
class CausticnessVariant:
    """A variant of causticness computation."""
    name: str
    description: str
    compute: callable  # (angles, n_bins) -> (frames,) array


# =============================================================================
# Helper functions (copied from features.py for standalone use)
# =============================================================================

def _wrap_2pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 2π)."""
    return angles % (2 * np.pi)


def _angular_histogram(angles: np.ndarray, n_bins: int = 36) -> np.ndarray:
    """
    Compute histogram of angles across pendulums for each frame.
    Returns shape (frames, n_bins).
    """
    n_frames = angles.shape[0]
    wrapped = _wrap_2pi(angles)
    bin_width = 2 * np.pi / n_bins
    bin_indices = np.clip((wrapped / bin_width).astype(int), 0, n_bins - 1)

    counts = np.zeros((n_frames, n_bins), dtype=np.float64)
    frame_indices = np.arange(n_frames)[:, None]
    np.add.at(counts, (np.broadcast_to(frame_indices, bin_indices.shape), bin_indices), 1)
    return counts


def _gini_coefficient(counts: np.ndarray) -> np.ndarray:
    """Compute Gini coefficient of histogram counts. Shape (frames,)."""
    sorted_counts = np.sort(counts, axis=1)
    n_bins = counts.shape[1]
    total = counts.sum(axis=1, keepdims=True)
    total = np.where(total < 1e-9, 1e-9, total)
    ranks = np.arange(1, n_bins + 1)
    weights = 2 * ranks - n_bins - 1
    numerator = (sorted_counts * weights).sum(axis=1)
    return numerator / (n_bins * total.squeeze())


def _birthday_corrected_coverage(counts: np.ndarray, n_samples: int) -> np.ndarray:
    """Compute birthday-corrected coverage (occupied bins / expected)."""
    n_bins = counts.shape[1]
    occupied = (counts > 0).sum(axis=1)
    raw_coverage = occupied / n_bins
    expected = 1 - (1 - 1/n_bins) ** n_samples
    if expected > 0.01:
        return np.minimum(1.0, raw_coverage / expected)
    else:
        return raw_coverage


def _mean_resultant_length(angles: np.ndarray) -> np.ndarray:
    """Compute mean resultant length (circular concentration). Shape (frames,)."""
    mean_cos = np.mean(np.cos(angles), axis=1)
    mean_sin = np.mean(np.sin(angles), axis=1)
    return np.sqrt(mean_cos**2 + mean_sin**2)


def _entropy(counts: np.ndarray) -> np.ndarray:
    """Compute normalized entropy of histogram. Shape (frames,)."""
    n_bins = counts.shape[1]
    max_entropy = np.log(n_bins)

    # Normalize to probabilities
    total = counts.sum(axis=1, keepdims=True)
    total = np.where(total < 1e-9, 1e-9, total)
    probs = counts / total

    # Compute entropy per frame
    result = np.zeros(counts.shape[0])
    for i in range(counts.shape[0]):
        p = probs[i]
        p = p[p > 0]  # avoid log(0)
        result[i] = scipy_entropy(p) / max_entropy if len(p) > 0 else 0
    return result


# =============================================================================
# Causticness Variants
# =============================================================================

def coverage_only(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Just the birthday-corrected coverage."""
    counts = _angular_histogram(angles, n_bins)
    n_pendulums = angles.shape[1]
    return _birthday_corrected_coverage(counts, n_pendulums)


def gini_only(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Just the Gini coefficient."""
    counts = _angular_histogram(angles, n_bins)
    return _gini_coefficient(counts)


def coverage_times_gini(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Current formula: coverage × gini."""
    counts = _angular_histogram(angles, n_bins)
    n_pendulums = angles.shape[1]
    coverage = _birthday_corrected_coverage(counts, n_pendulums)
    gini = _gini_coefficient(counts)
    return coverage * gini


def entropy_based(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Entropy-based: 1 - normalized_entropy (high when concentrated)."""
    counts = _angular_histogram(angles, n_bins)
    return 1 - _entropy(counts)


def dispersion_based(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Dispersion: 1 - R (high when spread out)."""
    return 1 - _mean_resultant_length(angles)


def max_bin_ratio(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Max bin count / mean bin count (spikiness)."""
    counts = _angular_histogram(angles, n_bins)
    max_count = np.max(counts, axis=1)
    mean_count = np.mean(counts, axis=1)
    mean_count = np.where(mean_count < 1e-9, 1e-9, mean_count)
    return max_count / mean_count / n_bins  # normalize to ~[0, 1]


def coverage_times_entropy(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """Coverage × (1 - entropy) - alternative to gini."""
    counts = _angular_histogram(angles, n_bins)
    n_pendulums = angles.shape[1]
    coverage = _birthday_corrected_coverage(counts, n_pendulums)
    return coverage * (1 - _entropy(counts))


def sqrt_coverage_times_gini(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """sqrt(coverage) × gini - less aggressive coverage scaling."""
    counts = _angular_histogram(angles, n_bins)
    n_pendulums = angles.shape[1]
    coverage = _birthday_corrected_coverage(counts, n_pendulums)
    gini = _gini_coefficient(counts)
    return np.sqrt(coverage) * gini


def coverage_times_gini_squared(angles: np.ndarray, n_bins: int) -> np.ndarray:
    """coverage × gini² - emphasize inequality more."""
    counts = _angular_histogram(angles, n_bins)
    n_pendulums = angles.shape[1]
    coverage = _birthday_corrected_coverage(counts, n_pendulums)
    gini = _gini_coefficient(counts)
    return coverage * gini * gini


CAUSTICNESS_VARIANTS = [
    CausticnessVariant("coverage", "Birthday-corrected bin coverage", coverage_only),
    CausticnessVariant("gini", "Gini coefficient of histogram", gini_only),
    CausticnessVariant("coverage×gini", "Current formula: coverage × gini", coverage_times_gini),
    CausticnessVariant("1-entropy", "1 - normalized entropy", entropy_based),
    CausticnessVariant("1-R", "1 - mean resultant length (dispersion)", dispersion_based),
    CausticnessVariant("max_ratio", "Max bin / mean bin (spikiness)", max_bin_ratio),
    CausticnessVariant("coverage×(1-H)", "coverage × (1 - entropy)", coverage_times_entropy),
    CausticnessVariant("√coverage×gini", "sqrt(coverage) × gini", sqrt_coverage_times_gini),
    CausticnessVariant("coverage×gini²", "coverage × gini²", coverage_times_gini_squared),
]


# =============================================================================
# Analysis Functions
# =============================================================================

def extract_angle_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """Extract different angle arrays from simulation data."""
    th1 = data[:, :, TH1]
    th2 = data[:, :, TH2]
    x2 = data[:, :, X2]
    y2 = data[:, :, Y2]

    return {
        'th1': th1,
        'th2': th2,
        'th1+th2': th1 + th2,
        'tip_dir': np.arctan2(x2, y2),
    }


def compute_all_variants(
    dataset,
    n_bins_options: list[int] = [18, 36, 72],
    max_samples: int | None = None,
    max_pendulums: int = 2000,
) -> dict[str, np.ndarray]:
    """
    Compute all causticness variants for all simulations.

    Returns dict mapping "variant_name:angle_type:n_bins" -> (n_sims,) array of mean values.
    """
    annotations = dataset.annotations
    if max_samples:
        annotations = annotations[:max_samples]

    results = {}
    rng = np.random.RandomState(42)

    for i, ann in enumerate(annotations):
        sim = dataset.simulations[ann.id]
        data = sim.data

        # Subsample pendulums for speed
        if data.shape[1] > max_pendulums:
            indices = rng.choice(data.shape[1], max_pendulums, replace=False)
            data = data[:, indices, :]

        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(annotations)}...")

        angle_features = extract_angle_features(data)

        for variant in CAUSTICNESS_VARIANTS:
            for angle_name, angles in angle_features.items():
                for n_bins in n_bins_options:
                    key = f"{variant.name}:{angle_name}:{n_bins}"
                    values = variant.compute(angles, n_bins)

                    # Aggregate per simulation (mean over frames)
                    if key not in results:
                        results[key] = []
                    results[key].append(np.mean(values))

    # Convert to arrays
    return {k: np.array(v) for k, v in results.items()}


def evaluate_variant_importance(
    features: dict[str, np.ndarray],
    boom_frames: np.ndarray,
    n_cv_folds: int = 5,
) -> dict[str, dict]:
    """
    Evaluate importance of each variant for boom prediction.

    Returns dict with correlation and CV score for each variant.
    """
    results = {}

    for key, values in features.items():
        # Correlation with boom frame
        corr, pval = spearmanr(values, boom_frames)

        # Quick CV score using single feature
        X = values.reshape(-1, 1)
        y = boom_frames

        model = HistGradientBoostingRegressor(
            max_iter=50, max_depth=3, random_state=42
        )
        try:
            scores = cross_val_score(model, X, y, cv=n_cv_folds, scoring='neg_mean_absolute_error')
            mae = -np.mean(scores)
            mae_std = np.std(scores)
        except Exception:
            mae = float('inf')
            mae_std = 0.0

        results[key] = {
            'correlation': corr,
            'pvalue': pval,
            'mae': mae,
            'mae_std': mae_std,
        }

    return results


def print_results_by_formula(results: dict[str, dict]):
    """Print results grouped by formula."""
    print("\n" + "=" * 80)
    print("RESULTS BY FORMULA")
    print("=" * 80)

    # Group by variant name
    by_variant = {}
    for key, metrics in results.items():
        variant, angle, n_bins = key.split(':')
        if variant not in by_variant:
            by_variant[variant] = []
        by_variant[variant].append((key, angle, int(n_bins), metrics))

    for variant_name in [v.name for v in CAUSTICNESS_VARIANTS]:
        if variant_name not in by_variant:
            continue

        print(f"\n{variant_name}:")
        print(f"  {'Angle':<12} {'Bins':>6} {'Corr':>8} {'MAE':>10}")
        print(f"  {'-'*40}")

        items = sorted(by_variant[variant_name], key=lambda x: x[3]['mae'])
        for key, angle, n_bins, m in items:
            print(f"  {angle:<12} {n_bins:>6} {m['correlation']:>+8.3f} {m['mae']:>7.2f}±{m['mae_std']:.2f}")


def print_best_variants(results: dict[str, dict], top_n: int = 20):
    """Print the best variants overall."""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} VARIANTS (by MAE)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])

    print(f"\n{'Variant':<25} {'Angle':<10} {'Bins':>5} {'Corr':>8} {'MAE':>12}")
    print("-" * 65)

    for key, m in sorted_results[:top_n]:
        parts = key.split(':')
        variant, angle, n_bins = parts[0], parts[1], parts[2]
        print(f"{variant:<25} {angle:<10} {n_bins:>5} {m['correlation']:>+8.3f} {m['mae']:>7.2f}±{m['mae_std']:.2f}")


def print_n_bins_comparison(results: dict[str, dict]):
    """Compare n_bins settings for each formula."""
    print("\n" + "=" * 80)
    print("N_BINS COMPARISON (for coverage×gini on th1)")
    print("=" * 80)

    # Filter for coverage×gini on th1
    filtered = {k: v for k, v in results.items() if k.startswith('coverage×gini:th1:')}

    print(f"\n{'n_bins':>10} {'Correlation':>12} {'MAE':>12}")
    print("-" * 40)

    for key in sorted(filtered.keys(), key=lambda x: int(x.split(':')[2])):
        n_bins = key.split(':')[2]
        m = filtered[key]
        print(f"{n_bins:>10} {m['correlation']:>+12.3f} {m['mae']:>8.2f}±{m['mae_std']:.2f}")


def compare_angle_sources(results: dict[str, dict]):
    """Compare different angle sources for the best formula."""
    print("\n" + "=" * 80)
    print("ANGLE SOURCE COMPARISON (for coverage×gini, n_bins=36)")
    print("=" * 80)

    # Filter for coverage×gini with n_bins=36
    filtered = {k: v for k, v in results.items() if ':36' in k and k.startswith('coverage×gini:')}

    print(f"\n{'Angle':<15} {'Correlation':>12} {'MAE':>12}")
    print("-" * 45)

    for key in sorted(filtered.keys(), key=lambda x: filtered[x]['mae']):
        angle = key.split(':')[1]
        m = filtered[key]
        print(f"{angle:<15} {m['correlation']:>+12.3f} {m['mae']:>8.2f}±{m['mae_std']:.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Investigate causticness formulations")
    parser.add_argument("data_path", nargs="?", default="data", help="Path to data directory")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer bins)")
    args = parser.parse_args()

    print("=" * 80)
    print("CAUSTICNESS FORMULATION INVESTIGATION")
    print("=" * 80)

    print(f"\nLoading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path, verbose=True, max_samples=args.max_samples)
    print(f"Loaded {len(dataset)} simulations")

    # Get boom frames
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])

    # Choose n_bins options
    if args.quick:
        n_bins_options = [36]
    else:
        n_bins_options = [12, 18, 24, 36, 48, 72]

    print(f"\nComputing causticness variants...")
    print(f"  Formulas: {len(CAUSTICNESS_VARIANTS)}")
    print(f"  Angle sources: 4 (th1, th2, th1+th2, tip_dir)")
    print(f"  n_bins options: {n_bins_options}")
    print(f"  Total variants: {len(CAUSTICNESS_VARIANTS) * 4 * len(n_bins_options)}")

    features = compute_all_variants(dataset, n_bins_options, args.max_samples)

    print(f"\nEvaluating variant importance...")
    results = evaluate_variant_importance(features, boom_frames)

    # Print analysis
    print_best_variants(results, top_n=25)
    print_n_bins_comparison(results)
    compare_angle_sources(results)
    print_results_by_formula(results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find the best overall
    best_key = min(results.keys(), key=lambda k: results[k]['mae'])
    best = results[best_key]
    parts = best_key.split(':')

    print(f"\nBest variant: {parts[0]}")
    print(f"  Angle source: {parts[1]}")
    print(f"  n_bins: {parts[2]}")
    print(f"  Correlation: {best['correlation']:+.3f}")
    print(f"  MAE: {best['mae']:.2f} ± {best['mae_std']:.2f}")

    # Compare to current default
    default_key = "coverage×gini:th1:36"
    if default_key in results:
        default = results[default_key]
        print(f"\nCurrent default ({default_key}):")
        print(f"  Correlation: {default['correlation']:+.3f}")
        print(f"  MAE: {default['mae']:.2f} ± {default['mae_std']:.2f}")

        improvement = (default['mae'] - best['mae']) / default['mae'] * 100
        if improvement > 0:
            print(f"\n  Improvement possible: {improvement:.1f}% MAE reduction")
        else:
            print(f"\n  Current default is {'optimal' if improvement == 0 else 'near-optimal'}!")


if __name__ == "__main__":
    main()
