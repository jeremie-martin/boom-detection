"""
Confidence Calibration Check.

This script investigates whether the accept_score is well-calibrated.

A well-calibrated confidence should mean: "when I say 70% confident, I'm right 70% of the time."

We compute:
1. Reliability diagrams (predicted confidence vs actual accuracy)
2. Expected Calibration Error (ECE)
3. Brier score

Usage:
    uv run python scripts/calibration_check.py data
    uv run python scripts/calibration_check.py data --plot  # Save reliability diagram
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import KFold

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.sequence_models import set_global_seed


def compute_calibration_metrics(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics.

    Args:
        confidences: Array of confidence scores (0-1)
        correct: Binary array indicating if prediction was correct
        n_bins: Number of bins for ECE calculation

    Returns:
        Dictionary with calibration metrics
    """
    # Bin boundaries
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute per-bin statistics
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            in_bin = in_bin | (confidences == bin_edges[i + 1])

        n_in_bin = in_bin.sum()
        bin_counts.append(n_in_bin)

        if n_in_bin > 0:
            bin_accuracies.append(correct[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)

    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)

    # Expected Calibration Error (ECE)
    # ECE = sum(|bin_acc - bin_conf| * bin_count) / total_count
    total = bin_counts.sum()
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += abs(bin_accuracies[i] - bin_confidences[i]) * bin_counts[i]
    ece /= total

    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            mce = max(mce, abs(bin_accuracies[i] - bin_confidences[i]))

    # Brier score: mean((confidence - correct)^2)
    brier = np.mean((confidences - correct) ** 2)

    # Average confidence and accuracy
    avg_confidence = confidences.mean()
    avg_accuracy = correct.mean()

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'avg_confidence': avg_confidence,
        'avg_accuracy': avg_accuracy,
        'overconfidence': avg_confidence - avg_accuracy,  # Positive = overconfident
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
    }


def plot_reliability_diagram(metrics: dict, output_path: str = None):
    """Plot reliability diagram."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    bin_centers = metrics['bin_centers']
    bin_accuracies = metrics['bin_accuracies']
    bin_counts = metrics['bin_counts']

    # Remove NaN bins for plotting
    valid = ~np.isnan(bin_accuracies)
    centers = bin_centers[valid]
    accs = bin_accuracies[valid]
    counts = bin_counts[valid]

    # Bar width
    width = 0.08

    # Plot bars
    ax1.bar(centers, accs, width=width, alpha=0.7, label='Actual accuracy')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    ax1.set_xlabel('Mean predicted confidence')
    ax1.set_ylabel('Actual accuracy')
    ax1.set_title(f"Reliability Diagram (ECE={metrics['ece']:.3f})")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confidence distribution
    ax2.bar(bin_centers, bin_counts, width=width, alpha=0.7, color='orange')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {output_path}")
    else:
        plt.show()


def run_calibration_check(
    data_path: str,
    error_threshold: int = 5,  # Within N frames = "correct"
    n_folds: int = 5,
    seeds: list[int] = None,
    plot: bool = False,
    verbose: bool = True,
):
    """
    Run calibration check experiment.

    Args:
        data_path: Path to data directory
        error_threshold: Max error for prediction to be considered "correct"
        n_folds: Number of CV folds
        seeds: Random seeds
        plot: Whether to generate plots
        verbose: Print progress
    """
    if seeds is None:
        seeds = [42, 43, 44]

    print(f"Loading dataset from: {data_path}")
    dataset = load_dataset(data_path, verbose=verbose)
    print(f"Loaded {len(dataset)} simulations")

    # Set up feature cache
    config = FeatureConfig(max_pendulums=2000)
    cache = FeatureCache(config=config, cache_dir='.feature_cache')
    cache.extract_all(dataset, verbose=verbose)

    # Get data
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    # Collect all predictions across seeds and folds
    all_confidences = []
    all_correct = []
    all_accept_scores = []
    all_accepted = []

    print(f"\nRunning {n_folds}-fold CV × {len(seeds)} seeds")
    print(f"Error threshold for 'correct': {error_threshold} frames")
    print("=" * 60)

    for seed in seeds:
        if verbose:
            print(f"\nSeed {seed}:")

        set_global_seed(seed)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sim_ids)):
            train_ids = [sim_ids[i] for i in train_idx]
            test_ids = [sim_ids[i] for i in test_idx]
            train_booms = boom_frames[train_idx]
            train_quals = qualities[train_idx]
            test_booms = boom_frames[test_idx]

            # Train pipeline
            pipeline = BoomDetectionPipeline(
                accept_threshold=0.0,  # Accept all for calibration analysis
                seed=seed * 1000 + fold_idx,
            )
            pipeline.fit(train_ids, train_booms, train_quals, cache)

            # Get predictions with confidence scores
            for i, test_id in enumerate(test_ids):
                result = pipeline.predict_one(cache[test_id])

                # Use accept_score as confidence
                confidence = result.accept_score if result.accept_score else 0.0
                error = abs(result.cnn_pred - test_booms[i])
                correct = int(error <= error_threshold)

                all_confidences.append(confidence)
                all_correct.append(correct)
                all_accept_scores.append(result.accept_score)
                all_accepted.append(result.accepted)

            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds} complete")

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # Compute calibration metrics
    metrics = compute_calibration_metrics(all_confidences, all_correct)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Confidence Calibration Check")
    print("=" * 60)
    print(f"Error threshold: {error_threshold} frames")
    print(f"Total predictions: {len(all_confidences)}")
    print()
    print(f"Expected Calibration Error (ECE): {metrics['ece']:.4f}")
    print(f"Maximum Calibration Error (MCE): {metrics['mce']:.4f}")
    print(f"Brier Score:                      {metrics['brier']:.4f}")
    print()
    print(f"Average Confidence:               {metrics['avg_confidence']:.4f}")
    print(f"Average Accuracy:                 {metrics['avg_accuracy']:.4f}")
    print(f"Overconfidence:                   {metrics['overconfidence']:+.4f}")
    print()

    # Interpretation
    if metrics['ece'] < 0.05:
        print("Interpretation: Well-calibrated (ECE < 0.05)")
    elif metrics['ece'] < 0.10:
        print("Interpretation: Moderately calibrated (0.05 <= ECE < 0.10)")
    else:
        print("Interpretation: Poorly calibrated (ECE >= 0.10)")

    if metrics['overconfidence'] > 0.05:
        print(f"  -> Model is overconfident by {metrics['overconfidence']:.2%}")
    elif metrics['overconfidence'] < -0.05:
        print(f"  -> Model is underconfident by {-metrics['overconfidence']:.2%}")
    else:
        print("  -> Confidence roughly matches accuracy")

    # Per-bin breakdown
    print("\nPer-bin breakdown:")
    print(f"{'Bin':>12} {'Count':>8} {'Confidence':>12} {'Accuracy':>12} {'Gap':>10}")
    print("-" * 54)
    for i in range(len(metrics['bin_centers'])):
        if metrics['bin_counts'][i] > 0:
            gap = metrics['bin_accuracies'][i] - metrics['bin_confidences'][i]
            print(f"{metrics['bin_centers'][i]:>12.2f} {metrics['bin_counts'][i]:>8d} "
                  f"{metrics['bin_confidences'][i]:>12.3f} {metrics['bin_accuracies'][i]:>12.3f} "
                  f"{gap:>+10.3f}")

    if plot:
        plot_reliability_diagram(metrics, 'reliability_diagram.png')

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence calibration check")
    parser.add_argument("data_path", nargs="?", default="data")
    parser.add_argument("--error-threshold", type=int, default=5,
                        help="Max error for 'correct' (default: 5 frames)")
    parser.add_argument("-k", "--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--plot", action="store_true", help="Save reliability diagram")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))

    run_calibration_check(
        args.data_path,
        error_threshold=args.error_threshold,
        n_folds=args.folds,
        seeds=seeds,
        plot=args.plot,
        verbose=not args.quiet,
    )
