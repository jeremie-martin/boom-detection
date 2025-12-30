#!/usr/bin/env python3
"""
B5: Quality Calibration Verification.

Test whether isotonic regression calibration helps quality predictions.
Compare calibrated vs raw quality predictions.

Usage:
    uv run python scripts/b5_quality_calibration.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, SelectivePrediction
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def compute_calibration_error(pred_probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    """Compute expected calibration error (ECE) and bin data for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    ece = 0.0
    total_samples = len(pred_probs)

    for i in range(n_bins):
        mask = (pred_probs >= bins[i]) & (pred_probs < bins[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (pred_probs >= bins[i]) & (pred_probs <= bins[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_confidence = np.mean(pred_probs[mask])
            avg_accuracy = np.mean(true_labels[mask])
            ece += (n_in_bin / total_samples) * abs(avg_accuracy - avg_confidence)

            bin_data.append({
                'bin_center': (bins[i] + bins[i + 1]) / 2,
                'n': n_in_bin,
                'avg_pred': avg_confidence,
                'avg_true': avg_accuracy,
            })

    return ece, bin_data


def main():
    parser = argparse.ArgumentParser(description='B5: Quality Calibration Verification')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/b5_calibration'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("b5_quality_calibration")

    logger.info("=" * 70)
    logger.info("B5: Quality Calibration Verification")
    logger.info("=" * 70)

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

    # Combiner config
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    # Test configurations
    configs = [
        ('calibrated', True),
        ('uncalibrated', False),
    ]

    results = {}

    for config_name, calibrate in configs:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {config_name}")
        logger.info("=" * 70)

        all_data = []

        for seed in args.seeds:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            sim_ids_arr = np.array(evaluator.sim_ids)
            boom_frames = evaluator.boom_frames
            boom_qualities = evaluator.boom_qualities

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sim_ids_arr)):
                fold_seed = seed * 1000 + fold_idx

                train_ids = sim_ids_arr[train_idx].tolist()
                test_ids = sim_ids_arr[test_idx].tolist()

                # Train pipeline
                pipeline = BoomDetectionPipeline(
                    frame_models=('cnn', 'hgb', 'lstm'),
                    combiner=combiner,
                    calibrate_quality=calibrate,
                )
                pipeline.set_seed(fold_seed)
                pipeline.fit(train_ids, boom_frames[train_idx], boom_qualities[train_idx], cache)

                # Predict
                predictions = pipeline.predict(test_ids, cache)

                # Record data
                for pred, test_i in zip(predictions, test_idx):
                    true_qual = float(boom_qualities[test_i])
                    true_boom = int(boom_frames[test_i])

                    if isinstance(pred, dict):
                        sp = SelectivePrediction.from_dict(pred)
                    else:
                        sp = pred

                    pred_boom = sp.boom_frame if sp.boom_frame is not None else int(np.median(list(sp.model_predictions.values())))
                    error = abs(pred_boom - true_boom)

                    all_data.append({
                        'true_quality': true_qual,
                        'pred_quality': sp.predicted_quality,
                        'accepted': sp.accepted,
                        'error': error,
                    })

            logger.info(f"  Seed {seed}: completed")

        # Analyze
        pred_quals = np.array([d['pred_quality'] for d in all_data])
        true_quals = np.array([d['true_quality'] for d in all_data])
        accepted = np.array([d['accepted'] for d in all_data])
        errors = np.array([d['error'] for d in all_data])

        # For reliability diagram, we need binary labels
        # Let's use threshold of 0.7 to define "high quality"
        high_qual_threshold = 0.7
        true_high_qual = true_quals >= high_qual_threshold

        # Compute calibration error
        ece, bin_data = compute_calibration_error(pred_quals, true_high_qual)

        # Compute selective metrics
        if accepted.sum() > 0:
            selective_mae = np.mean(errors[accepted])
        else:
            selective_mae = float('nan')
        coverage = accepted.mean()

        # Correlation between predicted and true quality
        from scipy import stats
        spearman_r, spearman_p = stats.spearmanr(pred_quals, true_quals)

        results[config_name] = {
            'ece': ece,
            'bin_data': bin_data,
            'selective_mae': selective_mae,
            'coverage': coverage,
            'spearman_r': spearman_r,
            'pred_quals': pred_quals,
            'true_quals': true_quals,
            'true_high_qual': true_high_qual,
        }

        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.2e})")
        logger.info(f"  Selective MAE: {selective_mae:.2f} at {coverage:.1%} coverage")

    # Compare
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    for config_name, r in results.items():
        logger.info(f"{config_name}: ECE={r['ece']:.4f}, MAE={r['selective_mae']:.2f}, Coverage={r['coverage']:.1%}, r={r['spearman_r']:.3f}")

    ece_diff = results['uncalibrated']['ece'] - results['calibrated']['ece']
    mae_diff = results['uncalibrated']['selective_mae'] - results['calibrated']['selective_mae']

    logger.info(f"\nCalibration improvement:")
    logger.info(f"  ECE reduction: {ece_diff:.4f} ({'better' if ece_diff > 0 else 'worse'} with calibration)")
    logger.info(f"  MAE change: {mae_diff:+.2f} frames ({'better' if mae_diff > 0 else 'worse'} with calibration)")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Plot 1: Reliability diagrams
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (config_name, r) in zip(axes, results.items()):
        bin_data = r['bin_data']
        if bin_data:
            bin_centers = [b['bin_center'] for b in bin_data]
            avg_preds = [b['avg_pred'] for b in bin_data]
            avg_trues = [b['avg_true'] for b in bin_data]
            counts = [b['n'] for b in bin_data]

            # Bar chart
            ax.bar(bin_centers, avg_trues, width=0.08, alpha=0.7, label='Actual')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

            # Add sample counts
            for bc, at, n in zip(bin_centers, avg_trues, counts):
                ax.annotate(f'n={n}', (bc, at + 0.02), ha='center', fontsize=8)

        ax.set_xlabel('Predicted Quality')
        ax.set_ylabel('Actual Fraction High Quality')
        ax.set_title(f'{config_name.capitalize()} (ECE={r["ece"]:.4f})')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(args.output / 'reliability_diagrams.png', dpi=150)
    plt.close()

    # Plot 2: Predicted vs True Quality scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (config_name, r) in zip(axes, results.items()):
        ax.scatter(r['true_quals'], r['pred_quals'], alpha=0.5, s=20)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('True Quality')
        ax.set_ylabel('Predicted Quality')
        ax.set_title(f'{config_name.capitalize()} (r={r["spearman_r"]:.3f})')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(args.output / 'quality_scatter.png', dpi=150)
    plt.close()

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# B5: Quality Calibration Verification\n\n")
        f.write("## Objective\n")
        f.write("Test whether isotonic regression calibration improves quality predictions.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Models: CNN + HGB + LSTM (3-model pipeline)\n")
        f.write(f"- Combiner: std/s=15/t=0.70\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | ECE | Spearman r | Selective MAE | Coverage |\n")
        f.write("|---------------|-----|------------|---------------|----------|\n")
        for config_name, r in results.items():
            f.write(f"| {config_name} | {r['ece']:.4f} | {r['spearman_r']:.3f} | {r['selective_mae']:.2f} | {r['coverage']:.1%} |\n")

        f.write("\n## Comparison\n\n")
        f.write(f"- ECE reduction from calibration: {ece_diff:.4f}\n")
        f.write(f"- MAE change from calibration: {mae_diff:+.2f} frames\n\n")

        f.write("## Analysis\n\n")

        # Determine recommendation
        if ece_diff > 0.01 and mae_diff >= 0:
            f.write("**Calibration helps.** Lower ECE means predictions are better calibrated,\n")
            f.write("and MAE is not worse.\n\n")
            f.write("**Recommendation:** Keep `calibrate_quality=True` (current default)\n")
        elif ece_diff < -0.01:
            f.write("**Calibration hurts.** ECE is actually worse with calibration.\n")
            f.write("This might indicate overfitting of the calibration model.\n\n")
            f.write("**Recommendation:** Consider using `calibrate_quality=False`\n")
        else:
            f.write("**Calibration has minimal effect.** ECE difference is small.\n\n")
            f.write("**Recommendation:** Either setting works; calibration is optional.\n")

        f.write("\n## Expected Calibration Error (ECE)\n\n")
        f.write("ECE measures how well predicted probabilities match actual outcomes.\n")
        f.write("Lower is better. A perfectly calibrated model has ECE=0.\n\n")

        f.write("## Plots\n")
        f.write("- `reliability_diagrams.png`: Shows calibration quality\n")
        f.write("- `quality_scatter.png`: Predicted vs true quality\n")

    logger.info(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
