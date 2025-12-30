#!/usr/bin/env python3
"""
B4: Temporal Error Bias Analysis.

Are predictions systematically early or late? Analyze signed errors
to detect any temporal bias in predictions.

Usage:
    uv run python scripts/b4_temporal_bias.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, SelectivePrediction
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='B4: Temporal Error Bias Analysis')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/b4_temporal_bias'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("b4_temporal_bias")

    logger.info("=" * 70)
    logger.info("B4: Temporal Error Bias Analysis")
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

    # Collect data across all seeds/folds
    all_data = []

    logger.info("Running 5-fold CV to collect predictions...")
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
            )
            pipeline.set_seed(fold_seed)
            pipeline.fit(train_ids, boom_frames[train_idx], boom_qualities[train_idx], cache)

            # Predict
            predictions = pipeline.predict(test_ids, cache)

            # Record data for each test sample
            for pred, test_i in zip(predictions, test_idx):
                true_boom = int(boom_frames[test_i])
                true_qual = float(boom_qualities[test_i])

                if isinstance(pred, dict):
                    sp = SelectivePrediction.from_dict(pred)
                else:
                    sp = pred

                # Get individual model predictions
                cnn_pred = sp.model_predictions.get('cnn')
                hgb_pred = sp.model_predictions.get('hgb')
                lstm_pred = sp.model_predictions.get('lstm')

                if cnn_pred is None or hgb_pred is None or lstm_pred is None:
                    continue

                # Compute signed errors (positive = late, negative = early)
                cnn_signed = cnn_pred - true_boom
                hgb_signed = hgb_pred - true_boom
                lstm_signed = lstm_pred - true_boom

                # Final prediction
                if sp.boom_frame is not None:
                    final_pred = sp.boom_frame
                else:
                    final_pred = int(np.median([cnn_pred, hgb_pred, lstm_pred]))

                final_signed = final_pred - true_boom

                all_data.append({
                    'seed': seed,
                    'fold': fold_idx,
                    'true_boom': true_boom,
                    'true_quality': true_qual,
                    'cnn_signed': cnn_signed,
                    'hgb_signed': hgb_signed,
                    'lstm_signed': lstm_signed,
                    'final_signed': final_signed,
                    'accepted': sp.accepted,
                })

            logger.info(f"  Seed {seed}, Fold {fold_idx+1}/5: {len(predictions)} samples")

    # Convert to arrays
    cnn_signed = np.array([d['cnn_signed'] for d in all_data])
    hgb_signed = np.array([d['hgb_signed'] for d in all_data])
    lstm_signed = np.array([d['lstm_signed'] for d in all_data])
    final_signed = np.array([d['final_signed'] for d in all_data])
    true_quals = np.array([d['true_quality'] for d in all_data])
    accepted = np.array([d['accepted'] for d in all_data])

    # Overall bias analysis
    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL TEMPORAL BIAS (positive = late, negative = early)")
    logger.info("=" * 70)

    logger.info(f"\nCNN:  mean={np.mean(cnn_signed):+.2f}, median={np.median(cnn_signed):+.1f}")
    logger.info(f"HGB:  mean={np.mean(hgb_signed):+.2f}, median={np.median(hgb_signed):+.1f}")
    logger.info(f"LSTM: mean={np.mean(lstm_signed):+.2f}, median={np.median(lstm_signed):+.1f}")
    logger.info(f"Final: mean={np.mean(final_signed):+.2f}, median={np.median(final_signed):+.1f}")

    # Statistical test for bias (t-test vs 0)
    logger.info("")
    logger.info("Statistical test for bias (t-test vs 0):")
    for name, data in [('CNN', cnn_signed), ('HGB', hgb_signed), ('LSTM', lstm_signed), ('Final', final_signed)]:
        t_stat, p_val = stats.ttest_1samp(data, 0)
        sig = "**" if p_val < 0.05 else ""
        logger.info(f"  {name}: t={t_stat:.2f}, p={p_val:.3f} {sig}")

    # Bias by quality
    logger.info("")
    logger.info("=" * 70)
    logger.info("BIAS BY QUALITY STRATUM")
    logger.info("=" * 70)

    quality_bins = [
        ('Low (0-0.5)', true_quals < 0.5),
        ('Med (0.5-0.7)', (true_quals >= 0.5) & (true_quals < 0.7)),
        ('High (0.7+)', true_quals >= 0.7),
    ]

    bias_by_quality = []
    for bin_name, mask in quality_bins:
        if mask.sum() == 0:
            continue

        final_sub = final_signed[mask]
        n = len(final_sub)
        mean_bias = np.mean(final_sub)
        median_bias = np.median(final_sub)

        # Per-model
        cnn_bias = np.mean(cnn_signed[mask])
        hgb_bias = np.mean(hgb_signed[mask])
        lstm_bias = np.mean(lstm_signed[mask])

        bias_by_quality.append({
            'bin': bin_name,
            'n': n,
            'final_mean': mean_bias,
            'final_median': median_bias,
            'cnn_mean': cnn_bias,
            'hgb_mean': hgb_bias,
            'lstm_mean': lstm_bias,
        })

        logger.info(f"\n{bin_name} (n={n}):")
        logger.info(f"  Final: mean={mean_bias:+.2f}, median={median_bias:+.1f}")
        logger.info(f"  CNN: {cnn_bias:+.2f}, HGB: {hgb_bias:+.2f}, LSTM: {lstm_bias:+.2f}")

    # Bias for accepted vs rejected
    logger.info("")
    logger.info("=" * 70)
    logger.info("BIAS: ACCEPTED VS REJECTED")
    logger.info("=" * 70)

    if accepted.sum() > 0:
        acc_bias = np.mean(final_signed[accepted])
        acc_median = np.median(final_signed[accepted])
        logger.info(f"Accepted (n={accepted.sum()}): mean={acc_bias:+.2f}, median={acc_median:+.1f}")

    if (~accepted).sum() > 0:
        rej_bias = np.mean(final_signed[~accepted])
        rej_median = np.median(final_signed[~accepted])
        logger.info(f"Rejected (n={(~accepted).sum()}): mean={rej_bias:+.2f}, median={rej_median:+.1f}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Plot 1: Histogram of signed errors
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, data) in zip(axes.flat, [('CNN', cnn_signed), ('HGB', hgb_signed),
                                              ('LSTM', lstm_signed), ('Final', final_signed)]):
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(data), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(data):+.1f}')
        ax.axvline(x=np.median(data), color='orange', linestyle='-', linewidth=2,
                   label=f'Median: {np.median(data):+.1f}')
        ax.set_xlabel('Signed Error (frames)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Signed Error Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.output / 'signed_error_histograms.png', dpi=150)
    plt.close()

    # Plot 2: Bias by quality
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bias_by_quality))
    width = 0.2

    cnn_biases = [r['cnn_mean'] for r in bias_by_quality]
    hgb_biases = [r['hgb_mean'] for r in bias_by_quality]
    lstm_biases = [r['lstm_mean'] for r in bias_by_quality]
    final_biases = [r['final_mean'] for r in bias_by_quality]

    ax.bar(x - 1.5*width, cnn_biases, width, label='CNN', alpha=0.7)
    ax.bar(x - 0.5*width, hgb_biases, width, label='HGB', alpha=0.7)
    ax.bar(x + 0.5*width, lstm_biases, width, label='LSTM', alpha=0.7)
    ax.bar(x + 1.5*width, final_biases, width, label='Final', alpha=0.7, color='black')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Quality Stratum')
    ax.set_ylabel('Mean Signed Error (frames)')
    ax.set_title('Temporal Bias by Quality (positive = late)')
    ax.set_xticks(x)
    ax.set_xticklabels([r['bin'] for r in bias_by_quality])
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output / 'bias_by_quality.png', dpi=150)
    plt.close()

    # Plot 3: Scatter of signed error vs true quality
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue' if a else 'red' for a in accepted]
    ax.scatter(true_quals, final_signed, c=colors, alpha=0.5, s=20)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('True Quality')
    ax.set_ylabel('Signed Error (frames)')
    ax.set_title('Signed Error vs True Quality')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Accepted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Rejected'),
    ])
    plt.tight_layout()
    plt.savefig(args.output / 'signed_error_vs_quality.png', dpi=150)
    plt.close()

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# B4: Temporal Error Bias Analysis\n\n")
        f.write("## Objective\n")
        f.write("Determine if predictions are systematically early or late.\n")
        f.write("Positive signed error = late prediction, negative = early.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Total samples: {len(all_data)} (5-fold CV x {len(args.seeds)} seeds)\n\n")

        f.write("## Overall Bias\n\n")
        f.write("| Model | Mean Bias | Median Bias | t-statistic | p-value |\n")
        f.write("|-------|-----------|-------------|-------------|----------|\n")
        for name, data in [('CNN', cnn_signed), ('HGB', hgb_signed), ('LSTM', lstm_signed), ('Final', final_signed)]:
            t_stat, p_val = stats.ttest_1samp(data, 0)
            sig = "**" if p_val < 0.05 else ""
            f.write(f"| {name} | {np.mean(data):+.2f} | {np.median(data):+.1f} | {t_stat:.2f} | {p_val:.3f}{sig} |\n")

        f.write("\n## Bias by Quality\n\n")
        f.write("| Quality | N | Final Mean | Final Median | CNN | HGB | LSTM |\n")
        f.write("|---------|---|------------|--------------|-----|-----|------|\n")
        for r in bias_by_quality:
            f.write(f"| {r['bin']} | {r['n']} | {r['final_mean']:+.2f} | {r['final_median']:+.1f} | {r['cnn_mean']:+.2f} | {r['hgb_mean']:+.2f} | {r['lstm_mean']:+.2f} |\n")

        f.write("\n## Accepted vs Rejected\n\n")
        if accepted.sum() > 0:
            f.write(f"- Accepted (n={accepted.sum()}): mean={np.mean(final_signed[accepted]):+.2f}, median={np.median(final_signed[accepted]):+.1f}\n")
        if (~accepted).sum() > 0:
            f.write(f"- Rejected (n={(~accepted).sum()}): mean={np.mean(final_signed[~accepted]):+.2f}, median={np.median(final_signed[~accepted]):+.1f}\n")

        f.write("\n## Analysis\n\n")

        # Determine if there's significant bias
        _, final_p = stats.ttest_1samp(final_signed, 0)
        mean_bias = np.mean(final_signed)

        if final_p < 0.05 and abs(mean_bias) > 1.0:
            direction = "late" if mean_bias > 0 else "early"
            f.write(f"**Significant temporal bias detected!**\n\n")
            f.write(f"Predictions are systematically {direction} by {abs(mean_bias):.1f} frames on average.\n\n")
            f.write("**Recommendation:** Consider implementing bias correction (D2) by:\n")
            f.write(f"1. Subtracting {mean_bias:.1f} frames from predictions\n")
            f.write("2. Or training models with shifted labels\n")
        elif final_p < 0.05:
            direction = "late" if mean_bias > 0 else "early"
            f.write(f"**Minor temporal bias detected.**\n\n")
            f.write(f"Predictions tend {direction} by {abs(mean_bias):.1f} frames, but effect is small.\n\n")
            f.write("**Recommendation:** Bias correction may help marginally but not critical.\n")
        else:
            f.write("**No significant temporal bias detected.**\n\n")
            f.write("Predictions are approximately centered around ground truth.\n\n")
            f.write("**Recommendation:** No bias correction needed.\n")

        # Check for quality-dependent bias
        f.write("\n### Quality-Dependent Bias\n\n")
        if len(bias_by_quality) >= 2:
            low_bias = bias_by_quality[0]['final_mean']
            high_bias = bias_by_quality[-1]['final_mean']
            if abs(low_bias - high_bias) > 2.0:
                f.write(f"Bias differs significantly by quality:\n")
                f.write(f"- Low quality: {low_bias:+.1f} frames\n")
                f.write(f"- High quality: {high_bias:+.1f} frames\n\n")
                f.write("Consider quality-dependent bias correction.\n")
            else:
                f.write("Bias is relatively consistent across quality levels.\n")

        f.write("\n## Plots\n")
        f.write("- `signed_error_histograms.png`: Distribution of signed errors per model\n")
        f.write("- `bias_by_quality.png`: Mean bias by quality stratum\n")
        f.write("- `signed_error_vs_quality.png`: Scatter of signed error vs quality\n")

    logger.info(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
