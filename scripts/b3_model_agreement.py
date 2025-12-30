#!/usr/bin/env python3
"""
B3: Model Agreement Analysis.

When models disagree, which one is correct? Analyze which model is most
accurate under different disagreement conditions.

Usage:
    uv run python scripts/b3_model_agreement.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boom_detection.combine import ThresholdCombiner, disagreement, ModelPrediction
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, SelectivePrediction
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='B3: Model Agreement Analysis')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/b3_model_agreement'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("b3_model_agreement")

    logger.info("=" * 70)
    logger.info("B3: Model Agreement Analysis")
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

                # Compute errors for each model
                cnn_error = abs(cnn_pred - true_boom)
                hgb_error = abs(hgb_pred - true_boom)
                lstm_error = abs(lstm_pred - true_boom)

                # Find best model
                errors = {'cnn': cnn_error, 'hgb': hgb_error, 'lstm': lstm_error}
                best_model = min(errors, key=errors.get)

                # Compute disagreement metrics
                model_preds = [ModelPrediction(m, f) for m, f in sp.model_predictions.items()]
                dis_std = disagreement(model_preds, metric='std')
                dis_range = disagreement(model_preds, metric='range')

                # Find which model is outlier (furthest from median)
                preds_array = np.array([cnn_pred, hgb_pred, lstm_pred])
                median_pred = np.median(preds_array)
                distances = {
                    'cnn': abs(cnn_pred - median_pred),
                    'hgb': abs(hgb_pred - median_pred),
                    'lstm': abs(lstm_pred - median_pred),
                }
                outlier_model = max(distances, key=distances.get)

                all_data.append({
                    'seed': seed,
                    'fold': fold_idx,
                    'true_boom': true_boom,
                    'true_quality': true_qual,
                    'cnn_pred': cnn_pred,
                    'hgb_pred': hgb_pred,
                    'lstm_pred': lstm_pred,
                    'cnn_error': cnn_error,
                    'hgb_error': hgb_error,
                    'lstm_error': lstm_error,
                    'best_model': best_model,
                    'disagreement_std': dis_std,
                    'disagreement_range': dis_range,
                    'outlier_model': outlier_model,
                    'accepted': sp.accepted,
                })

            logger.info(f"  Seed {seed}, Fold {fold_idx+1}/5: {len(predictions)} samples")

    # Analysis
    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL MODEL ACCURACY")
    logger.info("=" * 70)

    cnn_errors = np.array([d['cnn_error'] for d in all_data])
    hgb_errors = np.array([d['hgb_error'] for d in all_data])
    lstm_errors = np.array([d['lstm_error'] for d in all_data])

    logger.info(f"CNN:  MAE {np.mean(cnn_errors):.2f}, Median {np.median(cnn_errors):.1f}")
    logger.info(f"HGB:  MAE {np.mean(hgb_errors):.2f}, Median {np.median(hgb_errors):.1f}")
    logger.info(f"LSTM: MAE {np.mean(lstm_errors):.2f}, Median {np.median(lstm_errors):.1f}")

    # Which model wins most often
    best_counts = defaultdict(int)
    for d in all_data:
        best_counts[d['best_model']] += 1

    total = len(all_data)
    logger.info("")
    logger.info("Model wins (closest to ground truth):")
    for model in ['cnn', 'hgb', 'lstm']:
        logger.info(f"  {model}: {best_counts[model]} ({100*best_counts[model]/total:.1f}%)")

    # Breakdown by disagreement magnitude
    logger.info("")
    logger.info("=" * 70)
    logger.info("ACCURACY BY DISAGREEMENT LEVEL")
    logger.info("=" * 70)

    dis_std = np.array([d['disagreement_std'] for d in all_data])

    # Define bins
    bins = [
        ('Low (<3)', dis_std < 3),
        ('Medium (3-8)', (dis_std >= 3) & (dis_std < 8)),
        ('High (>=8)', dis_std >= 8),
    ]

    bin_results = []
    for bin_name, mask in bins:
        if mask.sum() == 0:
            continue

        subset = [d for d, m in zip(all_data, mask) if m]
        n = len(subset)

        cnn_mae = np.mean([d['cnn_error'] for d in subset])
        hgb_mae = np.mean([d['hgb_error'] for d in subset])
        lstm_mae = np.mean([d['lstm_error'] for d in subset])

        # Who wins in this bin
        wins = defaultdict(int)
        for d in subset:
            wins[d['best_model']] += 1

        best_in_bin = max(wins, key=wins.get)
        best_pct = 100 * wins[best_in_bin] / n

        bin_results.append({
            'bin': bin_name,
            'n': n,
            'cnn_mae': cnn_mae,
            'hgb_mae': hgb_mae,
            'lstm_mae': lstm_mae,
            'best_model': best_in_bin,
            'best_pct': best_pct,
            'cnn_wins': wins['cnn'],
            'hgb_wins': wins['hgb'],
            'lstm_wins': wins['lstm'],
        })

        logger.info(f"\n{bin_name} (n={n}):")
        logger.info(f"  CNN MAE: {cnn_mae:.2f}, HGB MAE: {hgb_mae:.2f}, LSTM MAE: {lstm_mae:.2f}")
        logger.info(f"  Best model: {best_in_bin} wins {best_pct:.1f}% of the time")
        for m in ['cnn', 'hgb', 'lstm']:
            logger.info(f"    {m}: {wins[m]} wins ({100*wins[m]/n:.1f}%)")

    # LSTM outlier analysis
    logger.info("")
    logger.info("=" * 70)
    logger.info("LSTM OUTLIER ANALYSIS")
    logger.info("=" * 70)

    outlier_counts = defaultdict(int)
    for d in all_data:
        outlier_counts[d['outlier_model']] += 1

    logger.info("Which model is most often the outlier (furthest from median):")
    for model in ['cnn', 'hgb', 'lstm']:
        logger.info(f"  {model}: {outlier_counts[model]} ({100*outlier_counts[model]/total:.1f}%)")

    # When LSTM is outlier, is it correct or wrong?
    lstm_outlier_data = [d for d in all_data if d['outlier_model'] == 'lstm']
    if lstm_outlier_data:
        lstm_outlier_correct = sum(1 for d in lstm_outlier_data if d['best_model'] == 'lstm')
        lstm_outlier_wrong = len(lstm_outlier_data) - lstm_outlier_correct

        logger.info("")
        logger.info("When LSTM is the outlier:")
        logger.info(f"  Correct (LSTM wins): {lstm_outlier_correct} ({100*lstm_outlier_correct/len(lstm_outlier_data):.1f}%)")
        logger.info(f"  Wrong (another model wins): {lstm_outlier_wrong} ({100*lstm_outlier_wrong/len(lstm_outlier_data):.1f}%)")

        # Mean LSTM error when it's the outlier
        lstm_outlier_errors = [d['lstm_error'] for d in lstm_outlier_data]
        other_errors = [(d['cnn_error'] + d['hgb_error']) / 2 for d in lstm_outlier_data]
        logger.info(f"  LSTM MAE when outlier: {np.mean(lstm_outlier_errors):.2f}")
        logger.info(f"  CNN+HGB avg MAE when LSTM is outlier: {np.mean(other_errors):.2f}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Plot 1: Model wins by disagreement level
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bin_results))
    width = 0.25

    cnn_wins = [r['cnn_wins'] for r in bin_results]
    hgb_wins = [r['hgb_wins'] for r in bin_results]
    lstm_wins = [r['lstm_wins'] for r in bin_results]

    ax.bar(x - width, cnn_wins, width, label='CNN', color='blue', alpha=0.7)
    ax.bar(x, hgb_wins, width, label='HGB', color='green', alpha=0.7)
    ax.bar(x + width, lstm_wins, width, label='LSTM', color='orange', alpha=0.7)

    ax.set_xlabel('Disagreement Level (std)')
    ax.set_ylabel('Number of Wins')
    ax.set_title('Model Wins by Disagreement Level')
    ax.set_xticks(x)
    ax.set_xticklabels([r['bin'] for r in bin_results])
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output / 'wins_by_disagreement.png', dpi=150)
    plt.close()

    # Plot 2: MAE by model and disagreement level
    fig, ax = plt.subplots(figsize=(10, 6))

    cnn_maes = [r['cnn_mae'] for r in bin_results]
    hgb_maes = [r['hgb_mae'] for r in bin_results]
    lstm_maes = [r['lstm_mae'] for r in bin_results]

    ax.bar(x - width, cnn_maes, width, label='CNN', color='blue', alpha=0.7)
    ax.bar(x, hgb_maes, width, label='HGB', color='green', alpha=0.7)
    ax.bar(x + width, lstm_maes, width, label='LSTM', color='orange', alpha=0.7)

    ax.set_xlabel('Disagreement Level (std)')
    ax.set_ylabel('MAE (frames)')
    ax.set_title('Model Error by Disagreement Level')
    ax.set_xticks(x)
    ax.set_xticklabels([r['bin'] for r in bin_results])
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output / 'mae_by_disagreement.png', dpi=150)
    plt.close()

    # Plot 3: LSTM as outlier - correct vs wrong
    if lstm_outlier_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [lstm_outlier_correct, lstm_outlier_wrong]
        labels = ['LSTM Correct', 'LSTM Wrong']
        colors = ['green', 'red']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('When LSTM is Outlier: Is It Correct?')
        plt.tight_layout()
        plt.savefig(args.output / 'lstm_outlier_accuracy.png', dpi=150)
        plt.close()

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# B3: Model Agreement Analysis\n\n")
        f.write("## Objective\n")
        f.write("When models disagree, which one is correct? Understand model behavior\n")
        f.write("under different disagreement conditions.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Total samples: {len(all_data)} (5-fold CV x {len(args.seeds)} seeds)\n")
        f.write(f"- Models: CNN + HGB + LSTM\n\n")

        f.write("## Overall Model Accuracy\n\n")
        f.write("| Model | MAE | Median Error | Win Rate |\n")
        f.write("|-------|-----|--------------|----------|\n")
        f.write(f"| CNN | {np.mean(cnn_errors):.2f} | {np.median(cnn_errors):.1f} | {100*best_counts['cnn']/total:.1f}% |\n")
        f.write(f"| HGB | {np.mean(hgb_errors):.2f} | {np.median(hgb_errors):.1f} | {100*best_counts['hgb']/total:.1f}% |\n")
        f.write(f"| LSTM | {np.mean(lstm_errors):.2f} | {np.median(lstm_errors):.1f} | {100*best_counts['lstm']/total:.1f}% |\n\n")

        f.write("## Accuracy by Disagreement Level\n\n")
        f.write("| Disagreement | N | CNN MAE | HGB MAE | LSTM MAE | Best Model |\n")
        f.write("|--------------|---|---------|---------|----------|------------|\n")
        for r in bin_results:
            f.write(f"| {r['bin']} | {r['n']} | {r['cnn_mae']:.2f} | {r['hgb_mae']:.2f} | {r['lstm_mae']:.2f} | {r['best_model']} ({r['best_pct']:.0f}%) |\n")

        f.write("\n## LSTM Outlier Analysis\n\n")
        f.write("| Model | Times as Outlier | Percentage |\n")
        f.write("|-------|------------------|------------|\n")
        for model in ['cnn', 'hgb', 'lstm']:
            f.write(f"| {model} | {outlier_counts[model]} | {100*outlier_counts[model]/total:.1f}% |\n")

        if lstm_outlier_data:
            f.write(f"\n**When LSTM is the outlier ({len(lstm_outlier_data)} cases):**\n")
            f.write(f"- LSTM is correct: {lstm_outlier_correct} ({100*lstm_outlier_correct/len(lstm_outlier_data):.1f}%)\n")
            f.write(f"- LSTM is wrong: {lstm_outlier_wrong} ({100*lstm_outlier_wrong/len(lstm_outlier_data):.1f}%)\n")
            f.write(f"- LSTM MAE when outlier: {np.mean(lstm_outlier_errors):.2f}\n")
            f.write(f"- CNN+HGB avg MAE when LSTM outlier: {np.mean(other_errors):.2f}\n")

        f.write("\n## Key Insights\n\n")

        # Determine primary model recommendation
        overall_best = min(['cnn', 'hgb', 'lstm'], key=lambda m: np.mean([d[f'{m}_error'] for d in all_data]))
        f.write(f"1. **Overall best model: {overall_best.upper()}** with lowest MAE\n")

        # High disagreement behavior
        high_dis_data = [r for r in bin_results if 'High' in r['bin']]
        if high_dis_data:
            high_best = high_dis_data[0]['best_model']
            f.write(f"2. **At high disagreement: {high_best.upper()}** is most reliable\n")

        # LSTM outlier conclusion
        if lstm_outlier_data:
            if lstm_outlier_wrong > lstm_outlier_correct:
                f.write(f"3. **LSTM outlier usually wrong** ({100*lstm_outlier_wrong/len(lstm_outlier_data):.0f}%) - using `std` metric helps\n")
            else:
                f.write(f"3. **LSTM outlier often correct** ({100*lstm_outlier_correct/len(lstm_outlier_data):.0f}%) - LSTM provides valuable diversity\n")

        f.write("\n## Recommendations\n\n")
        f.write("- Continue using `std` metric for 3-model (robust to outliers)\n")
        if overall_best == 'cnn':
            f.write("- CNN is a good choice for `primary_model` - most accurate overall\n")
        else:
            f.write(f"- Consider switching `primary_model` to {overall_best.upper()} for better accuracy\n")

        f.write("\n## Plots\n")
        f.write("- `wins_by_disagreement.png`: Which model wins at each disagreement level\n")
        f.write("- `mae_by_disagreement.png`: Model error at each disagreement level\n")
        f.write("- `lstm_outlier_accuracy.png`: When LSTM is outlier, is it correct?\n")

    logger.info(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
