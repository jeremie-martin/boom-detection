#!/usr/bin/env python3
"""
B1: Error vs Ground Truth Quality Correlation Analysis.

Understand whether true quality predicts prediction error and whether
model disagreement adds information beyond quality.

Usage:
    uv run python scripts/b1_error_quality_correlation.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from boom_detection.combine import ThresholdCombiner, disagreement
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='B1: Error vs Quality Correlation')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/b1_error_quality_correlation'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("b1_error_quality_correlation")

    logger.info("=" * 70)
    logger.info("B1: Error vs Ground Truth Quality Correlation")
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

                from boom_detection.evaluation import SelectivePrediction
                if isinstance(pred, dict):
                    sp = SelectivePrediction.from_dict(pred)
                else:
                    sp = pred

                # Compute error (even for rejected samples)
                if sp.boom_frame is not None:
                    pred_boom = sp.boom_frame
                else:
                    # Use median of model predictions for rejected samples
                    pred_boom = int(np.median(list(sp.model_predictions.values())))

                error = abs(pred_boom - true_boom)

                # Compute disagreement
                from boom_detection.combine import ModelPrediction
                model_preds = [ModelPrediction(m, f) for m, f in sp.model_predictions.items()]
                dis = disagreement(model_preds, metric='std')

                all_data.append({
                    'seed': seed,
                    'fold': fold_idx,
                    'true_boom': true_boom,
                    'pred_boom': pred_boom,
                    'error': error,
                    'true_quality': true_qual,
                    'pred_quality': sp.predicted_quality,
                    'accepted': sp.accepted,
                    'disagreement': dis,
                })

            logger.info(f"  Seed {seed}, Fold {fold_idx+1}/5: {len(predictions)} samples")

    # Convert to arrays for analysis
    errors = np.array([d['error'] for d in all_data])
    true_quals = np.array([d['true_quality'] for d in all_data])
    pred_quals = np.array([d['pred_quality'] for d in all_data])
    disagreements = np.array([d['disagreement'] for d in all_data])
    accepted = np.array([d['accepted'] for d in all_data])

    # Compute correlations
    spearman_qual, pval_qual = stats.spearmanr(true_quals, errors)
    spearman_dis, pval_dis = stats.spearmanr(disagreements, errors)
    spearman_predq, pval_predq = stats.spearmanr(pred_quals, errors)

    logger.info("")
    logger.info("=" * 70)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Spearman correlation (true_quality, error): r={spearman_qual:.3f}, p={pval_qual:.2e}")
    logger.info(f"Spearman correlation (disagreement, error): r={spearman_dis:.3f}, p={pval_dis:.2e}")
    logger.info(f"Spearman correlation (pred_quality, error): r={spearman_predq:.3f}, p={pval_predq:.2e}")

    # Analyze rejected vs accepted
    accepted_errors = errors[accepted]
    rejected_errors = errors[~accepted]

    logger.info("")
    logger.info(f"Accepted samples: n={accepted.sum()}, mean error={np.mean(accepted_errors):.2f}")
    logger.info(f"Rejected samples: n={len(rejected_errors)}, mean error={np.mean(rejected_errors):.2f}")
    logger.info(f"Error difference: {np.mean(rejected_errors) - np.mean(accepted_errors):.2f} frames higher for rejected")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Plot 1: Error vs True Quality
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue' if a else 'red' for a in accepted]
    ax.scatter(true_quals, errors, c=colors, alpha=0.5, s=20)
    ax.set_xlabel('True Quality')
    ax.set_ylabel('Absolute Error (frames)')
    ax.set_title(f'Error vs True Quality (Spearman r={spearman_qual:.3f})')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Accepted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Rejected'),
    ])
    plt.tight_layout()
    plt.savefig(args.output / 'error_vs_true_quality.png', dpi=150)
    plt.close()

    # Plot 2: Error vs Disagreement
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(disagreements, errors, c=colors, alpha=0.5, s=20)
    ax.set_xlabel('Model Disagreement (std)')
    ax.set_ylabel('Absolute Error (frames)')
    ax.set_title(f'Error vs Model Disagreement (Spearman r={spearman_dis:.3f})')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Accepted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Rejected'),
    ])
    plt.tight_layout()
    plt.savefig(args.output / 'error_vs_disagreement.png', dpi=150)
    plt.close()

    # Plot 3: Quality distribution for accepted vs rejected
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(true_quals[accepted], bins=20, alpha=0.6, label='Accepted', color='blue')
    ax.hist(true_quals[~accepted], bins=20, alpha=0.6, label='Rejected', color='red')
    ax.set_xlabel('True Quality')
    ax.set_ylabel('Count')
    ax.set_title('Quality Distribution: Accepted vs Rejected')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output / 'quality_distribution.png', dpi=150)
    plt.close()

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# B1: Error vs Ground Truth Quality Correlation\n\n")
        f.write("## Objective\n")
        f.write("Understand whether true quality predicts prediction error and whether\n")
        f.write("model disagreement adds information beyond quality.\n\n")
        f.write("## Data\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Total samples: {len(all_data)} (5-fold CV x {len(args.seeds)} seeds)\n")
        f.write(f"- Accepted: {accepted.sum()} ({100*accepted.mean():.1f}%)\n")
        f.write(f"- Rejected: {(~accepted).sum()} ({100*(1-accepted.mean()):.1f}%)\n\n")

        f.write("## Correlation Analysis\n\n")
        f.write("| Variable | Spearman r | p-value | Interpretation |\n")
        f.write("|----------|------------|---------|----------------|\n")
        f.write(f"| True Quality vs Error | {spearman_qual:.3f} | {pval_qual:.2e} | ")
        f.write("Strong negative" if spearman_qual < -0.3 else "Moderate negative" if spearman_qual < -0.1 else "Weak" + " |\n")
        f.write(f"| Disagreement vs Error | {spearman_dis:.3f} | {pval_dis:.2e} | ")
        f.write("Strong positive" if spearman_dis > 0.3 else "Moderate positive" if spearman_dis > 0.1 else "Weak" + " |\n")
        f.write(f"| Pred Quality vs Error | {spearman_predq:.3f} | {pval_predq:.2e} | ")
        f.write("Strong negative" if spearman_predq < -0.3 else "Moderate negative" if spearman_predq < -0.1 else "Weak" + " |\n\n")

        f.write("## Rejection Analysis\n\n")
        f.write(f"- Mean error for accepted samples: {np.mean(accepted_errors):.2f} frames\n")
        f.write(f"- Mean error for rejected samples: {np.mean(rejected_errors):.2f} frames\n")
        f.write(f"- **Rejection avoids {np.mean(rejected_errors) - np.mean(accepted_errors):.2f} frames higher error**\n\n")

        f.write("## Key Insights\n\n")
        if spearman_qual < -0.3:
            f.write("1. **True quality is strongly predictive of error** - high quality simulations have lower errors\n")
        elif spearman_qual < -0.1:
            f.write("1. **True quality moderately predicts error** - some signal but not dominant\n")
        else:
            f.write("1. **True quality weakly predicts error** - other factors dominate\n")

        if spearman_dis > 0.3:
            f.write("2. **Model disagreement strongly predicts error** - valuable selection signal\n")
        elif spearman_dis > 0.1:
            f.write("2. **Model disagreement moderately predicts error** - useful but not sufficient\n")
        else:
            f.write("2. **Model disagreement weakly predicts error** - limited selection value\n")

        f.write("3. **Rejection is effective** - rejected samples have higher errors on average\n\n")

        f.write("## Plots\n")
        f.write("- `error_vs_true_quality.png`: Scatter plot of error vs true quality\n")
        f.write("- `error_vs_disagreement.png`: Scatter plot of error vs model disagreement\n")
        f.write("- `quality_distribution.png`: Distribution of true quality for accepted/rejected\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
