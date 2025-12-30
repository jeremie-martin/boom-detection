#!/usr/bin/env python3
"""
D2: Bias Correction Post-Processing.

Test if correcting for the systematic early bias (-3.4 frames) improves performance.
From B4, predictions are systematically early. Adding a bias offset may help.

Usage:
    uv run python scripts/d2_bias_correction.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from boom_detection.combine import ThresholdCombiner, ModelPrediction, median_frame, disagreement
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, SelectivePrediction, compute_selective_metrics
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='D2: Bias Correction Post-Processing')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/d2_bias_correction'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("d2_bias_correction")

    logger.info("=" * 70)
    logger.info("D2: Bias Correction Post-Processing")
    logger.info("=" * 70)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    cache = FeatureCache(config=PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = {a.id: a.boom_frame for a in dataset.annotations}
    boom_qualities = {a.id: a.boom_quality for a in dataset.annotations}

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

    # Bias offsets to test (positive = shift predictions later)
    # From B4: bias is -3.4 frames (predictions are early)
    # So we test adding 2-5 frames to shift later
    offsets = [0, 1, 2, 3, 4, 5]

    # Standard combiner
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    all_results = {}

    for offset in offsets:
        config_name = f"offset={offset}"

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {config_name}")
        logger.info("=" * 70)

        seed_metrics = []

        for seed in args.seeds:
            logger.info(f"  Seed {seed}...")

            outer_kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_predictions = []
            fold_true_booms = []
            fold_true_qualities = []

            for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(evaluator.sim_ids)):
                train_sims = [evaluator.sim_ids[i] for i in train_idx]
                test_sims = [evaluator.sim_ids[i] for i in test_idx]

                # Train pipeline
                pipeline = BoomDetectionPipeline(
                    frame_models=('cnn', 'hgb', 'lstm'),
                    combiner=combiner,
                    seed=seed * 1000 + fold_idx,
                )
                train_booms = [boom_frames[s] for s in train_sims]
                train_qualities = [boom_qualities[s] for s in train_sims]
                pipeline.fit(train_sims, train_booms, train_qualities, cache)

                # Evaluate on test set with bias correction
                for sim_id in test_sims:
                    true_boom = boom_frames[sim_id]
                    true_quality = boom_qualities[sim_id]
                    features = cache[sim_id]
                    pred = pipeline.predict_one(features)

                    # Apply bias correction
                    if pred.accepted and pred.boom_frame is not None:
                        corrected_frame = pred.boom_frame + offset
                        # Clip to valid range
                        n_frames = features.shape[0]
                        corrected_frame = max(0, min(corrected_frame, n_frames - 1))

                        fold_predictions.append(SelectivePrediction(
                            boom_frame=corrected_frame,
                            accepted=True,
                            accept_score=pred.accept_score,
                            predicted_quality=pred.predicted_quality,
                            model_predictions={k: v + offset for k, v in pred.model_predictions.items()},
                            disagreement=pred.disagreement,
                        ))
                    else:
                        fold_predictions.append(pred)

                    fold_true_booms.append(true_boom)
                    fold_true_qualities.append(true_quality)

            # Compute metrics for this seed
            metrics = compute_selective_metrics(
                fold_predictions,
                np.array(fold_true_booms),
                np.array(fold_true_qualities),
            )
            seed_metrics.append(metrics)
            logger.info(f"    MAE: {metrics['selective_mae']:.2f}, Coverage: {metrics['coverage']:.1%}")

        # Aggregate metrics across seeds
        mae_values = [m['selective_mae'] for m in seed_metrics if not np.isnan(m['selective_mae'])]
        coverage_values = [m['coverage'] for m in seed_metrics]

        if mae_values:
            mean_mae = np.mean(mae_values)
            std_mae = np.std(mae_values)
        else:
            mean_mae = float('nan')
            std_mae = 0

        mean_coverage = np.mean(coverage_values)
        std_coverage = np.std(coverage_values)

        all_results[config_name] = {
            'offset': offset,
            'mae': mean_mae,
            'mae_std': std_mae,
            'coverage': mean_coverage,
            'coverage_std': std_coverage,
        }

        logger.info(f"  Overall: MAE {mean_mae:.2f} ± {std_mae:.2f}, Coverage {mean_coverage:.1%}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Config':<15} {'MAE':<15} {'Coverage':<12}")
    logger.info("-" * 42)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{name:<15} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['coverage']:.1%}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("offset,mae,mae_std,coverage,coverage_std\n")
        for name, r in all_results.items():
            f.write(f"{r['offset']},{r['mae']:.2f},{r['mae_std']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# D2: Bias Correction Post-Processing\n\n")
        f.write("## Objective\n")
        f.write("Test if correcting for systematic early bias improves performance.\n")
        f.write("From B4, predictions are systematically early by ~3.4 frames.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Results\n\n")
        f.write("| Offset | MAE | Coverage |\n")
        f.write("|--------|-----|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['offset']):
            f.write(f"| {r['offset']} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['coverage']:.1%} |\n")

        # Find best
        best = min(all_results.values(), key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))
        baseline = all_results['offset=0']

        f.write(f"\n## Best Configuration\n\n")
        f.write(f"**Offset = {best['offset']}**: MAE {best['mae']:.2f} ± {best['mae_std']:.2f}\n\n")

        f.write("## Analysis\n\n")

        improvement = baseline['mae'] - best['mae']
        if best['offset'] > 0 and improvement > 0.1:
            f.write(f"**Bias correction helps!** Adding {best['offset']} frames improves MAE by {improvement:.2f}.\n\n")
            f.write("Recommendation: Implement bias correction in production pipeline.\n")
        elif improvement > 0:
            f.write(f"Marginal improvement of {improvement:.2f} frames with offset {best['offset']}.\n\n")
            f.write("Recommendation: Marginal, likely not worth the complexity.\n")
        else:
            f.write("**Bias correction does not help.** Baseline (offset=0) is optimal.\n\n")
            f.write("Recommendation: Keep using predictions without bias correction.\n")

        f.write("\n### Why Bias Correction May/May Not Help\n\n")
        f.write("1. B4 showed bias on ALL predictions, but we filter with ThresholdCombiner\n")
        f.write("2. Accepted samples (high quality, low disagreement) may have different bias\n")
        f.write("3. The combiner already selects for predictions that are likely correct\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
