#!/usr/bin/env python3
"""
D5: Stratified Analysis by Quality.

Analyze performance across different quality strata to understand:
1. How performance varies by true quality
2. Whether low-quality simulations are correctly rejected
3. If acceptance is well-calibrated to quality

Usage:
    uv run python scripts/d5_stratified_analysis.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import SelectivePrediction
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='D5: Stratified Analysis by Quality')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/d5_stratified_analysis'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("d5_stratified_analysis")

    logger.info("=" * 70)
    logger.info("D5: Stratified Analysis by Quality")
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

    logger.info(f"Evaluating on {len(sim_ids)} simulations")

    # Quality strata
    strata = [
        ('low (0-0.3)', 0.0, 0.3),
        ('med-low (0.3-0.5)', 0.3, 0.5),
        ('med-high (0.5-0.7)', 0.5, 0.7),
        ('high (0.7-1.0)', 0.7, 1.0),
    ]

    # Standard combiner
    combiner = ThresholdCombiner(
        primary_model='cnn',
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    )

    # Collect all predictions across seeds
    all_predictions = []  # list of (sim_id, true_quality, true_boom, prediction)

    for seed in args.seeds:
        logger.info(f"\nSeed {seed}...")

        outer_kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(sim_ids)):
            train_sims = [sim_ids[i] for i in train_idx]
            test_sims = [sim_ids[i] for i in test_idx]

            # Train pipeline
            pipeline = BoomDetectionPipeline(
                frame_models=('cnn', 'hgb', 'lstm'),
                combiner=combiner,
                seed=seed * 1000 + fold_idx,
            )
            train_booms = [boom_frames[s] for s in train_sims]
            train_qualities = [boom_qualities[s] for s in train_sims]
            pipeline.fit(train_sims, train_booms, train_qualities, cache)

            # Evaluate on test set
            for sim_id in test_sims:
                true_boom = boom_frames[sim_id]
                true_quality = boom_qualities[sim_id]
                features = cache[sim_id]
                pred = pipeline.predict_one(features)

                all_predictions.append({
                    'sim_id': sim_id,
                    'seed': seed,
                    'true_quality': true_quality,
                    'true_boom': true_boom,
                    'pred_boom': pred.boom_frame,
                    'accepted': pred.accepted,
                    'pred_quality': pred.predicted_quality,
                    'disagreement': pred.disagreement,
                })

    logger.info(f"\nTotal predictions: {len(all_predictions)}")

    # Analyze by strata
    stratum_results = {}

    for stratum_name, q_min, q_max in strata:
        stratum_preds = [p for p in all_predictions
                        if q_min <= p['true_quality'] < q_max]

        n_total = len(stratum_preds)
        n_accepted = sum(1 for p in stratum_preds if p['accepted'])

        # Calculate errors for accepted predictions
        errors = []
        for p in stratum_preds:
            if p['accepted'] and p['pred_boom'] is not None:
                error = abs(p['pred_boom'] - p['true_boom'])
                errors.append(error)

        if errors:
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        else:
            mae = float('nan')
            rmse = float('nan')

        coverage = n_accepted / n_total if n_total > 0 else 0
        rejection_rate = 1 - coverage

        stratum_results[stratum_name] = {
            'q_min': q_min,
            'q_max': q_max,
            'n_total': n_total,
            'n_accepted': n_accepted,
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'rejection_rate': rejection_rate,
        }

        logger.info(f"\n{stratum_name}:")
        logger.info(f"  Simulations: {n_total}")
        logger.info(f"  Accepted: {n_accepted} ({coverage:.1%})")
        logger.info(f"  MAE: {mae:.2f}" if not np.isnan(mae) else "  MAE: N/A")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Stratum':<20} {'N':<6} {'Accepted':<10} {'Coverage':<12} {'MAE':<10}")
    logger.info("-" * 58)
    for name, r in stratum_results.items():
        mae_str = f"{r['mae']:.2f}" if not np.isnan(r['mae']) else "N/A"
        logger.info(f"{name:<20} {r['n_total']:<6} {r['n_accepted']:<10} {r['coverage']:.1%}        {mae_str}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("stratum,q_min,q_max,n_total,n_accepted,mae,rmse,coverage,rejection_rate\n")
        for name, r in stratum_results.items():
            f.write(f"{name},{r['q_min']},{r['q_max']},{r['n_total']},{r['n_accepted']},")
            f.write(f"{r['mae']:.2f},{r['rmse']:.2f},{r['coverage']:.3f},{r['rejection_rate']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write detailed predictions
    pred_csv_path = args.output / 'predictions.csv'
    with open(pred_csv_path, 'w') as f:
        f.write("sim_id,seed,true_quality,true_boom,pred_boom,accepted,pred_quality,disagreement,error\n")
        for p in all_predictions:
            error = abs(p['pred_boom'] - p['true_boom']) if p['pred_boom'] is not None else ''
            pred_boom = p['pred_boom'] if p['pred_boom'] is not None else ''
            f.write(f"{p['sim_id']},{p['seed']},{p['true_quality']:.3f},{p['true_boom']},")
            f.write(f"{pred_boom},{p['accepted']},{p['pred_quality']:.3f},{p['disagreement']:.3f},{error}\n")
    logger.info(f"Detailed predictions saved to: {pred_csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# D5: Stratified Analysis by Quality\n\n")
        f.write("## Objective\n")
        f.write("Analyze performance across different quality strata to understand:\n")
        f.write("1. How performance varies by true quality\n")
        f.write("2. Whether low-quality simulations are correctly rejected\n")
        f.write("3. If acceptance is well-calibrated to quality\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(sim_ids)} simulations\n")
        f.write("- 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n")
        f.write("- Combiner: ThresholdCombiner (std/s=15/t=0.70)\n\n")

        f.write("## Quality Distribution\n\n")
        total_sims = len(sim_ids)
        for name, r in stratum_results.items():
            n_unique = r['n_total'] // len(args.seeds)  # Approximate unique sims
            pct = n_unique / total_sims * 100
            f.write(f"- **{name}**: {n_unique} simulations ({pct:.0f}%)\n")

        f.write("\n## Results by Stratum\n\n")
        f.write("| Quality Stratum | N | Accepted | Coverage | MAE |\n")
        f.write("|-----------------|---|----------|----------|-----|\n")
        for name, r in stratum_results.items():
            mae_str = f"{r['mae']:.2f}" if not np.isnan(r['mae']) else "N/A"
            f.write(f"| {name} | {r['n_total']} | {r['n_accepted']} | {r['coverage']:.1%} | {mae_str} |\n")

        f.write("\n## Analysis\n\n")

        # Check if rejection rate increases with lower quality
        low_reject = stratum_results['low (0-0.3)']['rejection_rate']
        high_reject = stratum_results['high (0.7-1.0)']['rejection_rate']

        f.write("### Rejection Behavior\n\n")
        if low_reject > high_reject + 0.1:
            f.write("**Good calibration:** Low-quality simulations are rejected at higher rates.\n")
            f.write(f"- Low quality rejection: {low_reject:.1%}\n")
            f.write(f"- High quality rejection: {high_reject:.1%}\n\n")
        else:
            f.write("**Concerning:** Low-quality simulations are not preferentially rejected.\n")
            f.write(f"- Low quality rejection: {low_reject:.1%}\n")
            f.write(f"- High quality rejection: {high_reject:.1%}\n\n")
            f.write("The combiner may not be effectively filtering by quality.\n")

        # Check if MAE is lower for high quality
        f.write("### Error by Quality\n\n")
        high_mae = stratum_results['high (0.7-1.0)']['mae']
        med_high_mae = stratum_results['med-high (0.5-0.7)']['mae']

        if not np.isnan(high_mae):
            if high_mae < 3.0:
                f.write(f"**High quality predictions are accurate:** MAE = {high_mae:.2f}\n")
            else:
                f.write(f"**Warning:** High quality MAE = {high_mae:.2f} is relatively high.\n")

        # Check where most accepted predictions come from
        f.write("\n### Source of Accepted Predictions\n\n")
        total_accepted = sum(r['n_accepted'] for r in stratum_results.values())
        for name, r in stratum_results.items():
            if total_accepted > 0:
                pct = r['n_accepted'] / total_accepted * 100
                f.write(f"- {name}: {r['n_accepted']} ({pct:.0f}% of accepted)\n")

        f.write("\n### Recommendations\n\n")
        if low_reject > 0.9 and high_reject < 0.5:
            f.write("1. **Current system is working well** - correctly rejecting low quality\n")
            f.write("2. Most accepted predictions come from high-quality simulations\n")
        else:
            f.write("1. Consider tuning threshold to better reject low-quality simulations\n")
            f.write("2. Quality prediction may need improvement\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
