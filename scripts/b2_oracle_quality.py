#!/usr/bin/env python3
"""
B2: Oracle Study - Perfect Quality Prediction.

Determine the ceiling of quality-based selection by using true_quality
instead of predicted_quality. This answers: "How good could we be if
we had perfect quality predictions?"

Usage:
    uv run python scripts/b2_oracle_quality.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boom_detection.combine import ThresholdCombiner, QualityGatedCombiner, disagreement, ModelPrediction
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, SelectivePrediction
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def compute_oracle_acceptance(
    sp: SelectivePrediction,
    true_quality: float,
    combiner: ThresholdCombiner | QualityGatedCombiner,
) -> tuple[bool, float]:
    """Recompute acceptance using true quality instead of predicted quality."""
    if isinstance(combiner, QualityGatedCombiner):
        # Simple threshold on true quality
        oracle_score = true_quality
        oracle_accepted = true_quality >= combiner.threshold
    else:
        # ThresholdCombiner - combine agreement with true quality
        model_preds = [ModelPrediction(m, f) for m, f in sp.model_predictions.items()]
        dis = disagreement(model_preds, metric=combiner.disagreement_metric)

        # Compute agreement score using combiner's transform
        if combiner.agreement_transform == 'sqrt':
            agreement = 1.0 / (1.0 + np.sqrt(dis / combiner.disagreement_scale))
        elif combiner.agreement_transform == 'sigmoid':
            agreement = 1.0 / (1.0 + np.exp((dis - combiner.disagreement_scale) / 3.0))
        elif combiner.agreement_transform == 'linear':
            agreement = max(0.0, 1.0 - dis / combiner.disagreement_scale)
        else:
            agreement = 1.0 / (1.0 + dis / combiner.disagreement_scale)

        # Combine with oracle quality
        oracle_score = agreement * true_quality
        oracle_accepted = oracle_score >= combiner.threshold

    return oracle_accepted, oracle_score


def main():
    parser = argparse.ArgumentParser(description='B2: Oracle Quality Study')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/b2_oracle_quality'),
                        help='Output directory')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("b2_oracle_quality")

    logger.info("=" * 70)
    logger.info("B2: Oracle Quality Study")
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

    # Best combiner config
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

                # Compute error
                if sp.boom_frame is not None:
                    pred_boom = sp.boom_frame
                else:
                    pred_boom = int(np.median(list(sp.model_predictions.values())))

                error = abs(pred_boom - true_boom)

                # Compute oracle acceptance
                oracle_accepted, oracle_score = compute_oracle_acceptance(sp, true_qual, combiner)

                all_data.append({
                    'seed': seed,
                    'fold': fold_idx,
                    'true_boom': true_boom,
                    'pred_boom': pred_boom,
                    'error': error,
                    'true_quality': true_qual,
                    'pred_quality': sp.predicted_quality,
                    'pred_accepted': sp.accepted,
                    'pred_score': sp.accept_score,
                    'oracle_accepted': oracle_accepted,
                    'oracle_score': oracle_score,
                })

            logger.info(f"  Seed {seed}, Fold {fold_idx+1}/5: {len(predictions)} samples")

    # Convert to arrays for analysis
    errors = np.array([d['error'] for d in all_data])
    true_quals = np.array([d['true_quality'] for d in all_data])
    pred_quals = np.array([d['pred_quality'] for d in all_data])
    pred_accepted = np.array([d['pred_accepted'] for d in all_data])
    oracle_accepted = np.array([d['oracle_accepted'] for d in all_data])
    pred_scores = np.array([d['pred_score'] for d in all_data])
    oracle_scores = np.array([d['oracle_score'] for d in all_data])

    logger.info("")
    logger.info("=" * 70)
    logger.info("ORACLE VS PREDICTED COMPARISON")
    logger.info("=" * 70)

    # Predicted quality results
    pred_mae = np.mean(errors[pred_accepted]) if pred_accepted.sum() > 0 else float('nan')
    pred_coverage = pred_accepted.mean()

    # Oracle quality results
    oracle_mae = np.mean(errors[oracle_accepted]) if oracle_accepted.sum() > 0 else float('nan')
    oracle_coverage = oracle_accepted.mean()

    logger.info(f"Predicted quality: MAE {pred_mae:.2f} at {pred_coverage:.1%} coverage")
    logger.info(f"Oracle quality:    MAE {oracle_mae:.2f} at {oracle_coverage:.1%} coverage")
    logger.info(f"Oracle improvement: {pred_mae - oracle_mae:+.2f} frames")

    # Risk-coverage curves
    # For various threshold levels, compute MAE and coverage
    thresholds = np.linspace(0.0, 1.0, 101)

    pred_curve = []
    oracle_curve = []

    for thresh in thresholds:
        # Predicted quality curve
        mask = pred_scores >= thresh
        if mask.sum() > 0:
            pred_curve.append({'threshold': thresh, 'mae': np.mean(errors[mask]), 'coverage': mask.mean()})

        # Oracle quality curve
        mask = oracle_scores >= thresh
        if mask.sum() > 0:
            oracle_curve.append({'threshold': thresh, 'mae': np.mean(errors[mask]), 'coverage': mask.mean()})

    # Find best MAE at specific coverage levels
    logger.info("")
    logger.info("=" * 70)
    logger.info("BEST MAE AT TARGET COVERAGE LEVELS")
    logger.info("=" * 70)

    target_coverages = [0.10, 0.15, 0.20, 0.25, 0.30]
    results_table = []

    for target_cov in target_coverages:
        # Find threshold that gives closest coverage
        best_pred = None
        best_oracle = None

        for p in pred_curve:
            if best_pred is None or abs(p['coverage'] - target_cov) < abs(best_pred['coverage'] - target_cov):
                best_pred = p

        for o in oracle_curve:
            if best_oracle is None or abs(o['coverage'] - target_cov) < abs(best_oracle['coverage'] - target_cov):
                best_oracle = o

        if best_pred and best_oracle:
            improvement = best_pred['mae'] - best_oracle['mae']
            results_table.append({
                'target': target_cov,
                'pred_mae': best_pred['mae'],
                'pred_cov': best_pred['coverage'],
                'oracle_mae': best_oracle['mae'],
                'oracle_cov': best_oracle['coverage'],
                'improvement': improvement,
            })
            logger.info(f"~{target_cov:.0%} coverage:")
            logger.info(f"  Predicted: MAE {best_pred['mae']:.2f} at {best_pred['coverage']:.1%}")
            logger.info(f"  Oracle:    MAE {best_oracle['mae']:.2f} at {best_oracle['coverage']:.1%}")
            logger.info(f"  Room for improvement: {improvement:.2f} frames")

    # Test QualityGatedCombiner with oracle
    logger.info("")
    logger.info("=" * 70)
    logger.info("QUALITY-GATED COMBINER: ORACLE VS PREDICTED")
    logger.info("=" * 70)

    qg_results = []
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        # Predicted
        pred_mask = pred_quals >= thresh
        pred_mae_qg = np.mean(errors[pred_mask]) if pred_mask.sum() > 0 else float('nan')
        pred_cov_qg = pred_mask.mean()

        # Oracle
        oracle_mask = true_quals >= thresh
        oracle_mae_qg = np.mean(errors[oracle_mask]) if oracle_mask.sum() > 0 else float('nan')
        oracle_cov_qg = oracle_mask.mean()

        qg_results.append({
            'threshold': thresh,
            'pred_mae': pred_mae_qg,
            'pred_cov': pred_cov_qg,
            'oracle_mae': oracle_mae_qg,
            'oracle_cov': oracle_cov_qg,
        })

        logger.info(f"threshold={thresh:.2f}:")
        logger.info(f"  Predicted: MAE {pred_mae_qg:.2f} at {pred_cov_qg:.1%}")
        logger.info(f"  Oracle:    MAE {oracle_mae_qg:.2f} at {oracle_cov_qg:.1%}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Plot 1: Risk-Coverage Curves
    fig, ax = plt.subplots(figsize=(10, 6))

    pred_coverages = [p['coverage'] for p in pred_curve]
    pred_maes = [p['mae'] for p in pred_curve]
    oracle_coverages = [o['coverage'] for o in oracle_curve]
    oracle_maes = [o['mae'] for o in oracle_curve]

    ax.plot(pred_coverages, pred_maes, 'b-', linewidth=2, label='Predicted Quality')
    ax.plot(oracle_coverages, oracle_maes, 'g-', linewidth=2, label='Oracle Quality')

    # Mark baseline point
    ax.scatter([0.109], [2.97], c='red', s=100, zorder=5, label='Documented Baseline')

    ax.set_xlabel('Coverage')
    ax.set_ylabel('MAE (frames)')
    ax.set_title('Risk-Coverage Curves: Oracle vs Predicted Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 20)
    plt.tight_layout()
    plt.savefig(args.output / 'risk_coverage_curves.png', dpi=150)
    plt.close()

    # Plot 2: Oracle improvement by coverage
    fig, ax = plt.subplots(figsize=(8, 6))

    targets = [r['target'] for r in results_table]
    improvements = [r['improvement'] for r in results_table]

    ax.bar(range(len(targets)), improvements, color='green', alpha=0.7)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"{t:.0%}" for t in targets])
    ax.set_xlabel('Target Coverage')
    ax.set_ylabel('MAE Improvement (frames)')
    ax.set_title('Oracle Improvement Over Predicted Quality')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.output / 'oracle_improvement.png', dpi=150)
    plt.close()

    # Plot 3: Predicted vs True Quality scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue' if a else 'red' for a in pred_accepted]
    ax.scatter(true_quals, pred_quals, c=colors, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('True Quality')
    ax.set_ylabel('Predicted Quality')
    ax.set_title('Predicted vs True Quality')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Accepted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Rejected'),
        plt.Line2D([0], [0], linestyle='--', color='black', label='Perfect prediction'),
    ])
    plt.tight_layout()
    plt.savefig(args.output / 'quality_prediction.png', dpi=150)
    plt.close()

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# B2: Oracle Quality Study\n\n")
        f.write("## Objective\n")
        f.write("Determine the ceiling of quality-based selection by using true_quality\n")
        f.write("instead of predicted_quality. This shows how much better we could do\n")
        f.write("with perfect quality predictions.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Total samples: {len(all_data)} (5-fold CV x {len(args.seeds)} seeds)\n")
        f.write(f"- Models: CNN + HGB + LSTM (3-model pipeline)\n")
        f.write(f"- Combiner: std/s=15/t=0.70\n\n")

        f.write("## Key Results\n\n")
        f.write("### Current vs Oracle (at threshold=0.70)\n\n")
        f.write("| Quality Source | MAE | Coverage |\n")
        f.write("|----------------|-----|----------|\n")
        f.write(f"| Predicted | {pred_mae:.2f} | {pred_coverage:.1%} |\n")
        f.write(f"| Oracle | {oracle_mae:.2f} | {oracle_coverage:.1%} |\n")
        f.write(f"| **Improvement** | **{pred_mae - oracle_mae:+.2f}** | |\n\n")

        f.write("### Best MAE at Target Coverage Levels\n\n")
        f.write("| Target | Predicted MAE | Pred Cov | Oracle MAE | Oracle Cov | Room to Improve |\n")
        f.write("|--------|---------------|----------|------------|------------|----------------|\n")
        for r in results_table:
            f.write(f"| ~{r['target']:.0%} | {r['pred_mae']:.2f} | {r['pred_cov']:.1%} | {r['oracle_mae']:.2f} | {r['oracle_cov']:.1%} | {r['improvement']:.2f} |\n")

        f.write("\n### Quality-Gated Combiner Comparison\n\n")
        f.write("| Threshold | Pred MAE | Pred Cov | Oracle MAE | Oracle Cov |\n")
        f.write("|-----------|----------|----------|------------|------------|\n")
        for r in qg_results:
            f.write(f"| {r['threshold']:.2f} | {r['pred_mae']:.2f} | {r['pred_cov']:.1%} | {r['oracle_mae']:.2f} | {r['oracle_cov']:.1%} |\n")

        f.write("\n## Analysis\n\n")

        avg_improvement = np.mean([r['improvement'] for r in results_table])
        if avg_improvement > 1.0:
            f.write(f"**Significant room for improvement ({avg_improvement:.1f} frames on average).**\n\n")
            f.write("The quality prediction model is a bottleneck. Improving quality prediction\n")
            f.write("could substantially reduce MAE at similar coverage levels.\n\n")
            f.write("**Recommendations:**\n")
            f.write("- Explore better quality prediction models (C1)\n")
            f.write("- Consider ensemble quality predictions\n")
            f.write("- Investigate what makes quality hard to predict\n")
        elif avg_improvement > 0.3:
            f.write(f"**Moderate room for improvement ({avg_improvement:.1f} frames on average).**\n\n")
            f.write("Quality prediction is decent but not optimal. Some gains possible\n")
            f.write("from better quality models.\n\n")
            f.write("**Recommendations:**\n")
            f.write("- Quality model improvements may help marginally\n")
            f.write("- Focus more on model agreement signals\n")
        else:
            f.write(f"**Minimal room for improvement ({avg_improvement:.1f} frames on average).**\n\n")
            f.write("Quality prediction is already near-optimal. Further improvements\n")
            f.write("should focus on the frame prediction models, not quality prediction.\n\n")
            f.write("**Recommendations:**\n")
            f.write("- Quality prediction is not the bottleneck\n")
            f.write("- Focus on improving frame models or combiner logic\n")

        f.write("\n## Plots\n")
        f.write("- `risk_coverage_curves.png`: MAE vs coverage for predicted and oracle quality\n")
        f.write("- `oracle_improvement.png`: Room for improvement by coverage level\n")
        f.write("- `quality_prediction.png`: Scatter of predicted vs true quality\n")

    logger.info(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
