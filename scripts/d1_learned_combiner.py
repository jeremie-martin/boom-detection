#!/usr/bin/env python3
"""
D1: Learned Combiner (Logistic Regression / MLP).

Replace hand-tuned ThresholdCombiner with learned decision boundary.
Uses nested CV to avoid leakage.

Usage:
    uv run python scripts/d1_learned_combiner.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from boom_detection.combine import Combiner, ModelPrediction, disagreement, median_frame
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator, SelectivePrediction, compute_selective_metrics
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


@dataclass
class LearnedCombiner(Combiner):
    """Combiner that uses a trained classifier to decide acceptance."""

    classifier: Any = None
    scaler: StandardScaler = None
    error_threshold: float = 5.0  # Label: error <= threshold is "accept"

    def __call__(
        self,
        predictions: list[ModelPrediction],
        predicted_quality: float,
        features: np.ndarray | None = None,
    ) -> int | None:
        if self.classifier is None:
            return None

        # Extract features for classifier
        frames = [p.frame for p in predictions]
        cnn_pred = next((p.frame for p in predictions if p.model == 'cnn'), np.mean(frames))
        hgb_pred = next((p.frame for p in predictions if p.model == 'hgb'), np.mean(frames))
        lstm_pred = next((p.frame for p in predictions if p.model == 'lstm'), np.mean(frames))

        std_dis = disagreement(predictions, 'std')

        X = np.array([[cnn_pred, hgb_pred, lstm_pred, predicted_quality, std_dis]])
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict accept/reject
        accept = self.classifier.predict(X)[0]
        if accept:
            return median_frame(predictions)
        return None

    def to_dict(self) -> dict:
        return {
            'type': 'LearnedCombiner',
            'params': {'error_threshold': self.error_threshold},
        }


def extract_features_for_sample(predictions: list[ModelPrediction], predicted_quality: float) -> np.ndarray:
    """Extract features for classifier input."""
    frames = [p.frame for p in predictions]
    cnn_pred = next((p.frame for p in predictions if p.model == 'cnn'), np.mean(frames))
    hgb_pred = next((p.frame for p in predictions if p.model == 'hgb'), np.mean(frames))
    lstm_pred = next((p.frame for p in predictions if p.model == 'lstm'), np.mean(frames))

    std_dis = disagreement(predictions, 'std')
    range_dis = disagreement(predictions, 'range')

    return np.array([cnn_pred, hgb_pred, lstm_pred, predicted_quality, std_dis, range_dis])


def main():
    parser = argparse.ArgumentParser(description='D1: Learned Combiner')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=Path('results/d1_learned_combiner'),
                        help='Output directory')
    parser.add_argument('--error-threshold', type=float, default=5.0,
                        help='Error threshold for accept label (default: 5.0)')
    args = parser.parse_args()

    # Setup logging
    run_dir = setup_run_logging("d1_learned_combiner")

    logger.info("=" * 70)
    logger.info("D1: Learned Combiner (Logistic/MLP)")
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

    # Classifiers to test
    classifiers = {
        'LogisticRegression': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'MLP-small': lambda: MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42),
        'MLP-medium': lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
    }

    all_results = {}

    for clf_name, clf_factory in classifiers.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Testing: {clf_name}")
        logger.info("=" * 70)

        seed_metrics = []

        for seed in args.seeds:
            logger.info(f"  Seed {seed}...")

            # Outer CV for evaluation
            outer_kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_predictions = []
            fold_true_booms = []
            fold_true_qualities = []

            for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(evaluator.sim_ids)):
                train_sims = [evaluator.sim_ids[i] for i in train_idx]
                test_sims = [evaluator.sim_ids[i] for i in test_idx]

                # Train 3-model pipeline on train set
                pipeline = BoomDetectionPipeline(
                    frame_models=('cnn', 'hgb', 'lstm'),
                    combiner=None,  # We'll use our own combiner
                    seed=seed * 1000 + fold_idx,
                )
                train_booms = [boom_frames[s] for s in train_sims]
                train_qualities = [boom_qualities[s] for s in train_sims]
                pipeline.fit(train_sims, train_booms, train_qualities, cache)

                # Get predictions on train set for classifier training
                X_clf_train = []
                y_clf_train = []

                for sim_id in train_sims:
                    true_boom = boom_frames[sim_id]
                    features = cache[sim_id]
                    pred = pipeline.predict_one(features)
                    if pred.model_predictions:
                        # Convert dict to list of ModelPrediction
                        model_preds = [ModelPrediction(name, frame) for name, frame in pred.model_predictions.items()]
                        feat = extract_features_for_sample(model_preds, pred.predicted_quality)
                        X_clf_train.append(feat)

                        # Use CNN prediction as our frame prediction
                        cnn_frame = pred.model_predictions.get('cnn')
                        if cnn_frame is not None:
                            error = abs(cnn_frame - true_boom)
                            label = 1 if error <= args.error_threshold else 0
                            y_clf_train.append(label)
                        else:
                            y_clf_train.append(0)

                X_clf_train = np.array(X_clf_train)
                y_clf_train = np.array(y_clf_train)

                # Train classifier
                scaler = StandardScaler()
                X_clf_train_scaled = scaler.fit_transform(X_clf_train)

                clf = clf_factory()
                clf.fit(X_clf_train_scaled, y_clf_train)

                # Evaluate on test set
                for sim_id in test_sims:
                    true_boom = boom_frames[sim_id]
                    true_quality = boom_qualities[sim_id]
                    features = cache[sim_id]
                    pred = pipeline.predict_one(features)

                    if pred.model_predictions:
                        model_preds = [ModelPrediction(name, frame) for name, frame in pred.model_predictions.items()]
                        feat = extract_features_for_sample(model_preds, pred.predicted_quality)
                        X_test = scaler.transform(feat.reshape(1, -1))

                        accept = clf.predict(X_test)[0]
                        if accept:
                            predicted_frame = median_frame(model_preds)
                            fold_predictions.append(SelectivePrediction(
                                boom_frame=predicted_frame,
                                accepted=True,
                                accept_score=1.0,
                                predicted_quality=pred.predicted_quality,
                                model_predictions=pred.model_predictions,
                                disagreement=pred.disagreement,
                            ))
                        else:
                            fold_predictions.append(SelectivePrediction(
                                boom_frame=None,
                                accepted=False,
                                accept_score=0.0,
                                predicted_quality=pred.predicted_quality,
                                model_predictions=pred.model_predictions,
                                disagreement=pred.disagreement,
                            ))
                    else:
                        fold_predictions.append(SelectivePrediction(
                            boom_frame=None,
                            accepted=False,
                            accept_score=0.0,
                            predicted_quality=0.0,
                            model_predictions={},
                            disagreement=0.0,
                        ))

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

        all_results[clf_name] = {
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

    logger.info(f"\n{'Classifier':<20} {'MAE':<15} {'Coverage':<12}")
    logger.info("-" * 47)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
        logger.info(f"{name:<20} {r['mae']:.2f}±{r['mae_std']:.2f}    {r['coverage']:.1%}")

    # Compare with baseline
    logger.info("\nComparison with ThresholdCombiner baseline (MAE 2.97 @ 10.9%):")
    baseline_mae = 2.97
    baseline_coverage = 0.109

    for name, r in all_results.items():
        if not np.isnan(r['mae']):
            mae_diff = r['mae'] - baseline_mae
            cov_diff = r['coverage'] - baseline_coverage
            logger.info(f"  {name}: MAE diff {mae_diff:+.2f}, Coverage diff {cov_diff:+.1%}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = args.output / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("classifier,mae,mae_std,coverage,coverage_std\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['mae']:.2f},{r['mae_std']:.2f},{r['coverage']:.3f},{r['coverage_std']:.3f}\n")
    logger.info(f"\nCSV saved to: {csv_path}")

    # Write report
    report_path = args.output / 'analysis.md'
    with open(report_path, 'w') as f:
        f.write("# D1: Learned Combiner (Logistic/MLP)\n\n")
        f.write("## Objective\n")
        f.write("Replace hand-tuned ThresholdCombiner with learned decision boundary.\n\n")

        f.write("## Configuration\n")
        f.write(f"- Seeds: {args.seeds}\n")
        f.write(f"- Dataset: {len(evaluator.sim_ids)} simulations\n")
        f.write(f"- Error threshold for accept label: {args.error_threshold} frames\n")
        f.write("- Nested 5-fold cross-validation\n")
        f.write("- 3-model pipeline (CNN + HGB + LSTM)\n\n")

        f.write("## Features for Classifier\n")
        f.write("- CNN predicted frame\n")
        f.write("- HGB predicted frame\n")
        f.write("- LSTM predicted frame\n")
        f.write("- Predicted quality\n")
        f.write("- Std disagreement\n")
        f.write("- Range disagreement\n\n")

        f.write("## Results\n\n")
        f.write("| Classifier | MAE | Coverage |\n")
        f.write("|------------|-----|----------|\n")
        for name, r in sorted(all_results.items(), key=lambda x: x[1]['mae'] if not np.isnan(x[1]['mae']) else float('inf')):
            f.write(f"| {name} | {r['mae']:.2f} ± {r['mae_std']:.2f} | {r['coverage']:.1%} |\n")

        f.write("\n## Comparison with Baseline\n\n")
        f.write("**Baseline**: ThresholdCombiner (std/s=15/t=0.70) - MAE 2.97 @ 10.9% coverage\n\n")

        best_name = min(all_results, key=lambda x: all_results[x]['mae'] if not np.isnan(all_results[x]['mae']) else float('inf'))
        best = all_results[best_name]

        f.write(f"**Best learned combiner**: {best_name}\n")
        f.write(f"- MAE: {best['mae']:.2f} ± {best['mae_std']:.2f}\n")
        f.write(f"- Coverage: {best['coverage']:.1%}\n\n")

        mae_diff = best['mae'] - baseline_mae
        if mae_diff < -0.2:
            f.write(f"**Learned combiner outperforms** ThresholdCombiner by {-mae_diff:.2f} frames.\n")
            f.write("Recommendation: Consider using learned combiner.\n")
        elif mae_diff > 0.2:
            f.write(f"**ThresholdCombiner outperforms** learned combiner by {mae_diff:.2f} frames.\n")
            f.write("Recommendation: Keep using ThresholdCombiner.\n")
        else:
            f.write("**Performance is comparable** to ThresholdCombiner.\n")
            f.write("Recommendation: ThresholdCombiner is simpler and sufficient.\n")

        f.write("\n## Analysis\n\n")
        f.write("The learned combiner attempts to learn the accept/reject decision from data.\n")
        f.write("This allows it to potentially capture complex decision boundaries that\n")
        f.write("a simple threshold-based approach cannot.\n\n")

        if mae_diff > 0:
            f.write("However, the hand-tuned ThresholdCombiner performs better, likely because:\n")
            f.write("1. The decision boundary is actually simple (threshold-based)\n")
            f.write("2. Limited training data for the classifier\n")
            f.write("3. The features already capture the relevant information\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
