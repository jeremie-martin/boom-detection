#!/usr/bin/env python3
"""
Evaluate the impact of new coverage-based caustic features.

Tests:
1. CNN model with/without new features
2. HGB model with/without new features
3. Quality prediction with/without new features
4. Full pipeline with/without new features

Uses the unified CachedEvaluator framework for consistent evaluation.

Usage:
    uv run python scripts/evaluate_coverage_features.py data
    uv run python scripts/evaluate_coverage_features.py data --quick  # single seed
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.evaluation import CachedEvaluator, compute_all_metrics
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.frame_models import FrameLevelClassifier


@dataclass
class EvalResult:
    """Evaluation result for comparison."""
    name: str
    mae_mean: float
    mae_std: float
    extra: dict


def evaluate_cnn_hgb_from_pipeline(
    dataset,
    cache: FeatureCache,
    seeds: list[int],
) -> tuple[EvalResult, EvalResult]:
    """
    Evaluate CNN and HGB models by training pipeline and extracting predictions.

    Returns separate results for CNN and HGB.
    """
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    cnn_maes = []
    hgb_maes = []

    for seed in seeds:
        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        seed_cnn_errors = []
        seed_hgb_errors = []

        for train_idx, test_idx in kf.split(sim_ids):
            train_ids = [sim_ids[i] for i in train_idx]
            train_booms = boom_frames[train_idx]
            train_quals = qualities[train_idx]

            test_ids = [sim_ids[i] for i in test_idx]
            test_booms = boom_frames[test_idx]

            # Train pipeline
            pipeline = BoomDetectionPipeline(
                accept_threshold=0.60,
                seed=seed,
            )
            pipeline.fit(train_ids, train_booms, train_quals, cache)

            # Get predictions
            for sid, true_boom in zip(test_ids, test_booms):
                features = cache[sid]
                pred = pipeline.predict_one(features)
                seed_cnn_errors.append(abs(pred.model_predictions['cnn'] - true_boom))
                seed_hgb_errors.append(abs(pred.model_predictions['hgb'] - true_boom))

        cnn_maes.append(np.mean(seed_cnn_errors))
        hgb_maes.append(np.mean(seed_hgb_errors))

    return (
        EvalResult(
            name="CNN",
            mae_mean=np.mean(cnn_maes),
            mae_std=np.std(cnn_maes),
            extra={}
        ),
        EvalResult(
            name="HGB",
            mae_mean=np.mean(hgb_maes),
            mae_std=np.std(hgb_maes),
            extra={}
        )
    )


def evaluate_hgb(
    dataset,
    cache: FeatureCache,
    seeds: list[int],
) -> EvalResult:
    """Evaluate HistGBM classifier using CachedEvaluator."""
    evaluator = CachedEvaluator(dataset, cache)

    result = evaluator.cross_validate(
        lambda: FrameLevelClassifier(model='hist_gbm'),
        seeds=seeds,
        verbose=False,
    )

    return EvalResult(
        name="HGB",
        mae_mean=result.mean_metrics['mae'],
        mae_std=result.std_metrics['mae'],
        extra={
            'within_5': result.mean_metrics['within_5'],
            'within_10': result.mean_metrics['within_10'],
        }
    )


def evaluate_pipeline(
    dataset,
    cache: FeatureCache,
    seeds: list[int],
    accept_threshold: float = 0.60,
) -> EvalResult:
    """Evaluate full pipeline."""
    evaluator = CachedEvaluator(dataset, cache)

    result = evaluator.cross_validate_selective(
        lambda: BoomDetectionPipeline(
            accept_threshold=accept_threshold,
            agreement_formula='sqrt',
            agreement_scale=15.0,
        ),
        seeds=seeds,
        verbose=False,
    )

    return EvalResult(
        name="Pipeline",
        mae_mean=result.mean_metrics['selective_mae'],
        mae_std=result.std_metrics['selective_mae'],
        extra={
            'coverage': result.mean_metrics['coverage'],
            'aurc': result.mean_metrics['aurc'],
        }
    )


def evaluate_quality_correlation(
    dataset,
    cache: FeatureCache,
) -> float:
    """Evaluate quality prediction correlation."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict
    from scipy.stats import spearmanr

    # Get summary features for each simulation
    sim_ids = [a.id for a in dataset.annotations]
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    # Aggregate features (mean over frames)
    X = np.array([cache[sid].mean(axis=0) for sid in sim_ids])

    # CV predictions
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    y_pred = cross_val_predict(model, X, qualities, cv=5)

    corr, _ = spearmanr(y_pred, qualities)
    return corr


def main():
    parser = argparse.ArgumentParser(description="Evaluate coverage features")
    parser.add_argument("data_path", nargs="?", default="data", help="Path to data")
    parser.add_argument("--quick", action="store_true", help="Single-seed evaluation")
    parser.add_argument("-p", "--max-pendulums", type=int, default=2000)
    args = parser.parse_args()

    seeds = [42] if args.quick else [42, 43, 44, 45, 46]

    print("=" * 80)
    print("COVERAGE FEATURE EVALUATION")
    print("=" * 80)
    print(f"Seeds: {len(seeds)} ({'quick' if args.quick else 'full'})")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    print(f"Loaded {len(dataset)} simulations")
    print()

    # Configuration WITHOUT caustic features (baseline)
    config_baseline = FeatureConfig(
        max_pendulums=args.max_pendulums,
        include_caustic=False,
    )

    # Configuration WITH new caustic features
    config_caustic = FeatureConfig(
        max_pendulums=args.max_pendulums,
        include_caustic=True,
    )

    # Extract features for both configs
    print("Extracting features (baseline - no caustic)...")
    t0 = time.time()
    cache_baseline = FeatureCache(config=config_baseline)
    cache_baseline.extract_all(dataset, verbose=True)
    n_baseline = cache_baseline[dataset.annotations[0].id].shape[1]
    print(f"  Features: {n_baseline}, Time: {time.time()-t0:.1f}s")
    print()

    print("Extracting features (with caustic)...")
    t0 = time.time()
    cache_caustic = FeatureCache(config=config_caustic)
    cache_caustic.extract_all(dataset, verbose=True)
    n_caustic = cache_caustic[dataset.annotations[0].id].shape[1]
    print(f"  Features: {n_caustic}, Time: {time.time()-t0:.1f}s")
    print()

    results = []

    # =========================================================================
    # 1. CNN & HGB Evaluation (from pipeline)
    # =========================================================================
    print("=" * 80)
    print("1. CNN & HGB MODEL EVALUATION (via pipeline)")
    print("=" * 80)

    print("\nEvaluating CNN & HGB without caustic features...")
    t0 = time.time()
    cnn_baseline, hgb_baseline = evaluate_cnn_hgb_from_pipeline(dataset, cache_baseline, seeds)
    print(f"  CNN MAE: {cnn_baseline.mae_mean:.2f} +/- {cnn_baseline.mae_std:.2f}")
    print(f"  HGB MAE: {hgb_baseline.mae_mean:.2f} +/- {hgb_baseline.mae_std:.2f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    print("\nEvaluating CNN & HGB with caustic features...")
    t0 = time.time()
    cnn_caustic, hgb_caustic = evaluate_cnn_hgb_from_pipeline(dataset, cache_caustic, seeds)
    print(f"  CNN MAE: {cnn_caustic.mae_mean:.2f} +/- {cnn_caustic.mae_std:.2f}")
    print(f"  HGB MAE: {hgb_caustic.mae_mean:.2f} +/- {hgb_caustic.mae_std:.2f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    cnn_improvement = (cnn_baseline.mae_mean - cnn_caustic.mae_mean) / cnn_baseline.mae_mean * 100
    hgb_improvement = (hgb_baseline.mae_mean - hgb_caustic.mae_mean) / hgb_baseline.mae_mean * 100
    print(f"\n  CNN Change: {cnn_improvement:+.1f}% {'improvement' if cnn_improvement > 0 else 'degradation'}")
    print(f"  HGB Change: {hgb_improvement:+.1f}% {'improvement' if hgb_improvement > 0 else 'degradation'}")

    results.append(('CNN', cnn_baseline, cnn_caustic))
    results.append(('HGB', hgb_baseline, hgb_caustic))

    # =========================================================================
    # 2. Standalone HGB Classifier (for comparison)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. FRAME-LEVEL HISTGBM CLASSIFIER")
    print("=" * 80)

    print("\nEvaluating frame-level HGB without caustic features...")
    t0 = time.time()
    hgb_frame_baseline = evaluate_hgb(dataset, cache_baseline, seeds)
    print(f"  MAE: {hgb_frame_baseline.mae_mean:.2f} +/- {hgb_frame_baseline.mae_std:.2f} ({time.time()-t0:.1f}s)")

    print("\nEvaluating frame-level HGB with caustic features...")
    t0 = time.time()
    hgb_frame_caustic = evaluate_hgb(dataset, cache_caustic, seeds)
    print(f"  MAE: {hgb_frame_caustic.mae_mean:.2f} +/- {hgb_frame_caustic.mae_std:.2f} ({time.time()-t0:.1f}s)")

    improvement = (hgb_frame_baseline.mae_mean - hgb_frame_caustic.mae_mean) / hgb_frame_baseline.mae_mean * 100
    print(f"\n  Change: {improvement:+.1f}% {'improvement' if improvement > 0 else 'degradation'}")
    results.append(('HGB-Frame', hgb_frame_baseline, hgb_frame_caustic))

    # =========================================================================
    # 3. Quality Prediction Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. QUALITY PREDICTION EVALUATION")
    print("=" * 80)

    print("\nEvaluating quality prediction without caustic features...")
    qual_baseline = evaluate_quality_correlation(dataset, cache_baseline)
    print(f"  Spearman correlation: {qual_baseline:.3f}")

    print("\nEvaluating quality prediction with caustic features...")
    qual_caustic = evaluate_quality_correlation(dataset, cache_caustic)
    print(f"  Spearman correlation: {qual_caustic:.3f}")

    qual_improvement = (qual_caustic - qual_baseline) / abs(qual_baseline) * 100 if qual_baseline != 0 else 0
    print(f"\n  Change: {qual_improvement:+.1f}% {'improvement' if qual_improvement > 0 else 'degradation'}")

    # =========================================================================
    # 4. Full Pipeline Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. FULL PIPELINE EVALUATION")
    print("=" * 80)

    print("\nEvaluating pipeline without caustic features...")
    t0 = time.time()
    pipe_baseline = evaluate_pipeline(dataset, cache_baseline, seeds)
    print(f"  Selective MAE: {pipe_baseline.mae_mean:.2f} +/- {pipe_baseline.mae_std:.2f}")
    print(f"  Coverage: {pipe_baseline.extra['coverage']*100:.1f}%")
    print(f"  Time: {time.time()-t0:.1f}s")

    print("\nEvaluating pipeline with caustic features...")
    t0 = time.time()
    pipe_caustic = evaluate_pipeline(dataset, cache_caustic, seeds)
    print(f"  Selective MAE: {pipe_caustic.mae_mean:.2f} +/- {pipe_caustic.mae_std:.2f}")
    print(f"  Coverage: {pipe_caustic.extra['coverage']*100:.1f}%")
    print(f"  Time: {time.time()-t0:.1f}s")

    improvement = (pipe_baseline.mae_mean - pipe_caustic.mae_mean) / pipe_baseline.mae_mean * 100
    print(f"\n  Change: {improvement:+.1f}% {'improvement' if improvement > 0 else 'degradation'}")
    results.append(('Pipeline', pipe_baseline, pipe_caustic))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<15} {'Baseline MAE':>15} {'Caustic MAE':>15} {'Change':>12}")
    print("-" * 60)

    for name, baseline, caustic in results:
        improvement = (baseline.mae_mean - caustic.mae_mean) / baseline.mae_mean * 100
        print(f"{name:<15} {baseline.mae_mean:>7.2f}+/-{baseline.mae_std:<5.2f} "
              f"{caustic.mae_mean:>7.2f}+/-{caustic.mae_std:<5.2f} {improvement:>+10.1f}%")

    print(f"\n{'Quality Corr':<15} {qual_baseline:>15.3f} {qual_caustic:>15.3f} "
          f"{(qual_caustic-qual_baseline)/abs(qual_baseline)*100 if qual_baseline else 0:>+10.1f}%")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Determine overall impact
    cnn_better = cnn_caustic.mae_mean < cnn_baseline.mae_mean
    hgb_better = hgb_caustic.mae_mean < hgb_baseline.mae_mean
    pipe_better = pipe_caustic.mae_mean < pipe_baseline.mae_mean
    qual_better = qual_caustic > qual_baseline

    improvements = sum([cnn_better, hgb_better, pipe_better, qual_better])

    if improvements >= 3:
        print("\nCaustic features IMPROVE overall performance.")
    elif improvements <= 1:
        print("\nCaustic features HURT overall performance.")
    else:
        print("\nCaustic features have MIXED impact.")

    print(f"\n  CNN: {'better' if cnn_better else 'worse'}")
    print(f"  HGB: {'better' if hgb_better else 'worse'}")
    print(f"  Pipeline: {'better' if pipe_better else 'worse'}")
    print(f"  Quality: {'better' if qual_better else 'worse'}")


if __name__ == "__main__":
    main()
