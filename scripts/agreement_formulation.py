"""
Agreement Score Formulation Experiment.

The current agreement score formula is:
    agreement_score = 1 - min(|CNN - HGB| / 10, 1.0)

This linearly interpolates from 0 (10+ frames apart) to 1 (identical).
But is 10 the right denominator? And should it be linear?

This script tests:
1. Different denominators: 5, 10, 15, 20
2. Different transforms: linear, sigmoid, sqrt

Usage:
    uv run python scripts/agreement_formulation.py data
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import KFold

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.metrics import SelectivePrediction, compute_selective_metrics
from boom_detection.sequence_models import set_global_seed


def agreement_score_linear(diff: float, scale: float) -> float:
    """Linear agreement: 1 - min(diff/scale, 1.0)"""
    return 1.0 - min(abs(diff) / scale, 1.0)


def agreement_score_sigmoid(diff: float, scale: float) -> float:
    """Sigmoid agreement: 1 / (1 + exp(diff/scale - 1))"""
    x = abs(diff) / scale
    return 1.0 / (1.0 + np.exp(2 * (x - 0.5)))  # Steeper sigmoid centered at scale/2


def agreement_score_sqrt(diff: float, scale: float) -> float:
    """Sqrt-based agreement: 1 - sqrt(min(diff/scale, 1.0))"""
    return 1.0 - np.sqrt(min(abs(diff) / scale, 1.0))


def agreement_score_quadratic(diff: float, scale: float) -> float:
    """Quadratic agreement: 1 - (min(diff/scale, 1.0))^2"""
    x = min(abs(diff) / scale, 1.0)
    return 1.0 - x * x


SCORE_FUNCTIONS = {
    'linear': agreement_score_linear,
    'sigmoid': agreement_score_sigmoid,
    'sqrt': agreement_score_sqrt,
    'quadratic': agreement_score_quadratic,
}


def evaluate_formulation(
    cnn_preds: np.ndarray,
    hgb_preds: np.ndarray,
    quality_preds: np.ndarray,
    true_booms: np.ndarray,
    true_qualities: np.ndarray,
    score_fn: str,
    scale: float,
    agreement_weight: float = 0.4,
    quality_weight: float = 0.6,
    accept_threshold: float = 0.53,
) -> dict:
    """
    Evaluate a specific agreement score formulation.

    Returns metrics for the selective prediction.
    """
    score_func = SCORE_FUNCTIONS[score_fn]

    predictions = []
    for i in range(len(cnn_preds)):
        diff = abs(cnn_preds[i] - hgb_preds[i])
        agreement = score_func(diff, scale)
        accept_score = agreement_weight * agreement + quality_weight * quality_preds[i]

        accepted = accept_score >= accept_threshold
        predictions.append(SelectivePrediction(
            boom_frame=int(cnn_preds[i]) if accepted else None,
            accepted=accepted,
            cnn_pred=int(cnn_preds[i]),
            hgb_pred=int(hgb_preds[i]),
            disagreement=diff,
            predicted_quality=float(quality_preds[i]),
            accept_score=float(accept_score),
        ))

    metrics = compute_selective_metrics(predictions, true_booms, true_qualities)
    return metrics


def run_agreement_sweep(
    data_path: str,
    scales: list[float] = None,
    score_functions: list[str] = None,
    n_folds: int = 5,
    seeds: list[int] = None,
    verbose: bool = True,
):
    """
    Run agreement formulation sweep.

    Tests different scales and score functions.
    """
    if scales is None:
        scales = [5, 10, 15, 20]

    if score_functions is None:
        score_functions = ['linear', 'sigmoid', 'sqrt', 'quadratic']

    if seeds is None:
        seeds = [42, 43, 44]

    print(f"Loading dataset from: {data_path}")
    dataset = load_dataset(data_path, verbose=verbose)
    print(f"Loaded {len(dataset)} simulations")

    # Set up feature cache
    config = FeatureConfig(max_pendulums=2000)
    cache = FeatureCache(config=config, cache_dir='.feature_cache')
    cache.extract_all(dataset, verbose=verbose, auto_release=True)

    # Get data
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    # Results storage: {(fn, scale): {metric: [values]}}
    results = {}
    for fn in score_functions:
        for scale in scales:
            results[(fn, scale)] = {
                'selective_mae': [],
                'coverage': [],
                'aurc': [],
            }

    print(f"\nRunning {n_folds}-fold CV × {len(seeds)} seeds")
    print(f"Score functions: {score_functions}")
    print(f"Scales: {scales}")
    print("=" * 70)

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
            test_quals = qualities[test_idx]

            # Train pipeline to get raw predictions
            pipeline = BoomDetectionPipeline(
                accept_threshold=0.0,  # We'll handle thresholding ourselves
                seed=seed * 1000 + fold_idx,
            )
            pipeline.fit(train_ids, train_booms, train_quals, cache)

            # Get raw model outputs (predictions for all test samples)
            cnn_preds = []
            hgb_preds = []
            quality_preds = []

            for test_id in test_ids:
                result = pipeline.predict_one(cache[test_id])
                cnn_preds.append(result.cnn_pred)
                hgb_preds.append(result.hgb_pred)
                quality_preds.append(result.predicted_quality)

            cnn_preds = np.array(cnn_preds)
            hgb_preds = np.array(hgb_preds)
            quality_preds = np.array(quality_preds)

            # Evaluate each formulation
            for fn in score_functions:
                for scale in scales:
                    metrics = evaluate_formulation(
                        cnn_preds, hgb_preds, quality_preds,
                        test_booms, test_quals,
                        fn, scale,
                    )
                    results[(fn, scale)]['selective_mae'].append(metrics.get('selective_mae', np.nan))
                    results[(fn, scale)]['coverage'].append(metrics['coverage'])
                    results[(fn, scale)]['aurc'].append(metrics.get('aurc', np.nan))

            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds} complete")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("RESULTS: Agreement Score Formulation Sweep")
    print("=" * 70)
    print(f"{'Function':<12} {'Scale':<8} {'MAE':>10} {'Coverage':>12} {'AURC':>12}")
    print("-" * 70)

    best_key = None
    best_mae = float('inf')

    for fn in score_functions:
        for scale in scales:
            key = (fn, scale)
            maes = [m for m in results[key]['selective_mae'] if not np.isnan(m)]
            covs = results[key]['coverage']
            aurcs = [a for a in results[key]['aurc'] if not np.isnan(a)]

            if not maes:
                continue

            mean_mae = np.mean(maes)
            mean_cov = np.mean(covs) * 100
            mean_aurc = np.mean(aurcs) if aurcs else np.nan

            marker = ""
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_key = key
                marker = " *"

            print(f"{fn:<12} {scale:<8} {mean_mae:>10.2f} {mean_cov:>11.1f}% {mean_aurc:>12.2f}{marker}")

    print("-" * 70)
    if best_key:
        print(f"Best: {best_key[0]} with scale={best_key[1]} -> MAE = {best_mae:.2f}")

    # Compare best to current default (linear, 10)
    default_key = ('linear', 10)
    if best_key and best_key != default_key:
        from scipy import stats

        best_maes = [m for m in results[best_key]['selective_mae'] if not np.isnan(m)]
        default_maes = [m for m in results[default_key]['selective_mae'] if not np.isnan(m)]

        if len(best_maes) == len(default_maes):
            t_stat, p_value = stats.ttest_rel(best_maes, default_maes)
            print(f"\nPaired t-test vs default (linear, 10): t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print("  -> Significant improvement! (p < 0.05)")
            else:
                print("  -> Not statistically significant at α=0.05")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agreement score formulation experiment")
    parser.add_argument("data_path", nargs="?", default="data")
    parser.add_argument("--scales", type=str, default="5,10,15,20",
                        help="Comma-separated scales")
    parser.add_argument("--functions", type=str, default="linear,sigmoid,sqrt,quadratic",
                        help="Comma-separated score functions")
    parser.add_argument("-k", "--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(',')]
    score_functions = args.functions.split(',')
    seeds = list(range(42, 42 + args.seeds))

    run_agreement_sweep(
        args.data_path,
        scales=scales,
        score_functions=score_functions,
        n_folds=args.folds,
        seeds=seeds,
        verbose=not args.quiet,
    )
