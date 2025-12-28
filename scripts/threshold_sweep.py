"""
CNN Threshold Sweep Experiment.

This script investigates whether the default 0.5 threshold for determining
the boom frame from CNN probabilities is optimal.

The CNN outputs P(after_boom | features) for each frame. Currently, we find
the first frame where P >= 0.5. But is 0.5 optimal?

Usage:
    uv run python scripts/threshold_sweep.py data --thresholds 0.3,0.4,0.5,0.6,0.7
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.model_selection import KFold

from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.sequence_models import CNNClassifier, SequenceTrainer, set_global_seed


def predict_with_threshold(
    trainer: SequenceTrainer,
    sim_ids: list[str],
    cache: FeatureCache,
    threshold: float,
) -> np.ndarray:
    """
    Predict boom frames using a custom threshold.

    Args:
        trainer: Trained SequenceTrainer
        sim_ids: Simulation IDs to predict
        cache: Feature cache
        threshold: Probability threshold for boom detection

    Returns:
        Array of predicted boom frames
    """
    trainer.model.eval()
    predictions = []

    with torch.no_grad():
        for sim_id in sim_ids:
            features = cache[sim_id]
            features_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
            features_t = features_t.to(trainer.device)

            logits = trainer.model(features_t)  # (1, frames)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            # Find first frame where prob >= threshold
            crossings = np.where(probs >= threshold)[0]
            if len(crossings) > 0:
                boom_pred = crossings[0]
            else:
                # No crossing found - use frame with prob closest to threshold
                boom_pred = np.argmin(np.abs(probs - threshold))

            predictions.append(int(boom_pred))

    return np.array(predictions)


def run_threshold_sweep(
    data_path: str,
    thresholds: list[float] = None,
    n_folds: int = 5,
    seeds: list[int] = None,
    verbose: bool = True,
):
    """
    Run threshold sweep experiment.

    Args:
        data_path: Path to data directory
        thresholds: Thresholds to test
        n_folds: Number of CV folds
        seeds: Random seeds for evaluation
        verbose: Print progress
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    if seeds is None:
        seeds = [42, 43, 44]

    print(f"Loading dataset from: {data_path}")
    dataset = load_dataset(data_path, verbose=verbose)
    print(f"Loaded {len(dataset)} simulations")

    # Set up feature cache
    config = FeatureConfig(max_pendulums=2000)
    cache = FeatureCache(config=config, cache_dir='.feature_cache')
    cache.extract_all(dataset, verbose=verbose)
    n_features = cache[dataset.annotations[0].id].shape[1]

    # Get data
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])

    # Results storage
    results = {t: {'maes': [], 'within_5': [], 'within_10': []} for t in thresholds}

    print(f"\nRunning {n_folds}-fold CV × {len(seeds)} seeds")
    print(f"Thresholds: {thresholds}")
    print("=" * 60)

    for seed in seeds:
        if verbose:
            print(f"\nSeed {seed}:")

        set_global_seed(seed)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sim_ids)):
            train_ids = [sim_ids[i] for i in train_idx]
            test_ids = [sim_ids[i] for i in test_idx]
            train_booms = boom_frames[train_idx]
            test_booms = boom_frames[test_idx]

            # Train CNN
            cnn = CNNClassifier(
                n_features=n_features,
                hidden_dim=64,
                kernel_sizes=(5, 11, 21),
                dropout=0.3,
            )
            trainer = SequenceTrainer(
                cnn,
                epochs=50,
                patience=10,
                seed=seed * 1000 + fold_idx,
                verbose=False,
            )
            trainer.fit(train_ids, train_booms, cache)

            # Evaluate at each threshold
            for threshold in thresholds:
                preds = predict_with_threshold(trainer, test_ids, cache, threshold)
                errors = np.abs(preds - test_booms)

                results[threshold]['maes'].extend(errors.tolist())
                results[threshold]['within_5'].extend((errors <= 5).tolist())
                results[threshold]['within_10'].extend((errors <= 10).tolist())

            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds} complete")

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("RESULTS: CNN Threshold Sweep")
    print("=" * 60)
    print(f"{'Threshold':<12} {'MAE':>10} {'Std':>8} {'Within 5':>12} {'Within 10':>12}")
    print("-" * 60)

    best_threshold = None
    best_mae = float('inf')

    for threshold in sorted(thresholds):
        all_maes = results[threshold]['maes']
        all_w5 = results[threshold]['within_5']
        all_w10 = results[threshold]['within_10']

        mean_mae = np.mean(all_maes)
        std_mae = np.std(all_maes)
        w5 = np.mean(all_w5) * 100
        w10 = np.mean(all_w10) * 100

        marker = ""
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_threshold = threshold
            marker = " *"

        print(f"{threshold:<12.2f} {mean_mae:>10.2f} {std_mae:>8.2f} {w5:>11.1f}% {w10:>11.1f}%{marker}")

    print("-" * 60)
    print(f"Best threshold: {best_threshold:.2f} with MAE = {best_mae:.2f}")

    # Statistical significance test: compare best to 0.5
    if best_threshold != 0.5:
        from scipy import stats
        best_errors = np.array(results[best_threshold]['maes'])
        baseline_errors = np.array(results[0.5]['maes'])
        t_stat, p_value = stats.ttest_rel(best_errors, baseline_errors)
        print(f"\nPaired t-test vs 0.5: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  -> Significant improvement! (p < 0.05)")
        else:
            print("  -> Not statistically significant at α=0.05")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN threshold sweep experiment")
    parser.add_argument("data_path", nargs="?", default="data")
    parser.add_argument("--thresholds", type=str, default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
                        help="Comma-separated thresholds")
    parser.add_argument("-k", "--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(',')]
    seeds = list(range(42, 42 + args.seeds))

    run_threshold_sweep(
        args.data_path,
        thresholds=thresholds,
        n_folds=args.folds,
        seeds=seeds,
        verbose=not args.quiet,
    )
