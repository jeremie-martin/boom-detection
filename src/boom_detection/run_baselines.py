"""
Run all baseline predictors and compare results.

Usage:
    uv run python -m boom_detection.run_baselines data

This validates the evaluation framework and establishes baseline performance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .loader import load_dataset
from .evaluation import Evaluator
from .baselines import get_all_baselines


def run_baselines(data_path: str, k: int = 5, seed: int = 42, max_samples: int | None = None):
    """Run all baselines and print comparison."""
    print(f"Loading dataset from: {data_path}")
    if max_samples:
        print(f"Limiting to {max_samples} samples")
    print("=" * 60)

    t0 = time.time()
    dataset = load_dataset(data_path, verbose=True, max_samples=max_samples)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print()

    evaluator = Evaluator(dataset)
    baselines = get_all_baselines()

    results = {}
    print(f"Running {k}-fold cross-validation (seed={seed})")
    print("=" * 60)

    for name, predictor in baselines.items():
        print(f"\n{name}:")
        print("-" * 40)

        t0 = time.time()
        try:
            result = evaluator.cross_validate(
                predictor,
                k=k,
                seed=seed,
                task="frame",
                verbose=False,
            )
            elapsed = time.time() - t0

            results[name] = result
            metrics = result.aggregate_metrics

            print(f"  MAE:       {metrics['mae']:.2f} frames")
            print(f"  Median AE: {metrics['median_ae']:.2f} frames")
            print(f"  RMSE:      {metrics['rmse']:.2f} frames")
            print(f"  Max AE:    {metrics['max_ae']:.0f} frames")
            print(f"  Within 5:  {metrics['within_5']:.1%}")
            print(f"  Within 10: {metrics['within_10']:.1%}")
            print(f"  Corr:      {metrics['correlation']:.3f}")
            print(f"  Time:      {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n")
    print("=" * 60)
    print("SUMMARY - Boom Frame Prediction")
    print("=" * 60)
    print(f"{'Method':<25} {'MAE':>8} {'MedAE':>8} {'W/in 5':>8} {'W/in 10':>8}")
    print("-" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1].aggregate_metrics['mae'])
    for name, result in sorted_results:
        m = result.aggregate_metrics
        print(f"{name:<25} {m['mae']:>8.2f} {m['median_ae']:>8.2f} {m['within_5']:>7.1%} {m['within_10']:>8.1%}")

    # Best result
    best_name, best_result = sorted_results[0]
    print("-" * 60)
    print(f"Best: {best_name} with MAE = {best_result.aggregate_metrics['mae']:.2f} frames")
    print()

    # Per-sample error analysis for best method
    print("Worst predictions (best method):")
    print("-" * 40)
    errors = best_result.per_sample_errors()[:5]
    for idx, gt, pred, err in errors:
        ann = dataset.annotations[idx]
        print(f"  {ann.id}: predicted {pred:.0f}, actual {gt:.0f}, error {err:.0f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline predictors")
    parser.add_argument("data_path", nargs="?", default="data", help="Path to data directory")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Limit samples (for quick testing)")
    parser.add_argument("-k", "--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_baselines(args.data_path, k=args.folds, seed=args.seed, max_samples=args.max_samples)
