#!/usr/bin/env python3
"""
Extended sweep of 2-model ThresholdCombiner parameters.

This script tests parameters that have NEVER been swept before:
- agreement_weight / quality_weight: Currently hardcoded at 0.4 / 0.6
- primary_model: Currently always CNN, but LSTM may be better
- score_function: New alternatives (min, product) vs weighted
- Extended thresholds: Push selectivity higher (0.75, 0.80, 0.85)

The key insight is that TESTING COMBINERS IS FREE - models are trained once,
then combiners can be swapped instantly. This allows comprehensive parameter
exploration at minimal cost.

Usage:
    uv run python scripts/sweep_extended.py data
    uv run python scripts/sweep_extended.py data --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np

from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline
from boom_detection.evaluation import CachedEvaluator
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG
from boom_detection.loader import load_dataset
from boom_detection.logging_config import logger, setup_run_logging, log_memory_usage


def main():
    parser = argparse.ArgumentParser(description='Extended sweep of 2-model ThresholdCombiner parameters')
    parser.add_argument('data_path', type=Path, help='Path to data directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for evaluation (default: 42 43 44)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: auto-generated timestamped)')
    args = parser.parse_args()

    # Setup run directory and file logging
    run_dir = args.output or setup_run_logging("sweep_extended")

    logger.info("SWEEP: Extended 2-Model ThresholdCombiner Parameters")
    logger.info(f"Frame models: CNN + HGB")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: PRODUCTION_CONFIG (no caustic)")
    logger.info("")
    logger.info("NEW PARAMETERS BEING TESTED:")
    logger.info("  - agreement_weight / quality_weight")
    logger.info("  - primary_model (cnn, hgb, median)")
    logger.info("  - score_function (weighted, min, product)")
    logger.info("  - Extended thresholds (0.75, 0.80, 0.85)")

    # Load dataset and features
    logger.info("")
    logger.info("Loading dataset...")
    dataset = load_dataset(args.data_path, verbose=False)
    log_memory_usage("after loading")

    cache = FeatureCache(config=PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')

    sim_ids = [a.id for a in dataset.annotations]
    try:
        loaded = cache.load_from_disk(sim_ids, verbose=False)
        if loaded < len(sim_ids):
            logger.info(f"Extracting features ({loaded}/{len(sim_ids)} cached)...")
            cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)
        else:
            logger.info(f"Using cached features ({loaded} sims)")
            dataset.release_simulation_data()
    except (ValueError, FileNotFoundError):
        logger.info("Extracting features...")
        cache.extract_all(dataset, auto_release=True, n_jobs=4, verbose=False)

    gc.collect()
    log_memory_usage("after feature extraction")

    # Create evaluator
    evaluator = CachedEvaluator(dataset, cache)
    logger.info(f"Evaluating on {len(evaluator.sim_ids)} simulations")

    # Create CombinerExperiment - trains models ONCE
    logger.info("")
    logger.info("Creating CombinerExperiment (training models once)...")
    start_time = time.time()
    experiment = evaluator.create_combiner_experiment(
        pipeline_factory=lambda: BoomDetectionPipeline(
            frame_models=('cnn', 'hgb'),  # 2-model pipeline
            combiner=ThresholdCombiner(),  # Default combiner, will be replaced
        ),
        k=5,
        seeds=args.seeds,
        verbose=True,
    )
    train_time = time.time() - start_time
    logger.info(f"CombinerExperiment created in {train_time:.1f}s")
    logger.info("Models trained - now sweeping combiners (this is FREE)")

    # =========================================================================
    # PHASE 1: Weight sweep (agreement_weight / quality_weight)
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: WEIGHT SWEEP (agreement_weight / quality_weight)")
    logger.info("=" * 70)

    # Generate weight combinations
    # Note: We sweep agreement_weight, quality_weight is implicitly 1 - agreement_weight
    # But we also test unconstrained weights (sum != 1)
    weight_combiners = ThresholdCombiner.grid(
        agreement_transform=['sqrt'],  # Best from previous experiments
        disagreement_scale=[15.0],  # Best from previous experiments
        threshold=[0.60, 0.70],
        agreement_weight=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        quality_weight=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
    )

    logger.info(f"Testing {len(weight_combiners)} weight configurations...")
    weight_results = []

    for i, combiner in enumerate(weight_combiners, 1):
        result = experiment.evaluate(combiner, verbose=False)
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        cov = result.mean_metrics.get('coverage', 0)

        weight_results.append({
            'agreement_weight': combiner.agreement_weight,
            'quality_weight': combiner.quality_weight,
            'threshold': combiner.threshold,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': cov,
        })

    # Find best weights
    valid_weights = [r for r in weight_results if not np.isnan(r['mae'])]
    if valid_weights:
        best_weight = min(valid_weights, key=lambda x: x['mae'])
        logger.info(f"Best weights: agreement={best_weight['agreement_weight']}, "
                   f"quality={best_weight['quality_weight']} "
                   f"-> MAE {best_weight['mae']:.2f} at {best_weight['coverage']:.1%}")

    # =========================================================================
    # PHASE 2: Primary model sweep
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: PRIMARY MODEL SWEEP")
    logger.info("=" * 70)

    primary_combiners = ThresholdCombiner.grid(
        agreement_transform=['sqrt'],
        disagreement_scale=[15.0],
        threshold=[0.60, 0.70],
        primary_model=['cnn', 'hgb', 'median'],
    )

    logger.info(f"Testing {len(primary_combiners)} primary_model configurations...")
    primary_results = []

    for combiner in primary_combiners:
        result = experiment.evaluate(combiner, verbose=False)
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        cov = result.mean_metrics.get('coverage', 0)

        primary_results.append({
            'primary_model': combiner.primary_model,
            'threshold': combiner.threshold,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': cov,
        })

        logger.info(f"primary_model={combiner.primary_model}, threshold={combiner.threshold}: "
                   f"MAE {mae:.2f} +/- {mae_std:.2f}, coverage {cov:.1%}")

    # =========================================================================
    # PHASE 3: Score function sweep
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: SCORE FUNCTION SWEEP (weighted vs min vs product)")
    logger.info("=" * 70)

    # For min/product, threshold meaning changes - need different values
    score_combiners = []

    # Weighted (baseline)
    score_combiners += ThresholdCombiner.grid(
        score_function=['weighted'],
        agreement_transform=['sqrt'],
        disagreement_scale=[15.0],
        threshold=[0.60, 0.65, 0.70, 0.75],
    )

    # Min function (both agreement AND quality must exceed threshold)
    score_combiners += ThresholdCombiner.grid(
        score_function=['min'],
        agreement_transform=['sqrt'],
        disagreement_scale=[15.0],
        threshold=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    )

    # Product function (penalizes weakness in either)
    score_combiners += ThresholdCombiner.grid(
        score_function=['product'],
        agreement_transform=['sqrt'],
        disagreement_scale=[15.0],
        threshold=[0.30, 0.40, 0.50, 0.60, 0.70],  # Product gives lower scores
    )

    logger.info(f"Testing {len(score_combiners)} score_function configurations...")
    score_results = []

    for combiner in score_combiners:
        result = experiment.evaluate(combiner, verbose=False)
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        cov = result.mean_metrics.get('coverage', 0)

        score_results.append({
            'score_function': combiner.score_function,
            'threshold': combiner.threshold,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': cov,
        })

    # Report best for each score function
    for score_func in ['weighted', 'min', 'product']:
        func_results = [r for r in score_results if r['score_function'] == score_func and not np.isnan(r['mae'])]
        if func_results:
            best = min(func_results, key=lambda x: x['mae'])
            logger.info(f"Best {score_func}: threshold={best['threshold']:.2f} "
                       f"-> MAE {best['mae']:.2f} +/- {best['mae_std']:.2f} at {best['coverage']:.1%}")

    # =========================================================================
    # PHASE 4: Extended threshold sweep (push selectivity higher)
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: EXTENDED THRESHOLD SWEEP (0.75, 0.80, 0.85)")
    logger.info("=" * 70)

    extended_combiners = ThresholdCombiner.grid(
        agreement_transform=['sqrt', 'sigmoid'],  # sigmoid good at extreme selectivity
        disagreement_scale=[3.0, 5.0, 10.0, 15.0],
        threshold=[0.70, 0.75, 0.80, 0.85],
    )

    logger.info(f"Testing {len(extended_combiners)} extended threshold configurations...")
    extended_results = []

    for combiner in extended_combiners:
        result = experiment.evaluate(combiner, verbose=False)
        mae = result.mean_metrics.get('selective_mae', float('nan'))
        mae_std = result.std_metrics.get('selective_mae', 0)
        cov = result.mean_metrics.get('coverage', 0)

        extended_results.append({
            'formula': combiner.agreement_transform,
            'scale': combiner.disagreement_scale,
            'threshold': combiner.threshold,
            'mae': mae,
            'mae_std': mae_std,
            'coverage': cov,
        })

    # Sort by MAE and show top 10
    extended_by_mae = sorted(extended_results, key=lambda x: x['mae'] if not np.isnan(x['mae']) else float('inf'))

    logger.info("")
    logger.info("TOP 10 EXTENDED THRESHOLD CONFIGURATIONS:")
    logger.info(f"{'Rank':<5} {'Formula':<10} {'Scale':<7} {'Thresh':<7} {'MAE +/- std':<16} {'Coverage':<12}")
    logger.info("-" * 65)

    for rank, r in enumerate(extended_by_mae[:10], 1):
        logger.info(f"{rank:<5} {r['formula']:<10} {r['scale']:<7.0f} {r['threshold']:<7.2f} "
                   f"{r['mae']:.2f} +/- {r['mae_std']:.2f}   {r['coverage']:.1%}")

    # =========================================================================
    # Save all results
    # =========================================================================
    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save weight results
        with open(run_dir / 'weight_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=weight_results[0].keys())
            writer.writeheader()
            writer.writerows(weight_results)

        # Save primary model results
        with open(run_dir / 'primary_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=primary_results[0].keys())
            writer.writeheader()
            writer.writerows(primary_results)

        # Save score function results
        with open(run_dir / 'score_function_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=score_results[0].keys())
            writer.writeheader()
            writer.writerows(score_results)

        # Save extended threshold results
        with open(run_dir / 'extended_threshold_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=extended_results[0].keys())
            writer.writeheader()
            writer.writerows(extended_results)

        logger.info(f"\nResults saved to {run_dir}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTENDED SWEEP COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
