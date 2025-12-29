#!/usr/bin/env python3
"""
Standalone inference demo for boom detection pipeline.

This script demonstrates the complete deployment workflow:
1. Train on the full dataset (once)
2. Save the trained model
3. Load the model and predict on new simulations

Usage:
    # Train and save model
    uv run python scripts/inference_demo.py train data/ --output models/boom_v1

    # Predict on a single simulation file
    uv run python scripts/inference_demo.py predict models/boom_v1 path/to/simulation_data.bin

    # Predict on all simulations in a directory
    uv run python scripts/inference_demo.py predict models/boom_v1 data/simulations/

    # Batch predict with JSON output
    uv run python scripts/inference_demo.py predict models/boom_v1 data/simulations/ --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def train_and_save(data_path: Path, output_path: Path, config_name: str = 'balanced') -> None:
    """Train pipeline on full dataset and save to disk."""
    from boom_detection.loader import load_dataset
    from boom_detection.features import FeatureCache, FeatureConfig
    from boom_detection.deploy_pipeline import BoomDetectionPipeline

    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset(data_path, verbose=False)
    print(f"  {len(dataset)} simulations loaded")

    # Extract features
    print("Extracting features...")
    config = FeatureConfig()
    cache = FeatureCache(config=config, cache_dir='.feature_cache/production')
    cache.extract_all(dataset, n_jobs=4, auto_release=True)

    # Prepare training data
    sim_ids = [a.id for a in dataset.annotations]
    boom_frames = np.array([a.boom_frame for a in dataset.annotations])
    qualities = np.array([a.boom_quality for a in dataset.annotations])

    # Configure pipeline based on selectivity preference
    configs = {
        'default': {'agreement_formula': 'linear', 'agreement_scale': 10.0},
        'balanced': {'agreement_formula': 'sqrt', 'agreement_scale': 15.0},
        'selective': {'agreement_formula': 'sqrt', 'agreement_scale': 5.0},
    }

    if config_name not in configs:
        print(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        sys.exit(1)

    print(f"\nTraining pipeline (config: {config_name})...")
    pipeline = BoomDetectionPipeline(
        accept_threshold=0.60,
        calibrate_quality=True,
        feature_config=config,  # Save feature config with model
        **configs[config_name],
    )
    pipeline.fit(sim_ids, boom_frames, qualities, cache)

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline.save(output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"  - cnn.pt (CNN model)")
    print(f"  - hgb.joblib (HistGBM model)")
    print(f"  - quality.joblib (quality predictor)")
    print(f"  - config.json (pipeline config)")
    print(f"  - feature_config.json (feature extraction config)")


def predict_single(model_path: Path, simulation_path: Path) -> dict:
    """Load model and predict on a single simulation."""
    from boom_detection.deploy_pipeline import BoomDetectionPipeline

    # Load pretrained model
    pipeline = BoomDetectionPipeline.from_pretrained(model_path)

    # Predict
    result = pipeline.predict_file(simulation_path)

    return {
        'path': str(simulation_path),
        'accepted': result.accepted,
        'boom_frame': result.boom_frame,
        'accept_score': result.accept_score,
        'predicted_quality': result.predicted_quality,
        'disagreement': result.disagreement,
        'model_predictions': result.model_predictions,
    }


def predict_batch(model_path: Path, input_path: Path, output_file: Path | None = None) -> list[dict]:
    """Predict on multiple simulations."""
    from boom_detection.deploy_pipeline import BoomDetectionPipeline

    # Find simulation files
    input_path = Path(input_path)
    if input_path.is_file():
        sim_files = [input_path]
    else:
        # Look for simulation_data.bin files recursively
        sim_files = list(input_path.rglob('simulation_data.bin'))
        if not sim_files:
            print(f"No simulation_data.bin files found in {input_path}")
            sys.exit(1)

    print(f"Found {len(sim_files)} simulation(s)")

    # Load model once
    print(f"Loading model from {model_path}...")
    pipeline = BoomDetectionPipeline.from_pretrained(model_path)

    # Predict
    results = []
    n_accepted = 0
    for i, sim_file in enumerate(sim_files, 1):
        try:
            result = pipeline.predict_file(sim_file)
            entry = {
                'path': str(sim_file),
                'accepted': result.accepted,
                'boom_frame': result.boom_frame,
                'accept_score': round(result.accept_score, 3),
                'predicted_quality': result.predicted_quality,
                'disagreement': round(result.disagreement, 2),
                'model_predictions': {k: int(v) for k, v in result.model_predictions.items()},
            }
            results.append(entry)
            if result.accepted:
                n_accepted += 1

            # Progress
            status = f"✓ boom@{result.boom_frame}" if result.accepted else "✗ rejected"
            print(f"  [{i}/{len(sim_files)}] {sim_file.parent.name}: {status} (score={result.accept_score:.2f})")

        except Exception as e:
            print(f"  [{i}/{len(sim_files)}] {sim_file}: ERROR - {e}")
            results.append({'path': str(sim_file), 'error': str(e)})

    print(f"\nSummary: {n_accepted}/{len(results)} simulations accepted ({n_accepted/len(results)*100:.0f}%)")

    # Save if output specified
    if output_file:
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Boom detection inference demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train and save model')
    train_parser.add_argument('data_path', type=Path, help='Path to data directory')
    train_parser.add_argument('--output', '-o', type=Path, required=True,
                             help='Output directory for saved model')
    train_parser.add_argument('--config', choices=['default', 'balanced', 'selective'],
                             default='balanced',
                             help='Pipeline config: default (39%% coverage), '
                                  'balanced (30%% coverage), selective (14%% coverage)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on simulations')
    predict_parser.add_argument('model_path', type=Path, help='Path to saved model directory')
    predict_parser.add_argument('input', type=Path,
                               help='Path to simulation_data.bin or directory of simulations')
    predict_parser.add_argument('--output', '-o', type=Path, default=None,
                               help='Output JSONL file for results')

    args = parser.parse_args()

    if args.command == 'train':
        train_and_save(args.data_path, args.output, args.config)

    elif args.command == 'predict':
        if args.input.is_file():
            # Single file prediction
            result = predict_single(args.model_path, args.input)
            print("\nPrediction:")
            print(f"  Accepted: {result['accepted']}")
            if result['accepted']:
                print(f"  Boom frame: {result['boom_frame']}")
            print(f"  Accept score: {result['accept_score']:.3f}")
            print(f"  Predicted quality: {result['predicted_quality']:.2f}")
            print(f"  Disagreement: {result['disagreement']:.2f}")
            print(f"  Model predictions: {result['model_predictions']}")
        else:
            # Batch prediction
            predict_batch(args.model_path, args.input, args.output)


if __name__ == '__main__':
    main()
