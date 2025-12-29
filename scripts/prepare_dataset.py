#!/usr/bin/env python3
"""
Script to prepare the boom detection dataset.

Copies simulation data from the double-pendulum project's eval2 directory.

Usage:
    uv run python scripts/prepare_dataset.py /path/to/double-pendulum/output/eval2
    uv run python scripts/prepare_dataset.py /path/to/double-pendulum/output/eval2 --output-dir /path/to/data
    uv run python scripts/prepare_dataset.py /path/to/double-pendulum/output/eval2 --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the boom detection dataset.")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Path to the double-pendulum eval output directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Destination dataset directory (default: ./data in repo root).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite annotations.json instead of appending when it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    dest_dir = args.output_dir

    # Validate source
    source_annotations = source_dir / 'annotations.json'
    if not source_annotations.exists():
        print(f"Error: annotations.json not found in {source_dir}")
        return 1

    with open(source_annotations) as f:
        data = json.load(f)

    print(f"Found {len(data['annotations'])} annotations")

    existing_annotations = []
    existing_ids = set()
    output_path = dest_dir / 'annotations.json'

    if output_path.exists() and not args.overwrite:
        with open(output_path) as f:
            existing_data = json.load(f)

        if (
            existing_data.get('version') != data.get('version')
            or existing_data.get('target_defs') != data.get('target_defs')
        ):
            print("Error: target_defs/version mismatch between source and destination annotations.")
            print(f"  Source version: {data.get('version')}")
            print(f"  Destination version: {existing_data.get('version')}")
            return 1

        existing_annotations = existing_data.get('annotations', [])
        existing_ids = {
            ann.get('id')
            for ann in existing_annotations
            if isinstance(ann, dict) and 'id' in ann
        }
        print(f"Appending to existing dataset with {len(existing_annotations)} annotations")

    # Create destination
    simulations_dir = dest_dir / 'simulations'
    simulations_dir.mkdir(parents=True, exist_ok=True)

    # Process each annotation
    updated_annotations = []
    copied_count = 0
    skipped_duplicates = 0

    for ann in data['annotations']:
        run_id = ann['id']

        if run_id in existing_ids:
            print(f"Skipping existing annotation id: {run_id}")
            skipped_duplicates += 1
            continue

        source_run_dir = source_dir / run_id

        if not source_run_dir.exists():
            print(f"Warning: Source directory not found: {source_run_dir}")
            continue

        dest_run_dir = simulations_dir / run_id
        dest_run_dir.mkdir(exist_ok=True)

        # Files to copy (NO metrics.csv)
        files_to_copy = [
            'simulation_data.bin',
            'metadata.json',
        ]

        for filename in files_to_copy:
            src = source_run_dir / filename
            dst = dest_run_dir / filename

            if src.exists():
                if not dst.exists():
                    print(f"Copying {run_id}/{filename}...")
                    shutil.copy2(src, dst)
                    copied_count += 1
            elif filename == 'simulation_data.bin':
                print(f"Warning: Missing required file {src}")
                continue

        # Update annotation path
        updated_ann = ann.copy()
        updated_ann['data_path'] = f"simulations/{run_id}/simulation_data.bin"
        updated_annotations.append(updated_ann)

    # Write updated annotations
    output_annotations = {
        'version': data['version'],
        'target_defs': data['target_defs'],
        'annotations': existing_annotations + updated_annotations,
    }

    with open(output_path, 'w') as f:
        json.dump(output_annotations, f, indent=2)

    print(f"\nWrote {len(output_annotations['annotations'])} annotations to {output_path}")
    if existing_annotations:
        print(f"  Appended {len(updated_annotations)} new annotations")
    if skipped_duplicates:
        print(f"  Skipped {skipped_duplicates} duplicate annotations")
    print(f"Copied {copied_count} files")

    # Calculate total size
    total_size = sum(
        f.stat().st_size
        for f in simulations_dir.rglob('*')
        if f.is_file()
    )

    print(f"\nDataset prepared successfully!")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Location: {dest_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
