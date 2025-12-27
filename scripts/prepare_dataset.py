#!/usr/bin/env python3
"""
Script to prepare the boom detection dataset.

Copies simulation data from the double-pendulum project's eval2 directory.

Usage:
    uv run python scripts/prepare_dataset.py /path/to/double-pendulum/output/eval2
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/prepare_dataset.py <source_eval_dir>")
        print("Example: python scripts/prepare_dataset.py ../double-pendulum/output/eval2")
        return 1

    source_dir = Path(sys.argv[1])
    dest_dir = Path(__file__).parent.parent / 'data'

    # Validate source
    source_annotations = source_dir / 'annotations.json'
    if not source_annotations.exists():
        print(f"Error: annotations.json not found in {source_dir}")
        return 1

    with open(source_annotations) as f:
        data = json.load(f)

    print(f"Found {len(data['annotations'])} annotations")

    # Create destination
    simulations_dir = dest_dir / 'simulations'
    simulations_dir.mkdir(parents=True, exist_ok=True)

    # Process each annotation
    updated_annotations = []
    copied_count = 0

    for ann in data['annotations']:
        run_id = ann['id']
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
        'annotations': updated_annotations,
    }

    output_path = dest_dir / 'annotations.json'
    with open(output_path, 'w') as f:
        json.dump(output_annotations, f, indent=2)

    print(f"\nWrote {len(updated_annotations)} annotations to {output_path}")
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
