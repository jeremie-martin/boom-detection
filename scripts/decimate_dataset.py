#!/usr/bin/env python3
"""
Create a downsampled (decimated) copy of a boom detection dataset.

This selects a subset of pendulums and/or frames (no interpolation) and writes
a new dataset directory with the same annotations and metadata.

Usage:
    uv run python scripts/decimate_dataset.py /path/to/data /path/to/output --pendulums 2000
    uv run python scripts/decimate_dataset.py /path/to/data /path/to/output --frames 512
    uv run python scripts/decimate_dataset.py /path/to/data /path/to/output --pendulums 2000 --frames 512
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import zstandard

from boom_detection.loader import SimulationHeader, load_simulation


MAGIC = b"PNDL\x01\x00\x00\x00"
HEADER_STRUCT = struct.Struct("<3I 12d I 2Q 8x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decimate pendulum/frame counts for an existing dataset.",
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Dataset root containing annotations.json and simulations/",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output dataset root (will be created if missing).",
    )
    parser.add_argument(
        "-p",
        "--pendulums",
        type=int,
        help="Target number of pendulums per simulation.",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        help="Target number of frames per simulation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Allow simulations with fewer frames/pendulums than the target.",
    )
    parser.add_argument(
        "--skip-fewer",
        action="store_true",
        help="Skip simulations with fewer frames/pendulums than the target (default).",
    )
    return parser.parse_args()


def read_header(path: Path) -> SimulationHeader:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")
        header_data = f.read(136)

    values = HEADER_STRUCT.unpack(header_data)
    return SimulationHeader(
        format_version=values[0],
        pendulum_count=values[1],
        frame_count=values[2],
        duration_seconds=values[3],
        max_dt=values[4],
        gravity=values[5],
        length1=values[6],
        length2=values[7],
        mass1=values[8],
        mass2=values[9],
        initial_angle1=values[10],
        initial_angle2=values[11],
        initial_velocity1=values[12],
        initial_velocity2=values[13],
        angle_variation=values[14],
        floats_per_pendulum=values[15],
        uncompressed_size=values[16],
        compressed_size=values[17],
    )


def uniform_indices(total: int, target: int) -> np.ndarray:
    if target <= 0 or target > total:
        raise ValueError(f"Invalid target {target} for total {total}")
    if target == total:
        return np.arange(total, dtype=np.int64)

    indices = np.rint(np.linspace(0, total - 1, target)).astype(np.int64)
    if len(np.unique(indices)) != target:
        raise ValueError(
            f"Uniform decimation produced duplicate indices (total={total}, target={target})"
        )
    return indices


def write_simulation(path: Path, header: SimulationHeader, data: np.ndarray) -> None:
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim != 3 or data.shape[2] != 8:
        raise ValueError(f"Expected data shape (frames, pendulums, 8), got {data.shape}")

    payload = data.tobytes(order="C")
    compressor = zstandard.ZstdCompressor(level=3)
    compressed = compressor.compress(payload)

    header_values = (
        header.format_version,
        data.shape[1],
        data.shape[0],
        header.duration_seconds,
        header.max_dt,
        header.gravity,
        header.length1,
        header.length2,
        header.mass1,
        header.mass2,
        header.initial_angle1,
        header.initial_angle2,
        header.initial_velocity1,
        header.initial_velocity2,
        header.angle_variation,
        header.floats_per_pendulum,
        len(payload),
        len(compressed),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(HEADER_STRUCT.pack(*header_values))
        f.write(compressed)


def update_metadata(
    src_path: Path,
    dst_path: Path,
    pendulum_count: int | None,
    frame_count: int | None,
    frame_indices: np.ndarray | None,
) -> None:
    if not src_path.exists():
        return

    with open(src_path) as f:
        data = json.load(f)

    updated = False
    if isinstance(data, dict):
        simulation = data.get("simulation")
        if isinstance(simulation, dict) and "pendulum_count" in simulation:
            if pendulum_count is not None:
                simulation["pendulum_count"] = pendulum_count
                updated = True
        if isinstance(simulation, dict) and "total_frames" in simulation:
            if frame_count is not None:
                simulation["total_frames"] = frame_count
                updated = True

        results = data.get("results")
        if isinstance(results, dict) and "frames_completed" in results:
            if frame_count is not None:
                results["frames_completed"] = frame_count
                updated = True
        if isinstance(results, dict) and "boom_frame" in results and frame_indices is not None:
            old = int(round(results["boom_frame"]))
            results["boom_frame"] = map_frame_index(old, frame_indices)
            updated = True

        predictions = data.get("predictions")
        if isinstance(predictions, dict):
            boom_pred = predictions.get("boom")
            if isinstance(boom_pred, dict) and "frame" in boom_pred and frame_indices is not None:
                old = int(round(boom_pred["frame"]))
                boom_pred["frame"] = map_frame_index(old, frame_indices)
                updated = True

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if updated:
        with open(dst_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        shutil.copy2(src_path, dst_path)


def map_frame_index(old_frame: int, selected: np.ndarray) -> int:
    pos = int(np.searchsorted(selected, old_frame))
    if pos <= 0:
        return 0
    if pos >= len(selected):
        return len(selected) - 1
    before = selected[pos - 1]
    after = selected[pos]
    if abs(old_frame - before) <= abs(after - old_frame):
        return pos - 1
    return pos


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir

    if args.pendulums is None and args.frames is None:
        print("Error: must specify --pendulums and/or --frames")
        return 1
    if args.pendulums is not None and args.pendulums <= 0:
        print("Error: --pendulums must be a positive integer")
        return 1
    if args.frames is not None and args.frames <= 0:
        print("Error: --frames must be a positive integer")
        return 1
    if args.allow_fewer and args.skip_fewer:
        print("Error: --allow-fewer and --skip-fewer are mutually exclusive")
        return 1
    if not args.allow_fewer and not args.skip_fewer:
        args.skip_fewer = True

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return 1

    if source_dir.resolve() == output_dir.resolve():
        print("Error: output_dir must be different from source_dir")
        return 1

    annotations_path = source_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Error: annotations.json not found in {source_dir}")
        return 1

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        print(f"Error: output directory is not empty: {output_dir}")
        print("Use --overwrite to allow writing into it.")
        return 1

    with open(annotations_path) as f:
        annotations_data = json.load(f)

    annotations = annotations_data.get("annotations", [])
    print(f"Found {len(annotations)} annotations")

    simulations_dir = output_dir / "simulations"
    simulations_dir.mkdir(parents=True, exist_ok=True)

    updated_count = 0
    copied_count = 0
    skipped_count = 0
    output_annotations = []

    for i, ann in enumerate(annotations, start=1):
        run_id = ann.get("id")
        data_path = ann.get("data_path")
        if not run_id or not data_path:
            print(f"Error: invalid annotation entry: {ann}")
            return 1

        data_path_obj = Path(data_path)
        if data_path_obj.is_absolute():
            print(f"Error: data_path must be relative, got: {data_path_obj}")
            return 1

        src_sim_path = source_dir / data_path_obj
        if not src_sim_path.exists():
            print(f"Error: missing simulation file: {src_sim_path}")
            return 1

        dst_sim_path = output_dir / data_path_obj

        header = read_header(src_sim_path)
        if header.format_version != 2:
            print(f"Error: unsupported format version {header.format_version} in {src_sim_path}")
            return 1
        if header.floats_per_pendulum != 8:
            print(f"Error: unsupported floats_per_pendulum {header.floats_per_pendulum} in {src_sim_path}")
            return 1

        target_pendulums = args.pendulums
        target_frames = args.frames

        pendulum_needs_decimation = (
            target_pendulums is not None and header.pendulum_count > target_pendulums
        )
        frame_needs_decimation = (
            target_frames is not None and header.frame_count > target_frames
        )

        if target_pendulums is not None and header.pendulum_count < target_pendulums and not args.allow_fewer:
            msg = (
                f"{run_id} has {header.pendulum_count} pendulums, "
                f"less than target {target_pendulums}"
            )
            if args.skip_fewer:
                print(f"Skipping: {msg}")
                skipped_count += 1
                continue
            print(f"Error: {msg}")
            print("Use --allow-fewer to copy these simulations unchanged.")
            print("Use --skip-fewer to skip these simulations.")
            return 1

        if target_frames is not None and header.frame_count < target_frames and not args.allow_fewer:
            msg = (
                f"{run_id} has {header.frame_count} frames, "
                f"less than target {target_frames}"
            )
            if args.skip_fewer:
                print(f"Skipping: {msg}")
                skipped_count += 1
                continue
            print(f"Error: {msg}")
            print("Use --allow-fewer to copy these simulations unchanged.")
            print("Use --skip-fewer to skip these simulations.")
            return 1

        pendulum_msg = (
            f"{header.pendulum_count} -> {target_pendulums}"
            if pendulum_needs_decimation
            else f"{header.pendulum_count} (unchanged)"
        )
        frame_msg = (
            f"{header.frame_count} -> {target_frames}"
            if frame_needs_decimation
            else f"{header.frame_count} (unchanged)"
        )
        print(f"[{i}/{len(annotations)}] {run_id}: pendulums {pendulum_msg}, frames {frame_msg}")

        frame_indices = None
        ann_out = copy.deepcopy(ann)
        if frame_needs_decimation:
            frame_indices = uniform_indices(header.frame_count, target_frames)

            targets = ann_out.get("targets")
            if isinstance(targets, dict) and "boom" in targets:
                old_boom = int(round(targets["boom"]))
                mapped = map_frame_index(old_boom, frame_indices)
                targets["boom"] = mapped
            else:
                print(f"Warning: missing targets.boom for {run_id}, annotation not updated")

        if not pendulum_needs_decimation and not frame_needs_decimation:
            dst_sim_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_sim_path, dst_sim_path)
            update_metadata(
                src_sim_path.parent / "metadata.json",
                dst_sim_path.parent / "metadata.json",
                header.pendulum_count if target_pendulums is not None else None,
                header.frame_count if target_frames is not None else None,
                frame_indices,
            )
            copied_count += 1
            output_annotations.append(ann_out)
            continue

        try:
            sim = load_simulation(src_sim_path)
            data = sim.data

            if frame_needs_decimation:
                data = data[frame_indices, :, :]
            if pendulum_needs_decimation:
                pendulum_indices = uniform_indices(sim.pendulum_count, target_pendulums)
                data = data[:, pendulum_indices, :]

            new_header = replace(
                sim.header,
                pendulum_count=data.shape[1],
                frame_count=data.shape[0],
            )
            write_simulation(dst_sim_path, new_header, data)
            update_metadata(
                src_sim_path.parent / "metadata.json",
                dst_sim_path.parent / "metadata.json",
                data.shape[1] if target_pendulums is not None else None,
                data.shape[0] if target_frames is not None else None,
                frame_indices,
            )
        except Exception as exc:
            print(f"Warning: Failed to process {run_id}: {exc}")
            skipped_count += 1
            if dst_sim_path.exists():
                dst_sim_path.unlink()
            meta_dst = dst_sim_path.parent / "metadata.json"
            if meta_dst.exists():
                meta_dst.unlink()
            if dst_sim_path.parent.exists() and not any(dst_sim_path.parent.iterdir()):
                dst_sim_path.parent.rmdir()
            continue
        finally:
            if "sim" in locals():
                del sim

        updated_count += 1
        output_annotations.append(ann_out)

    # Write annotations.json with updated boom frame indices if needed.
    annotations_data["annotations"] = output_annotations
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations_data, f, indent=2)

    print(f"\nCompleted: {updated_count} decimated, {copied_count} copied, {skipped_count} skipped")
    print(f"Output dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
