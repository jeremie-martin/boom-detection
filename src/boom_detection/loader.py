"""
Data loader utilities for the boom detection dataset.

Usage:
    from boom_detection import load_dataset, load_simulation

    # Load all annotations and simulation data
    dataset = load_dataset('data')

    # Or load a single simulation
    sim = load_simulation('data/simulations/run_xxx/simulation_data.bin')
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import zstandard


# Field indices for simulation data
X1, Y1, X2, Y2, TH1, TH2, W1, W2 = range(8)

FIELD_NAMES = ['x1', 'y1', 'x2', 'y2', 'th1', 'th2', 'w1', 'w2']


@dataclass
class SimulationHeader:
    """Header information from simulation_data.bin"""
    format_version: int
    pendulum_count: int
    frame_count: int
    duration_seconds: float
    max_dt: float
    gravity: float
    length1: float
    length2: float
    mass1: float
    mass2: float
    initial_angle1: float
    initial_angle2: float
    initial_velocity1: float
    initial_velocity2: float
    angle_variation: float
    floats_per_pendulum: int
    uncompressed_size: int
    compressed_size: int


@dataclass
class Simulation:
    """Complete simulation data."""
    header: SimulationHeader
    data: np.ndarray  # Shape: (frames, pendulums, 8)

    @property
    def frame_count(self) -> int:
        return self.header.frame_count

    @property
    def pendulum_count(self) -> int:
        return self.header.pendulum_count

    def get_frame(self, frame: int) -> np.ndarray:
        """Get all pendulum states at a specific frame. Shape: (pendulums, 8)"""
        return self.data[frame]

    def get_positions(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Get tip positions (x2, y2) at a frame. Returns (x_array, y_array)"""
        return self.data[frame, :, X2], self.data[frame, :, Y2]

    def get_angles(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Get angles (th1, th2) at a frame. Returns (th1_array, th2_array)"""
        return self.data[frame, :, TH1], self.data[frame, :, TH2]

    def get_velocities(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Get angular velocities (w1, w2) at a frame. Returns (w1_array, w2_array)"""
        return self.data[frame, :, W1], self.data[frame, :, W2]


@dataclass
class Annotation:
    """Single simulation annotation."""
    id: str
    data_path: str
    boom_frame: int
    boom_quality: float
    notes: str = ""


@dataclass
class Dataset:
    """Complete dataset with annotations and loaded simulations."""
    annotations: list[Annotation]
    simulations: dict[str, Simulation]  # id -> Simulation
    root: Path

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self) -> Iterator[tuple[Simulation, int, float]]:
        """Iterate over (simulation, boom_frame, boom_quality) tuples."""
        for ann in self.annotations:
            yield self.simulations[ann.id], ann.boom_frame, ann.boom_quality

    def get_sample(self, idx: int) -> tuple[Simulation, int, float]:
        """Get (simulation, boom_frame, boom_quality) for index."""
        ann = self.annotations[idx]
        return self.simulations[ann.id], ann.boom_frame, ann.boom_quality

    def get_by_id(self, id: str) -> tuple[Simulation, int, float]:
        """Get (simulation, boom_frame, boom_quality) by run ID."""
        for ann in self.annotations:
            if ann.id == id:
                return self.simulations[id], ann.boom_frame, ann.boom_quality
        raise KeyError(f"ID not found: {id}")


def load_simulation(path: str | Path) -> Simulation:
    """
    Load simulation data from a binary file.

    Args:
        path: Path to simulation_data.bin file

    Returns:
        Simulation object with header and data array
    """
    path = Path(path)

    with open(path, 'rb') as f:
        # Read and validate magic
        magic = f.read(8)
        if magic != b'PNDL\x01\x00\x00\x00':
            raise ValueError(f"Invalid magic bytes: {magic!r}")

        # Read header (136 bytes after magic)
        header_data = f.read(136)

        # Parse header: 3 uint32, 12 float64, 1 uint32, 2 uint64, 8 bytes padding
        values = struct.unpack('<3I 12d I 2Q 8x', header_data)

        header = SimulationHeader(
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

        if header.format_version != 2:
            raise ValueError(f"Unsupported format version: {header.format_version}")

        if header.floats_per_pendulum != 8:
            raise ValueError(f"Expected 8 floats per pendulum, got {header.floats_per_pendulum}")

        # Read compressed payload
        compressed = f.read(header.compressed_size)

    # Decompress
    dctx = zstandard.ZstdDecompressor()
    decompressed = dctx.decompress(compressed, max_output_size=header.uncompressed_size)

    # Convert to numpy array and reshape
    data = np.frombuffer(decompressed, dtype=np.float32).copy()
    data = data.reshape(header.frame_count, header.pendulum_count, 8)

    return Simulation(header=header, data=data)


def load_annotations(path: str | Path) -> list[Annotation]:
    """
    Load annotations from annotations.json.

    Args:
        path: Path to annotations.json

    Returns:
        List of Annotation objects
    """
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    if data.get('version') != 2:
        raise ValueError(f"Unsupported annotations version: {data.get('version')}")

    annotations = []
    for entry in data['annotations']:
        annotations.append(Annotation(
            id=entry['id'],
            data_path=entry['data_path'],
            boom_frame=int(entry['targets']['boom']),
            boom_quality=float(entry['targets']['boom_quality']),
            notes=entry.get('notes', ''),
        ))

    return annotations


def load_dataset(
    root: str | Path,
    load_simulations: bool = True,
    verbose: bool = True,
    max_samples: int | None = None,
) -> Dataset:
    """
    Load the complete dataset.

    Args:
        root: Root directory containing annotations.json and simulations/
        load_simulations: Whether to load raw simulation data (memory intensive)
        verbose: Print progress
        max_samples: Limit to first N samples (for quick testing)

    Returns:
        Dataset object with annotations and optionally loaded data
    """
    root = Path(root)

    # Load annotations
    annotations = load_annotations(root / 'annotations.json')

    if max_samples is not None and max_samples < len(annotations):
        annotations = annotations[:max_samples]
        if verbose:
            print(f"Limited to {max_samples} samples")

    if verbose:
        print(f"Loaded {len(annotations)} annotations")

    simulations = {}

    if load_simulations:
        failed = []
        for i, ann in enumerate(annotations):
            sim_path = root / ann.data_path
            if verbose:
                print(f"Loading simulation {i+1}/{len(annotations)}: {ann.id}")
            try:
                simulations[ann.id] = load_simulation(sim_path)
            except Exception as e:
                if verbose:
                    print(f"  WARNING: Failed to load {ann.id}: {e}")
                failed.append(ann.id)

        if failed:
            # Remove annotations for failed simulations
            annotations = [a for a in annotations if a.id not in failed]
            if verbose:
                print(f"Skipped {len(failed)} corrupted simulations, {len(annotations)} remaining")

    return Dataset(
        annotations=annotations,
        simulations=simulations,
        root=root,
    )


if __name__ == '__main__':
    import sys

    # Demo usage
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'data'

    print(f"Loading dataset from: {path}")

    # Load just annotations first (fast)
    annotations = load_annotations(Path(path) / 'annotations.json')
    print(f"\nFound {len(annotations)} annotated simulations")

    # Show statistics
    boom_frames = [a.boom_frame for a in annotations]
    boom_qualities = [a.boom_quality for a in annotations]

    print(f"\nBoom frame statistics:")
    print(f"  Min: {min(boom_frames)}")
    print(f"  Max: {max(boom_frames)}")
    print(f"  Mean: {np.mean(boom_frames):.1f}")
    print(f"  Std: {np.std(boom_frames):.1f}")

    print(f"\nBoom quality statistics:")
    print(f"  Min: {min(boom_qualities):.2f}")
    print(f"  Max: {max(boom_qualities):.2f}")
    print(f"  Mean: {np.mean(boom_qualities):.2f}")
    print(f"  Std: {np.std(boom_qualities):.2f}")

    # Load one simulation as example
    print(f"\nLoading first simulation...")
    first = annotations[0]
    sim = load_simulation(Path(path) / first.data_path)

    print(f"\nSimulation info:")
    print(f"  ID: {first.id}")
    print(f"  Frames: {sim.frame_count}")
    print(f"  Pendulums: {sim.pendulum_count}")
    print(f"  Boom frame: {first.boom_frame}")
    print(f"  Boom quality: {first.boom_quality}")
    print(f"  Data shape: {sim.data.shape}")
    print(f"  Memory: {sim.data.nbytes / 1e6:.1f} MB")
