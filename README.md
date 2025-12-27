# Boom Detection

A labeled dataset and tools for predicting critical transition frames in chaotic double pendulum simulations.

## Problem Statement

Given time-series data from a double pendulum simulation (positions, angles, and velocities of ~10,000+ pendulums over ~1,000 frames), predict:

1. **Primary task**: The "boom frame" - the specific frame where chaotic divergence becomes visually significant
2. **Secondary task**: The "boom quality" - a subjective rating (0-1) of how visually impressive the boom appears

### Success Criteria

- **Boom frame prediction**: Mean Absolute Error < 5 frames (currently best known: ~23 frames)
- **Boom quality prediction**: Mean Absolute Error < 0.15 on 0-1 scale (currently: ~0.22)

At 60 FPS video output, 5 frames = ~83ms precision, suitable for music synchronization.

## Quick Start

```bash
# Install dependencies
uv sync

# Test loading the dataset
uv run python -c "from boom_detection.loader import load_dataset; print(load_dataset('data'))"
```

## Domain Background

### Double Pendulum Chaos

A double pendulum consists of two rigid arms connected end-to-end, free to swing. The system exhibits deterministic chaos: nearly identical initial conditions diverge exponentially over time.

The simulation tracks thousands of pendulums with slightly different starting angles (±0.1°). Initially they move together. At some point - the "boom" - they rapidly diverge, creating visually striking patterns.

### What is the "Boom"?

The boom is a perceptual phenomenon with these characteristics:

- **Before boom**: Pendulums move coherently, appearing as a thick line or narrow band
- **At boom**: Rapid visual expansion as pendulums diverge
- **After boom**: Chaotic, space-filling patterns

The boom is **not** simply:
- The frame of maximum variance (often too late)
- The first frame of any divergence (often too early)
- A fixed threshold crossing (varies by simulation)

It is the moment a human observer would identify as "when it explodes" - a gestalt perception of sudden, dramatic change.

### Why is this hard?

1. **Non-linear dynamics**: The transition is gradual then sudden (sigmoid-like), but the inflection point varies
2. **Multi-scale features**: Relevant information spans local (individual pendulum motion) to global (distribution shape)
3. **Perceptual subjectivity**: The "boom" is defined by human perception, not a physical quantity
4. **Variable timing**: Boom can occur anywhere from frame 200 to frame 900+ depending on initial conditions

### Invariance Requirements

The prediction must be **agnostic to simulation resolution**:

- **Pendulum count**: Training data has 10,000-15,000 pendulums, but inference may run on simulations with as few as 1,000 pendulums
- **Frame count**: Training data has ~1,000 frames, but inference may run on shorter simulations (~200 frames)

**Why this matters**: In production, a fast low-resolution "probe" simulation is run first to predict boom timing and quality. Only simulations that pass filtering criteria proceed to full high-resolution rendering (which is computationally expensive). The probe uses fewer pendulums and frames for speed.

The training dataset intentionally varies pendulum counts across samples to encourage learning resolution-agnostic features. An alternative approach during training is to subsample from the full data to simulate lower-resolution inputs.

## Dataset Structure

```
boom-detection/
├── data/                          # NOT committed (~20GB, obtain separately)
│   ├── annotations.json           # Labels for all simulations
│   └── simulations/
│       └── run_YYYYMMDD_HHMMSS/
│           ├── simulation_data.bin    # Raw pendulum states (compressed)
│           └── metadata.json          # Simulation parameters
├── src/boom_detection/
│   └── loader.py                  # Data loading utilities
└── pyproject.toml
```

### Obtaining the Dataset

The `data/` directory is not committed to git. To obtain the dataset, copy from the double-pendulum project:

```bash
uv run python scripts/prepare_dataset.py /path/to/double-pendulum/output/eval2
```

### annotations.json

```json
{
  "version": 2,
  "target_defs": {
    "boom": "frame",
    "boom_quality": "score"
  },
  "annotations": [
    {
      "id": "run_20251226_110631",
      "data_path": "simulations/run_20251226_110631/simulation_data.bin",
      "targets": {
        "boom": 367,
        "boom_quality": 0.2
      },
      "notes": ""
    }
  ]
}
```

### Labels

| Label | Type | Range | Description |
|-------|------|-------|-------------|
| `boom` | int | 200-950 | Frame number of the boom |
| `boom_quality` | float | 0.0-1.0 | Subjective visual quality rating |

### Dataset Statistics (50 simulations)

| Property | Range |
|----------|-------|
| Pendulum count | 10,000 - 15,000 (varies per simulation) |
| Frame count | ~1,000 (varies per simulation) |
| Duration | 15 - 18 seconds |
| Boom frame | 204 - 933 (mean ~470) |
| Boom quality | 0.10 - 0.92 (mean ~0.52) |

## Data Format

### Raw Simulation Data (simulation_data.bin)

Binary file with ZSTD-compressed pendulum states.

**Header** (144 bytes, little-endian):

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | char[8] | magic | `"PNDL\x01\x00\x00\x00"` |
| 8 | uint32 | format_version | Always 2 |
| 12 | uint32 | pendulum_count | Number of pendulums (~10,000-15,000) |
| 16 | uint32 | frame_count | Total frames (~1,000-1,200) |
| 20 | float64 | duration_seconds | Simulation duration (typically 18s) |
| ... | ... | ... | (see DATA_FORMAT.md for full spec) |

**Payload** (ZSTD-compressed):

Each pendulum state is 8 float32 values (32 bytes):

| Index | Field | Description |
|-------|-------|-------------|
| 0 | x1 | First joint x position |
| 1 | y1 | First joint y position |
| 2 | x2 | Second joint (tip) x position |
| 3 | y2 | Second joint (tip) y position |
| 4 | th1 | First arm angle (radians) |
| 5 | th2 | Second arm angle (radians) |
| 6 | w1 | First arm angular velocity (rad/s) |
| 7 | w2 | Second arm angular velocity (rad/s) |

### Loading Example

```python
from boom_detection.loader import load_simulation, load_dataset

# Load single simulation
sim = load_simulation('data/simulations/run_xxx/simulation_data.bin')
print(f"Shape: {sim.data.shape}")  # (frames, pendulums, 8)

# Get tip positions at frame 500
x, y = sim.get_positions(frame=500)

# Load full dataset
dataset = load_dataset('data')
for sim, boom_frame, boom_quality in dataset:
    print(f"Boom at frame {boom_frame}, quality {boom_quality}")
```

## Evaluation

```python
import numpy as np

def evaluate_predictions(predictions, ground_truth):
    """Compute Mean Absolute Error in frames."""
    errors = np.abs(np.array(predictions) - np.array(ground_truth))
    return {
        'mae': np.mean(errors),
        'median_ae': np.median(errors),
        'within_5_frames': np.mean(errors <= 5),
        'within_10_frames': np.mean(errors <= 10),
    }
```

With only 50 samples, use leave-one-out or k-fold cross-validation (k=5 or k=10).

## Prior Approaches (for context)

The following classical approaches have been tried with limited success:

1. **Threshold crossing on hand-crafted metrics**: Best MAE ~23 frames
2. **Peak detection on various metrics**: MAE 25-35 frames
3. **Derivative analysis**: Similar performance to threshold methods

These results suggest the problem may require approaches that capture more complex temporal patterns or non-linear relationships in the data.

## License

This dataset is provided for research purposes.
