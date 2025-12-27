# Claude Context for Boom Detection

## Project Overview

Dataset and tools for predicting critical transition frames ("boom") in chaotic double pendulum simulations.

## The Problem

Given time-series data of ~10,000+ pendulums over ~1,000 frames, predict:

1. **Boom frame**: The frame where chaotic divergence becomes visually significant (MAE target: <5 frames)
2. **Boom quality**: Subjective rating 0-1 of visual impressiveness (MAE target: <0.15)

Current best classical approaches achieve ~23 frame MAE, which is insufficient.

## Critical Constraint: Resolution Invariance

Predictions must be **agnostic to pendulum count and frame count**.

**Why**: In production, fast "probe" simulations (~1,000 pendulums, ~200 frames) are run first for filtering. Only passing probes get full rendering (~10,000+ pendulums). The model trained on high-resolution data must generalize to low-resolution probes.

The training data intentionally varies pendulum counts (10k-15k) across samples. During training, subsampling from full data can simulate lower resolutions.

## Project Structure

```
boom-detection/
├── data/
│   ├── annotations.json       # Ground truth labels (50 samples)
│   └── simulations/           # Binary simulation data (~12GB)
│       └── run_YYYYMMDD_HHMMSS/
│           ├── simulation_data.bin
│           └── metadata.json
├── src/boom_detection/
│   ├── __init__.py
│   └── loader.py              # Data loading utilities
├── pyproject.toml             # UV/pip project config
├── README.md                  # Full problem description
└── DATA_FORMAT.md             # Binary format specification
```

## Key Files

| File | Purpose |
|------|---------|
| `src/boom_detection/loader.py` | Load simulation data and annotations |
| `data/annotations.json` | Ground truth boom frames and quality scores |
| `DATA_FORMAT.md` | Binary format spec for simulation_data.bin |

## Data Format

Each simulation contains ~10k-15k pendulums tracked over ~1000 frames.

**Per pendulum per frame** (8 float32 values):
- `x1, y1`: First joint position
- `x2, y2`: Tip position
- `th1, th2`: Arm angles (radians)
- `w1, w2`: Angular velocities (rad/s)

**Total per simulation**: ~300MB uncompressed, ~80MB ZSTD-compressed

## Quick Usage

```python
from boom_detection import load_dataset, load_simulation

# Load full dataset
dataset = load_dataset('data')
for sim, boom_frame, boom_quality in dataset:
    # sim.data has shape (frames, pendulums, 8)
    pass

# Load single simulation
sim = load_simulation('data/simulations/run_xxx/simulation_data.bin')
x, y = sim.get_positions(frame=500)  # tip positions at frame 500
```

## Development Commands

```bash
# Install dependencies
uv sync

# Run with optional ML dependencies
uv sync --extra ml

# Run tests
uv run pytest

# Run loader demo
uv run python -m boom_detection.loader data
```

## Dataset Statistics

- 50 annotated simulations
- Pendulum counts: varies (10,000-15,000 per simulation)
- Frame counts: varies (~1,000 per simulation, 15-18s duration)
- Boom frames: 204-933 (mean ~470)
- Boom quality: 0.10-0.92 (mean ~0.52)

Note: Dataset is not committed to git (~20GB). Obtain separately.

## Evaluation

Use k-fold cross-validation (k=5 or k=10) with only 50 samples.

```python
from sklearn.model_selection import KFold
import numpy as np

def evaluate(model, dataset):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    for train_idx, val_idx in kf.split(dataset.annotations):
        # train and evaluate
        pass
    return np.mean(maes)
```

## What NOT to Do

- Don't rely on `metadata.json` boom_frame - it's from unreliable auto-detection
- Don't use metrics.csv files - they were from failed classical approaches
- Always use `annotations.json` for ground truth
