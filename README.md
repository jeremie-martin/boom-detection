# Boom Detection

Predict the "boom" frame in chaotic double pendulum simulations - the moment when pendulums visually diverge.

## Current Results

| Model | MAE (frames) | Notes |
|-------|--------------|-------|
| Baseline (variance threshold) | 31.2 | Classical approach |
| HistGBM (frame-level) | 17.0 | Best non-neural |
| **CNN (sequence)** | **14.3** | Best overall |
| Target | <5 | ~83ms at 60fps |

## Quick Start

```bash
# Install
uv sync --extra ml

# Run best model (CNN)
uv run python -c "
from boom_detection import load_dataset, FeatureCache, FeatureConfig
from boom_detection.sequence_models import CNNClassifier, SequenceTrainer
from boom_detection.run_baselines import cross_validate_cached

dataset = load_dataset('data')
config = FeatureConfig(max_pendulums=2000, include_caustic=True, include_rolling=True)
cache = FeatureCache(config, cache_dir='.feature_cache')
cache.extract_all(dataset)

model = SequenceTrainer(CNNClassifier(cache[dataset.annotations[0].id].shape[1]))
result = cross_validate_cached(dataset, cache, model)
print(f'MAE: {result[\"metrics\"][\"mae\"]:.1f}')
"
```

## Fast Iteration

After first run, features are cached to disk. Use `quick_cv` to skip dataset loading:

```python
from boom_detection import FeatureCache, FeatureConfig
from boom_detection.run_baselines import quick_cv

cache = FeatureCache(
    FeatureConfig(max_pendulums=2000, include_caustic=True, include_rolling=True),
    cache_dir='.feature_cache'
)
# Loads from disk in ~0.2s instead of ~30s
result = quick_cv(your_model, cache)
```

## Project Structure

```
src/boom_detection/
├── loader.py          # Load simulations and annotations
├── features.py        # Feature extraction + caching
├── evaluation.py      # Metrics and cross-validation
├── baselines.py       # Simple baseline predictors
├── frame_models.py    # HistGBM classifier/regressor
└── sequence_models.py # CNN, LSTM, Transformer (PyTorch)
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `FeatureCache` | Extract & cache features (use `cache_dir` for disk persistence) |
| `FrameLevelClassifier` | Best non-neural model (`model='hist_gbm'`) |
| `SequenceTrainer` | Train CNN/LSTM on sequences |

## The Problem

Thousands of pendulums start with nearly identical initial conditions. At some point they rapidly diverge - the "boom". This is a perceptual phenomenon: the moment a human would say "it explodes".

**Inputs**: ~1000 frames × ~10000 pendulums × 8 features (positions, angles, velocities)

**Output**: Single frame number (integer)

**Constraint**: Must work on low-resolution "probe" simulations (~200 frames × ~1000 pendulums)

## Data

```bash
# Dataset not in git (~20GB). Copy from source:
cp -r /path/to/double-pendulum/output/eval2 data
```

- 49 valid simulations (1 corrupted)
- Boom frames: 204-933
- ~1000 frames, ~10000+ pendulums each

## Development

```bash
uv sync --extra ml    # Install with ML deps
uv run pytest         # Run tests
uv run python -m boom_detection.run_baselines data  # Run all baselines
```

## Approach Summary

1. **Features**: Aggregate pendulum statistics per frame (resolution-invariant)
   - Statistical: variance, IQR, skewness, kurtosis
   - Caustic: angular distribution metrics (Gini, coverage)
   - Temporal: rolling windows, derivatives

2. **Frame-level learning**: Each frame predicts before/after boom → find crossing

3. **Sequence models**: CNN/LSTM on full feature sequence

See the plan file for detailed methodology: `.claude/plans/tender-popping-adleman.md`
