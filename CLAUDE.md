# Claude Context

## 1. Project Goal

**The goal is NOT to accurately detect boom frames on ALL simulations.**

The actual goal is: **Produce high-quality animations with accurate boom detection for YouTube/social media.**

This means:
- We only care about HIGH-QUALITY simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we want accurate boom detection

## 2. What is the Boom?

The boom is the moment of chaotic divergence: when nearly-identical pendulums suddenly spread apart due to sensitivity to initial conditions (the butterfly effect). The boom is the visually dramatic moment when chaos erupts. But the boom moment is NOT when pendulums slowly start diverging, the boom moment marks the explosion of chaotic divergence. There's a clear before/after.

A boom with a high "boom quality score" is typically a boom that any human would trivially and objectively be able to find. Simulations with a low quality boom score typically have a not-so-well-defined boom moment (it can be very ambiguous, there's not one clear "boom" moment, sometimes it drags on, sometimes we can hesitate with different boom moments etc.).

High quality booms typically involve the pendulums slowly separating into 2+ distinct clusters (before boom) before accelerating and then meeting at high speed (boom moment), with caustic patterns emerging often at least a bit before the boom moment (and definitely right after it).

## 3. Quick Reference: Best Results

**90 simulations, 3-seed evaluation (42, 43, 44), PRODUCTION_CONFIG**

| Approach | Config | MAE | RMSE | Coverage | Notes |
|----------|--------|-----|------|----------|-------|
| **2-model best MAE** | sigmoid/s=3/t=0.70 | 2.53 ± 1.10 | 3.34 ± 1.68 | 4.1% | Best accuracy, low samples |
| **2-model stable** | sqrt/s=15/t=0.70 | 2.78 ± 0.90 | 3.75 ± 1.19 | 11.5% | Recommended |
| Quality-only | thresh=0.70 | 3.35 ± 1.39 | 4.49 ± 2.12 | 12.2% | Simplest |
| 2-model balanced | sqrt/s=15/t=0.60 | 4.86 ± 1.28 | 8.73 ± 3.40 | 30.4% | Good coverage |
| 3-model best | std/sqrt/s=8/t=0.55 | 5.18 ± 0.78 | 9.14 ± 2.37 | 34.4% | Not recommended |

**Key insight**: Testing combiner configurations is **FREE** - models train once, then combiners swap instantly. This is why we do parameter sweeps.

## 4. Detailed Results by Approach

### 4.1 Quality-Only Gating (QualityGatedCombiner)

Simplest approach: accept simulations based only on predicted quality score.

| Threshold | MAE | RMSE | Coverage |
|-----------|-----|------|----------|
| 0.70 | 3.35 ± 1.39 | 4.49 ± 2.12 | 12.2% |
| 0.65 | 4.50 ± 1.22 | 6.62 ± 1.81 | 31.5% |
| 0.60 | 5.86 ± 0.65 | 10.51 ± 0.50 | 45.2% |

```python
from boom_detection.combine import QualityGatedCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline

pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb'),
    combiner=QualityGatedCombiner(threshold=0.70),
)
```

### 4.2 2-Model Pipeline (CNN + HGB with ThresholdCombiner)

Uses model agreement to filter predictions. More complex but can achieve better accuracy.

**Best configurations by use case:**

| Use Case | Formula | Scale | Threshold | MAE | RMSE | Coverage |
|----------|---------|-------|-----------|-----|------|----------|
| Max accuracy | sigmoid | 3 | 0.70 | 2.53 ± 1.10 | 3.34 ± 1.68 | 4.1% |
| Stable accuracy | sqrt | 15 | 0.70 | 2.78 ± 0.90 | 3.75 ± 1.19 | 11.5% |
| Balanced | sqrt | 15 | 0.60 | 4.86 ± 1.28 | 8.73 ± 3.40 | 30.4% |
| High coverage | sqrt | 30 | 0.60 | 6.37 ± 0.53 | 12.60 ± 3.68 | 43.0% |

**Parameter effects:**
- **Scale**: Lower = more selective (tighter agreement required)
- **Threshold**: Higher = more selective (higher accept score required)
- **Formula**: sqrt is most stable; sigmoid best at extreme selectivity
- **score_function**: 'weighted' (default), 'min' (stricter), 'product' (even stricter)
- **quality_window**: Window around boom for quality features (default 25, 35 shows promise)
- **jitter_std**: Training noise for boom estimate (default 5, 10 shows promise)

```python
from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline

# Recommended: stable accuracy
pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb'),
    combiner=ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        threshold=0.70,
        primary_model='cnn',
    ),
)
```

### 4.3 3-Model Pipeline (CNN + HGB + LSTM)

**NOT RECOMMENDED** - Does not improve over 2-model with PRODUCTION_CONFIG.

Tested with `std` metric (robust to outliers) vs `range` metric:

| Metric | Scale | Threshold | MAE | Coverage | Notes |
|--------|-------|-----------|-----|----------|-------|
| std | sqrt/8 | 0.55 | 5.18 ± 0.78 | 34.4% | High variance |
| std | sqrt/5 | 0.50 | 5.29 ± 0.85 | 31.5% | |
| range | sqrt/15 | 0.50 | 5.07 ± 0.64 | 38.9% | |

**Why 3-model doesn't help:**
- With PRODUCTION_CONFIG (max_pendulums=2000), LSTM predictions are less stable
- The additional model adds noise rather than signal to the agreement metric
- 2-model is simpler and more accurate

### 4.4 Experimental: Specialized Models

**NEGATIVE RESULT** - Specialized models (trained on high-quality data only) perform worse.

Tested `hgb_0.5` (HGB trained only on quality >= 0.5 samples) vs baseline:

| Type | Threshold | MAE | Coverage | Notes |
|------|-----------|-----|----------|-------|
| baseline | 0.70 | 3.35 ± 1.39 | 12.2% | **Better** |
| hgb_0.5 | 0.70 | 5.03 ± 2.24 | 21.1% | Worse |
| hgb_0.5 | 0.75 | 2.68 ± 0.32 | 5.6% | Better MAE but very low coverage |

**Why specialized models don't help:**
- Training on ALL data provides beneficial regularization
- Quality gating already ensures predictions only on high-quality simulations
- At higher coverage levels, specialized models hurt performance

## 5. Standard Workflows

### Evaluate a Configuration
```bash
# Quick single-seed (development only - don't report!)
uv run python -m boom_detection.deploy_pipeline data --evaluate --quick \
    --acceptance-formula sqrt --scale 15

# Robust multi-seed (for reportable results)
uv run python -m boom_detection.deploy_pipeline data --evaluate \
    --acceptance-formula sqrt --scale 15 --threshold 0.70
```

### Sweep Combiner Configurations
```bash
# Quality-only sweep
uv run python scripts/sweep_quality_only.py data --seeds 42 43 44

# 2-model parameter sweep
uv run python scripts/sweep_2model.py data --seeds 42 43 44

# 3-model sweep (std vs range)
uv run python scripts/sweep_3model.py data --seeds 42 43 44
```

### Train and Deploy
```bash
# Train and save
uv run python -m boom_detection.deploy_pipeline data --train --output models/v1 \
    --acceptance-formula sqrt --scale 15

# Deploy as server
uv run python scripts/boom_server.py models/v1 --socket /tmp/boom.sock

# Batch inference
uv run python -m boom_detection.deploy_pipeline data --predict models/v1
```

## 6. Key Patterns & Code Examples

### CombinerExperiment - FREE Combiner Testing

**Critical concept**: Models train once, then combiners swap instantly.

```python
from boom_detection.combine import ThresholdCombiner, QualityGatedCombiner
from boom_detection.evaluation import CachedEvaluator
from boom_detection.deploy_pipeline import BoomDetectionPipeline

evaluator = CachedEvaluator(dataset, cache)

# Train models ONCE (slow)
experiment = evaluator.create_combiner_experiment(
    lambda: BoomDetectionPipeline(frame_models=('cnn', 'hgb')),
    seeds=[42, 43, 44],
)

# Iterate on combiners (FAST - no retraining!)
for scale in [5, 10, 15, 20]:
    combiner = ThresholdCombiner(disagreement_scale=scale, threshold=0.60)
    result = experiment.evaluate(combiner)
    print(f"scale={scale}: MAE {result.mean_metrics['selective_mae']:.2f}")
```

### FrameModelConfig - Unified Model Design

```python
from boom_detection.combine import FrameModelConfig

# Regular model (train on all data)
FrameModelConfig('hgb')           # equivalent to 'hgb' string

# Specialized model (train on high-quality data only)
FrameModelConfig('hgb', 0.5)      # hgb_0.5: trained on quality >= 0.5

# String shorthand
frame_models=('cnn', 'hgb', 'hgb_0.5')  # Parses automatically
```

### Robust Evaluation
```python
from boom_detection.evaluation import CachedEvaluator

evaluator = CachedEvaluator(dataset, cache)

# For selective models (use 3+ seeds!)
result = evaluator.cross_validate_selective(
    lambda: BoomDetectionPipeline(
        combiner=ThresholdCombiner(disagreement_scale=15.0, threshold=0.60),
    ),
    seeds=[42, 43, 44],
)
print(f"MAE: {result.mean_metrics['selective_mae']:.2f} ± {result.std_metrics['selective_mae']:.2f}")
print(f"Coverage: {result.mean_metrics['coverage']:.1%}")
```

### Selective Predictions
```python
from boom_detection.evaluation import SelectivePrediction, compute_selective_metrics

predictions = [SelectivePrediction.from_dict(p) for p in pipeline.predict(sim_ids, cache)]
metrics = compute_selective_metrics(predictions, true_booms, true_qualities)
print(f"MAE: {metrics['selective_mae']:.2f} at {metrics['coverage']:.1%} coverage")
```

## 7. Critical Requirements

### Reproducibility

**All results MUST be exactly reproducible.**

1. **Always use PRODUCTION_CONFIG**:
   ```python
   from boom_detection.features import PRODUCTION_CONFIG
   cache = FeatureCache(PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')
   ```

2. **Use `.feature_cache/no_caustic` directory**

3. **Use 3+ seeds** for all reported results (42, 43, 44 standard)

4. **Validate before reporting** - run sweep scripts to confirm

### Memory Management

Raw simulation data is ~35GB. Use two-phase pattern:

```python
import gc
from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, PRODUCTION_CONFIG

# PHASE 1: Extract features
dataset = load_dataset('data', verbose=False)  # ~35 GB
cache = FeatureCache(PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')
cache.extract_all(dataset, n_jobs=4)

# CRITICAL: Release raw data
dataset.release_simulation_data()
gc.collect()  # Now ~1 GB

# PHASE 2: Evaluate using cached features
result = evaluate(dataset, cache)
```

Or use `auto_release=True`:
```python
cache.extract_all(dataset, auto_release=True)  # Frees 35GB automatically
```

**Memory budget:**
| Phase | Memory |
|-------|--------|
| After load_dataset | ~35 GB |
| After release | ~1 GB |
| During evaluation | ~3-5 GB |

## 8. File Guide

### Core Modules (`src/boom_detection/`)

| File | Purpose |
|------|---------|
| `deploy_pipeline.py` | Production pipeline - CLI for evaluate/train/predict |
| `combine.py` | Combiner abstraction - ThresholdCombiner, QualityGatedCombiner |
| `evaluation.py` | Evaluation framework - CachedEvaluator, CombinerExperiment |
| `features.py` | Feature extraction + caching |
| `frame_models.py` | HistGBM classifier |
| `sequence_models.py` | CNN, LSTM (PyTorch) |
| `quality_models.py` | Quality prediction |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `sweep_quality_only.py` | Sweep QualityGatedCombiner thresholds |
| `sweep_2model.py` | Sweep 2-model ThresholdCombiner parameters |
| `sweep_3model.py` | Sweep 3-model with std/range metrics |
| `sweep_specialized.py` | Compare specialized vs baseline models |
| `sweep_extended.py` | Sweep weights, primary_model, score_function |
| `sweep_quality_params.py` | Sweep quality_window and jitter_std |
| `sweep_lstm_hgb.py` | Test LSTM+HGB pipeline (negative result) |
| `sweep_combined_best.py` | Test combined best parameters |
| `characterize_acceptance.py` | Validate documented results |
| `boom_server.py` | Deploy for real-time inference |

## 9. Do / Don't

### Do

- **Use 3+ seeds** for all reported results (mean ± std)
- **Use CachedEvaluator** and `CombinerExperiment` for evaluation
- **Use PRODUCTION_CONFIG** for all experiments
- **Use `HistGradientBoosting*`** (not `GradientBoosting*`) - 500x faster
- Split at simulation level (not frame level)

### Don't

- **Don't report single-seed results** - can vary by ±50%
- **Don't implement custom CV** - use unified framework
- **Don't use `metadata.json` boom_frame** - unreliable auto-detection
- **Don't use oracle quality** at inference
- **Don't commit** `.feature_cache/` or `runs/`

## Key Findings Summary

1. **2-model pipeline is optimal** - sqrt/s=15/t=0.70 gives MAE 2.78 at 11.5%
2. **Quality-only is competitive** - MAE 3.35 at 12.2%, much simpler
3. **3-model does NOT improve** - adds noise with PRODUCTION_CONFIG
4. **Specialized models don't help** - baseline + quality gating is best
5. **LSTM is best individual model** - MAE 18.3 (vs CNN 20.2, HGB 22.5)
6. **Caustic features don't help** - PRODUCTION_CONFIG excludes them
7. **LSTM+HGB pipeline is worse** - 50% worse MAE than CNN+HGB despite LSTM being best individual model

## Promising Results (Pending Validation)

These results show improvement but have < 10% coverage. Need validation on larger dataset.
See `EXPERIMENTS.md` for full details.

| Config | MAE | Coverage | Notes |
|--------|-----|----------|-------|
| quality_window=35, jitter_std=10 + sqrt/s=15/t=0.70 | 2.70 ± 0.34 | 8.1% | 3% MAE improvement |
| score_function='min', threshold=0.65 | 2.86 ± 1.42 | 9.3% | Alternative approach |
| primary_model='median' | 2.98 ± 0.41 | 10.0% | Slight improvement |

**Key insight**: Quality model parameters (quality_window, jitter_std) have significant impact.
Combiner transform (sqrt vs sigmoid) matters less.
