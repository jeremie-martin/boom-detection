# Claude Context

## Project Goal

**The goal is NOT to accurately detect boom frames on ALL simulations.**

The actual goal is: **Produce high-quality animations with accurate boom detection for YouTube/social media.**

This means:
- We only care about HIGH-QUALITY simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we want accurate boom detection

**Current best (90 simulations, 3-seed evaluation, PRODUCTION_CONFIG)**:

*Quality-gated with HGB - NEW BEST:*
- **Most selective (thresh=0.70)**: MAE 3.06 ± 1.96 at 12.2% coverage
- Very selective (thresh=0.72): MAE 2.33 ± 2.16 at 7.0% coverage

*2-model pipeline (CNN + HGB) with ThresholdCombiner:*
- Most selective (scale=5): MAE 3.38 ± 0.83 frames at 13.7% coverage
- Balanced (scale=15): MAE 4.86 ± 1.28 frames at 30.4% coverage

*3-model pipeline (CNN + HGB + LSTM) - NOT recommended:*
- With std/5: MAE 5.57 ± 1.93 at 11.9% (worse than 2-model)

**Key findings from comprehensive evaluation (Dec 2025)**:
- HGB alone: MAE 22.5 ± 0.6 frames (37% within 5 frames)
- LSTM alone: MAE 18.3 ± 1.6 frames (best individual model)
- CNN alone: MAE 20.2 ± 1.5 frames
- Caustic features do NOT improve the pipeline (no_caustic baseline is best)
- 3-model (CNN+HGB+LSTM) does NOT outperform 2-model with PRODUCTION_CONFIG
- QualityGatedCombiner(0.7) is surprisingly competitive with ThresholdCombiner
- **Specialized models trained only on high-quality data perform WORSE** (see below)

**Note**: Results using "oracle quality" (ground truth annotations) are NOT deployable. The above uses predicted quality, which is available at inference time.

## What is the Boom?

The boom is the moment of chaotic divergence: when nearly-identical pendulums suddenly spread apart due to sensitivity to initial conditions (the butterfly effect). The boom is the visually dramatic moment when chaos erupts. But the boom moment is NOT when pendulums slowly start diverging, the boom moment marks the explosion of chaotic divergence. There's a clear before/after. A boom with a high "boom quality score" is typically a boom that any human would trivially and objectively be able to find. Simulation with a low quality boom score (as annotated in the dataset) typically have a not-so-well-defined boom moment (it can be very ambiguous, there's not one clear "boom" moment, sometimes it can drags on, sometimes we can hesitate with different boom moments etc.). High quality boom typtically involves the pendulums slowly separating into 2+ distinct clusters (before boom) before accelerating and then meeting at high speed (boom moment), with caustic patterns emerging often at least a bit before the boom moment (and definitely right after it, since caustic-like patterns emerge from the chaotic divergence of such a simulation, and the boom moment marks the visually dramatic moment when true chaotic divergence begins).

## Standard Workflows

### 1. Evaluate a Configuration (Quick Check)
```bash
# Quick single-seed (for development only - don't report these!)
uv run python -m boom_detection.deploy_pipeline data --evaluate --quick \
    --acceptance-formula sqrt --scale 5

# Robust multi-seed (for reportable results)
uv run python -m boom_detection.deploy_pipeline data --evaluate \
    --acceptance-formula sqrt --scale 15 --threshold 0.60
```

### 2. Experiment with Combiner Configurations (Gold Standard)
```bash
# Validate documented results and explore parameter space
uv run python scripts/characterize_acceptance.py data --validate --sweep

# Just validate documented configurations
uv run python scripts/characterize_acceptance.py data --validate

# Comprehensive sweep with output
uv run python scripts/characterize_acceptance.py data --sweep --output runs/characterization
```

The `characterize_acceptance.py` script uses `CombinerExperiment.sweep()` to explore the parameter space **without retraining models**. This is the correct way to test new combiner configurations.

### 3. Train and Deploy a Model
```bash
# Step 1: Train and save (uses all data)
uv run python -m boom_detection.deploy_pipeline data --train --output models/v1 \
    --acceptance-formula sqrt --scale 15

# Step 2a: Low-latency server (for C++/real-time integration)
uv run python scripts/boom_server.py models/v1 --socket /tmp/boom.sock

# Step 2b: Batch inference
uv run python -m boom_detection.deploy_pipeline data --predict models/v1
```

### 4. Test New Features/Models
```bash
# Use CachedEvaluator in Python for new experiments
# See scripts/evaluate_3model_pipeline.py as template
```

## Key Patterns

### The Deployable Pipeline
```python
from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline

# Train pipeline with different selectivity levels using Combiner API:
# - Most selective: scale=5 -> MAE 3.4 at 14% coverage
# - Balanced: scale=10 -> MAE 4.9 at 22% coverage
# - Permissive: scale=15 -> MAE 4.9 at 30% coverage
pipeline = BoomDetectionPipeline(
    combiner=ThresholdCombiner(
        agreement_transform='sqrt',   # 'sqrt', 'linear', 'sigmoid', 'quadratic'
        disagreement_scale=5.0,       # Lower = more selective
        threshold=0.60,               # Accept if score >= threshold
        primary_model='cnn',          # Use CNN prediction when accepted
    ),
    frame_models=('cnn', 'hgb'),      # Default: 2-model pipeline
)

pipeline.fit(sim_ids, boom_frames, qualities, cache)

# Predict - returns SelectivePrediction objects
result = pipeline.predict_one(features)

if result.accepted:
    boom_frame = result.boom_frame      # Uses primary_model (CNN by default)
    accept_score = result.accept_score  # Combined confidence score (0-1)

# Save/load for deployment
pipeline.save(Path("models/v1"))
pipeline = BoomDetectionPipeline.from_pretrained(Path("models/v1"))
```

### Robust Evaluation (IMPORTANT!)
```python
# Always use multi-seed evaluation for honest results
from boom_detection.combine import ThresholdCombiner
from boom_detection.evaluation import CachedEvaluator

evaluator = CachedEvaluator(dataset, cache)

# For non-selective models:
result = evaluator.cross_validate(
    lambda: MyModel(),  # Factory function!
    seeds=[42, 43, 44, 45, 46],  # 5 seeds
)
print(f"MAE: {result.mean_metrics['mae']:.2f} ± {result.std_metrics['mae']:.2f}")

# For selective (abstaining) models like BoomDetectionPipeline:
result = evaluator.cross_validate_selective(
    lambda: BoomDetectionPipeline(
        combiner=ThresholdCombiner(
            disagreement_scale=15.0,
            threshold=0.60,
        ),
    ),
    seeds=[42, 43, 44, 45, 46],
)
print(f"Selective MAE: {result.mean_metrics['selective_mae']:.2f}")
print(f"Coverage: {result.mean_metrics['coverage']:.1%}")
```

### Fast Iteration (Development)
```python
# Quick single-seed for development (but report multi-seed for final results!)
result = evaluator.quick_evaluate(lambda: MyModel(), seed=42)
print(f"MAE: {result['mae']:.1f}")  # Single seed - don't report this!
```

### Adding a New Model
Models must implement `fit(sim_ids, boom_frames, cache)` and `predict(sim_ids, cache)`:
```python
class MyModel:
    def fit(self, sim_ids: list[str], boom_frames: np.ndarray, cache: FeatureCache):
        for sim_id, boom in zip(sim_ids, boom_frames):
            features = cache[sim_id]  # (frames, n_features)
            # Train...

    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray:
        return np.array([self._predict_one(cache[sid]) for sid in sim_ids])
```

### Resolution Invariance
All features must aggregate over pendulums (axis=1) so they work regardless of pendulum count.

### Selective Predictions (Unified Framework)
```python
from boom_detection.evaluation import SelectivePrediction, compute_selective_metrics, RunArtifact

# Convert pipeline output to canonical format
predictions = [SelectivePrediction.from_dict(p) for p in pipeline.predict(sim_ids, cache)]

# Compute selective metrics (coverage, selective_mae, etc.)
metrics = compute_selective_metrics(predictions, true_booms, true_qualities)
print(f"Selective MAE: {metrics['selective_mae']:.2f} at {metrics['coverage']:.1%} coverage")

# Save run for reproducibility
artifact = RunArtifact.create(config, predictions, true_booms, true_qualities, sim_ids)
artifact.save(Path("runs/my_experiment"))
```

## File Guide

### Core Modules (`src/boom_detection/`)

| File | Purpose |
|------|---------|
| `deploy_pipeline.py` | **Production pipeline** - CLI for evaluate/train/predict |
| `combine.py` | **Combiner abstraction** - ThresholdCombiner, utility functions |
| `evaluation.py` | **Evaluation framework** - CachedEvaluator, CombinerExperiment, SelectivePrediction |
| `features.py` | Feature extraction + caching |
| `frame_models.py` | HistGBM classifier |
| `sequence_models.py` | CNN, LSTM, Transformer (PyTorch) |
| `quality_models.py` | Quality prediction |

### Scripts (`scripts/`)

| Script | When to Use |
|--------|-------------|
| `characterize_acceptance.py` | **Experiment with combiner configurations** - validates results, sweeps parameters |
| `boom_server.py` | **Deploy for real-time inference** - Unix socket server for C++ integration |
| `evaluate_3model_pipeline.py` | Compare 2-model vs 3-model configurations |
| `evaluate_lstm.py` | Evaluate individual models (CNN/HGB/LSTM) |
| `comprehensive_evaluation.py` | Full evaluation with caustic formula comparison |

### Which Tool for Which Task?

| Task | Tool |
|------|------|
| Evaluate a specific configuration | `deploy_pipeline.py --evaluate` |
| Experiment with combiner configurations | `characterize_acceptance.py` |
| Train and save a model | `deploy_pipeline.py --train` |
| Run inference in production | `boom_server.py` or `deploy_pipeline.py --predict` |
| Test a new frame model | Create script using `CachedEvaluator` |
| Test a new feature | Modify `features.py`, then evaluate |

## Memory Management

### Critical: Two-phase pattern for evaluation scripts

Raw simulation data is **~35GB for 90 simulations**. Use this two-phase pattern:

```python
import gc
from boom_detection.loader import load_dataset
from boom_detection.features import FeatureCache, FeatureConfig
from boom_detection.logging_config import log_memory_usage

# PHASE 1: Extract features (requires raw data)
dataset = load_dataset('data', verbose=False)
log_memory_usage("after loading")  # ~35 GB

# Extract all configs you need FIRST
for name, config in configs:
    cache = FeatureCache(config=config, cache_dir=f'.feature_cache/{name}')
    cache.extract_all(dataset, n_jobs=4)  # Cached to disk
    del cache
    gc.collect()

# CRITICAL: Release raw data immediately after extraction
dataset.release_simulation_data()
gc.collect()
log_memory_usage("after release")  # Should be ~1 GB

# PHASE 2: Evaluate using disk caches (no raw data needed)
for name, config in configs:
    cache = FeatureCache(config=config, cache_dir=f'.feature_cache/{name}')
    sim_ids = [a.id for a in dataset.annotations]
    cache.load_from_disk(sim_ids, verbose=False)  # Load from disk, not extract!

    result = evaluate(dataset, cache)

    del cache
    gc.collect()
```

### Simple single-config case

For simple scripts testing one config:

```python
dataset = load_dataset('data', verbose=False)
cache = FeatureCache(config=config, cache_dir='.feature_cache/my_config')
cache.extract_all(dataset, auto_release=True)  # auto_release frees 35GB automatically

# Now dataset.simulations is empty, but features are cached
result = evaluate(dataset, cache)
```

### Memory budget

| Phase | Memory |
|-------|--------|
| Initial | ~0.5 GB |
| After load_dataset | ~35 GB |
| After release_simulation_data | ~1 GB |
| Per feature cache (in memory) | ~1.5 GB |
| During model training | +2-3 GB |

**Target**: Scripts should stay under 5GB after releasing simulation data.

## Reproducibility Requirements (CRITICAL)

**All experiment results MUST be exactly reproducible.** Not "more or less" - EXACTLY identical.

### Why This Matters
- We use fixed seeds (42, 43, 44) for all experiments
- Same parameters + same seeds = identical results every time
- If results differ, something is seriously wrong and must be fixed

### Mandatory Requirements

1. **Always use PRODUCTION_CONFIG** for all experiments:
   ```python
   from boom_detection.features import PRODUCTION_CONFIG
   cache = FeatureCache(PRODUCTION_CONFIG, cache_dir='.feature_cache/no_caustic')
   ```

2. **Use `.feature_cache/no_caustic` directory** - this ensures consistent feature extraction

3. **Validate before reporting** - run `characterize_acceptance.py --validate` to confirm documented results still hold

4. **If results don't match documented values**:
   - STOP and investigate immediately
   - Check feature config (PRODUCTION_CONFIG vs FeatureConfig())
   - Check cache directory (`.feature_cache/no_caustic` vs `.feature_cache`)
   - Check random seeds
   - DO NOT proceed until root cause is found and fixed

### What Went Wrong Before (Dec 2025)

The 3-model MAE 3.27 result was obtained with `FeatureConfig()` (all pendulums) instead of `PRODUCTION_CONFIG` (max_pendulums=2000). This caused:
- Irreproducible results between scripts
- False claim that 3-model beats 2-model
- Hours of debugging to find root cause

**Lesson**: Always use PRODUCTION_CONFIG. Always verify reproducibility.

## Do

- **Always use multi-seed evaluation** - report mean ± std, not single-seed results
- **Use the unified evaluation framework** - `CachedEvaluator`, `SelectivePrediction`, `compute_selective_metrics`
- Use `CachedEvaluator.cross_validate()` for robust evaluation of non-selective models
- Use `SelectivePrediction` for selective (abstaining) predictors
- Use `RunArtifact` to save experiment results for reproducibility
- Use `FeatureCache` with `cache_dir` for persistence
- Use `HistGradientBoosting*` (not `GradientBoosting*`) - 500x faster
- Split at simulation level (not frame level) to prevent data leakage

## Don't

- **Don't report single-seed results** - they can vary by ±50% due to small sample size
- **Don't implement custom CV/evaluation code** - use the unified framework
- Don't use `metadata.json` boom_frame (unreliable auto-detection)
- Don't use oracle quality at inference (annotations not available)
- Don't commit `.feature_cache/` or `runs/`

## Key Findings

1. **QualityGatedModelCombiner with HGB is NEW BEST** - MAE 3.06 at 12.2% beats ThresholdCombiner
2. **HGB beats CNN and median for primary model** - Use `primary_model='hgb'` for best accuracy
3. **2-model pipeline is optimal** - 3-model does NOT improve over 2-model with PRODUCTION_CONFIG
4. **scale=5 with ThresholdCombiner achieves MAE 3.38 at 13.7%** - Good alternative to quality-only
5. **LSTM is the best individual model** - MAE 18.3 (vs CNN 20.2, HGB 22.5)
6. **Accept threshold 0.60-0.70 works best** - ECE improved from 0.15 to 0.06
7. **Caustic features do NOT improve the pipeline** - all tested formulas performed worse than no_caustic baseline
8. **Agreement helps at moderate coverage** - ThresholdCombiner beats quality-only at 30% coverage
9. **PRODUCTION_CONFIG is required** - Use `max_pendulums=2000, include_caustic=False` for consistent results

## Acceptance Formula Characterization

Comprehensive sweep of 480 configurations (4 formulas × 20 scales × 6 thresholds).
Results in `runs/characterization_full/`.

### MAE vs Coverage Tradeoff

The relationship is **monotonic but non-linear** with diminishing returns at extreme selectivity:

| Coverage | Best MAE | Configuration | Notes |
|----------|----------|---------------|-------|
| 5-10% | 2.6 ± 0.7 | sigmoid/s=4/t=0.70 | Statistically fragile (~5-9 samples) |
| 10-15% | 2.8 ± 0.9 | sqrt/s=18/t=0.70 | Good accuracy, reasonable reliability |
| 15-20% | 3.0 ± 0.8 | sigmoid/s=8/t=0.70 | |
| 25-30% | 4.6 ± 1.4 | linear/s=15/t=0.70 | |
| 35-40% | 4.9 ± 1.3 | sqrt/s=18/t=0.60 | |
| 50-60% | 6.7 ± 1.6 | quadratic/s=40/t=0.70 | |

### Parameter Effects

**Scale** (disagreement tolerance):
- Lower scale = more selective (requires tighter agreement)
- sqrt/s=5 vs s=15: MAE 3.4→4.9 but coverage 14%→30%

**Threshold** (accept score cutoff):
- Higher threshold = more selective
- t=0.70 gives best MAE but lowest coverage
- t=0.60 is a good balance for production

**Formula comparison** (all similar when tuned):
- **sqrt**: Most stable across coverage range (recommended)
- **sigmoid**: Best at extreme selectivity
- **linear**: Simple, slightly worse at low coverage
- **quadratic**: Similar to sqrt

### 3-Model vs 2-Model (PRODUCTION_CONFIG)

**Important**: 3-model does NOT improve over 2-model when using PRODUCTION_CONFIG (max_pendulums=2000).

| Configuration | MAE | Coverage | Notes |
|---------------|-----|----------|-------|
| 2-model range/scale=5 | **3.38 ± 0.83** | 13.7% | **BEST** |
| QualityGated(0.7) | 3.35 ± 1.39 | 12.2% | Simplest, competitive |
| 3-model std/scale=5 | 5.57 ± 1.93 | 11.9% | NOT recommended |
| 3-model range/scale=30 | 5.17 ± 0.68 | 34.8% | Worse than 2-model |

**Why 3-model doesn't help with PRODUCTION_CONFIG**:
- With subsampled pendulums (max_pendulums=2000), LSTM predictions are less stable
- The additional model adds noise rather than signal to the agreement metric
- 2-model pipeline is simpler and more accurate

**Note**: Previous results showing 3-model MAE 3.27 were obtained with `FeatureConfig()` (all pendulums, no subsampling). These are not reproducible with PRODUCTION_CONFIG.

### Recommended Configurations

| Use Case | Config | MAE | Coverage |
|----------|--------|-----|----------|
| **Maximum accuracy** | **QualityGatedModelCombiner(thresh=0.70, primary='hgb')** | **3.06** | **~12%** |
| Very selective | QualityGatedModelCombiner(thresh=0.72, primary='hgb') | 2.33 | ~7% |
| ThresholdCombiner best | ThresholdCombiner(scale=5) | 3.38 | ~14% |
| **Balanced (default)** | ThresholdCombiner(scale=15) | 4.86 | ~30% |
| High coverage | ThresholdCombiner(scale=40) | ~7.9 | ~50% |

**To use the new best config:**
```python
from boom_detection.combine import QualityGatedModelCombiner

pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb'),  # Still need both for training quality model
    combiner=QualityGatedModelCombiner(
        threshold=0.70,
        primary_model='hgb',  # Use HGB prediction when accepted
    ),
)
```

### Specialized Model Experiment (Negative Result)

**Hypothesis**: Models trained only on high-quality data (quality >= threshold) would make more accurate predictions by avoiding learning from ambiguous booms.

**Result**: The hypothesis is **NOT supported**. Specialized models perform WORSE or equal to baselines.

#### Multi-Threshold HGB Experiment (Dec 2025)

Tested training HGB at different quality thresholds (0.4, 0.5, 0.6, 0.7) to see if more training data helps:

**At 12.2% coverage (accept=0.7) - Most Selective:**
| Training Threshold | Training Samples | MAE | Notes |
|--------------------|-----------------|-----|-------|
| hgb_0.5 | ~40/fold | **3.18 ± 1.15** | Best, lowest variance |
| hgb_0.6 | ~35/fold | 3.18 ± 1.61 | Tied, higher variance |
| hgb_0.4 | ~45/fold | 3.29 ± 1.59 | Slightly worse |
| **baseline** | all | **3.35 ± 1.39** | Simpler, competitive |
| hgb_0.7 | ~22/fold | 3.73 ± 1.66 | **WORST** - too few samples |

**At 45.2% coverage (accept=0.6) - Baseline is Best:**
| Training Threshold | MAE | Notes |
|--------------------|-----|-------|
| **baseline** | **5.86 ± 0.65** | **BEST** |
| hgb_0.6 | 5.96 ± 0.19 | |
| hgb_0.5 | 6.21 ± 0.21 | |
| hgb_0.4 | 6.28 ± 0.17 | |
| hgb_0.7 | 6.45 ± 0.91 | **WORST** |

**Key Insights from Multi-Threshold Experiment:**
1. **Training threshold 0.7 is consistently worst** - Too few samples (~20-26/fold)
2. **Training at 0.5 or 0.6 gives marginal improvement (~5%) at high selectivity** - MAE 3.18 vs 3.35 baseline
3. **At higher coverage, baseline is better** - Specialized models don't generalize to lower-quality samples
4. **The improvement is marginal** - Only ~0.17 frames at 12.2% coverage
5. **hgb_0.5 has lower variance** than hgb_0.6 (±1.15 vs ±1.61) - more stable

#### Original Single-Threshold Experiment

Previous experiment with threshold 0.7 only:

| Configuration | MAE | Coverage | Notes |
|---------------|-----|----------|-------|
| **baseline QualityGated(0.7)** | **3.35 ± 1.39** | 12.2% | **RECOMMENDED** |
| baseline ThresholdCombiner(scale=5) | 3.38 ± 0.83 | 13.7% | Close second |
| specialized_hgb_0.7 + QualityGated(0.7) | 3.73 ± 1.66 | 12.2% | WORSE |
| specialized_lstm + QualityGated(0.7) | 3.84 ± 1.36 | 12.2% | WORSE |
| specialized_cnn + QualityGated(0.7) | 6.54 ± 5.06 | 12.2% | Very poor |

**Why specialized models don't help**:
- Training on ALL data (including ambiguous cases) provides beneficial regularization
- Even with more training data (threshold 0.5: ~40/fold vs 0.7: ~22/fold), improvement is marginal
- At higher coverage levels, specialized models actually hurt performance
- The existing quality gating already ensures we only make predictions on high-quality simulations

**Conclusion**: Don't use specialized models. The baseline approach of training on all data with quality prediction for acceptance is simpler and nearly as accurate. The marginal gain (~5% at 12.2% coverage) doesn't justify the complexity.

See `scripts/experiment_specialized_model.py` and `scripts/experiment_multi_threshold.py` for detailed results.

---

See experiment scripts in `scripts/` for detailed results:
- `scripts/characterize_acceptance.py` - Comprehensive formula sweeps
- `scripts/experiment_combiner_ablations.py` - Combiner ablations (std vs range, baselines, weights)
- `scripts/experiment_quality_gated_model.py` - **QualityGatedModelCombiner experiments** (NEW BEST)
- `scripts/experiment_combined_best.py` - Combined best configuration testing
- `scripts/experiment_3model_optimization.py` - 3-model parameter sweeps
- `scripts/experiment_specialized_model.py` - **Specialized model experiment** (negative result)
- `scripts/evaluate_lstm.py` - Individual model (CNN/HGB/LSTM) evaluation
