# Claude Context

## Project Goal

**The goal is NOT to accurately detect boom frames on ALL simulations.**

The actual goal is: **Produce high-quality animations with accurate boom detection for YouTube/social media.**

This means:
- We only care about HIGH-QUALITY simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we want accurate boom detection

**Current best (90 simulations, 3-seed evaluation)**:
- Default (linear/10): MAE 7.3 ± 1.7 frames at 39% coverage
- Balanced (sqrt/15): MAE 4.9 ± 1.3 frames at 30% coverage
- Most selective (sqrt/5): MAE 3.4 ± 0.8 frames at 14% coverage

**Key findings from comprehensive evaluation (Dec 2025)**:
- HGB alone: MAE 22.5 ± 0.6 frames (37% within 5 frames)
- Caustic features do NOT improve the pipeline (no_caustic baseline is best)
- For HGB alone, entropy formula is slightly better (-0.5 MAE) but doesn't transfer to pipeline

**Note**: Results using "oracle quality" (ground truth annotations) are NOT deployable. The above uses predicted quality, which is available at inference time.

## What is the Boom?

The boom is the moment of chaotic divergence: when nearly-identical pendulums suddenly spread apart due to sensitivity to initial conditions (the butterfly effect). The boom is the visually dramatic moment when chaos erupts. But the boom moment is NOT when pendulums slowly start diverging, the boom moment marks the explosion of chaotic divergence. There's a clear before/after. A boom with a high "boom quality score" is typically a boom that any human would trivially and objectively be able to find. Simulation with a low quality boom score (as annotated in the dataset) typically have a not-so-well-defined boom moment (it can be very ambiguous, there's not one clear "boom" moment, sometimes it can drags on, sometimes we can hesitate with different boom moments etc.). High quality boom typtically involves the pendulums slowly separating into 2+ distinct clusters (before boom) before accelerating and then meeting at high speed (boom moment), with caustic patterns emerging often at least a bit before the boom moment (and definitely right after it, since caustic-like patterns emerge from the chaotic divergence of such a simulation, and the boom moment marks the visually dramatic moment when true chaotic divergence begins).

## Key Patterns

### The Deployable Pipeline
```python
from boom_detection.deploy_pipeline import BoomDetectionPipeline

# Train pipeline with different selectivity levels:
# - Default: accept_threshold=0.60, agreement_formula='linear', agreement_scale=10
# - Balanced: agreement_formula='sqrt', agreement_scale=15
# - Most selective: agreement_formula='sqrt', agreement_scale=5
pipeline = BoomDetectionPipeline(
    accept_threshold=0.60,
    agreement_formula='sqrt',  # 'linear' or 'sqrt'
    agreement_scale=15.0,      # Default: 10 for linear, 15 for sqrt
    calibrate_quality=True,
)
pipeline.fit(sim_ids, boom_frames, qualities, cache)

# Predict - returns SelectivePrediction objects
result = pipeline.predict_one(features)

if result.accepted:
    boom_frame = result.boom_frame  # Uses CNN prediction (more accurate than HGB)
    confidence = result.confidence  # Combined confidence score

# Save/load for deployment
pipeline.save(Path("models/v1"))
pipeline = BoomDetectionPipeline.from_pretrained(Path("models/v1"))
```

### Robust Evaluation (IMPORTANT!)
```python
# Always use multi-seed evaluation for honest results
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
    lambda: BoomDetectionPipeline(accept_threshold=0.60, agreement_formula='sqrt'),
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

| File | Purpose |
|------|---------|
| `deploy_pipeline.py` | **Start here** - production pipeline |
| `evaluation.py` | **Unified evaluation framework** - CachedEvaluator, SelectivePrediction, RunArtifact |
| `features.py` | Feature extraction + caching |
| `frame_models.py` | HistGBM classifier |
| `sequence_models.py` | CNN, LSTM, Transformer |
| `quality_models.py` | Quality prediction |
| `run_baselines.py` | Baseline comparison (uses CachedEvaluator) |
| `pipeline.py` | Multi-stage pipeline components |
| `ensemble.py` | Adaptive ensemble |

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

1. **Model agreement is the best confidence signal** - better than quality prediction alone
2. **Use CNN, not HGB or average** - CNN is more accurate and has lower variance
3. **Sqrt agreement formula with scale=15 is a good balance** - MAE 4.9 at 30% coverage
4. **Accept threshold 0.60 compensates for overconfidence** - ECE improved from 0.15 to 0.06
5. **Frame-level HistGBM classifier is best baseline** - MAE 22.5±0.6 at 37% within-5
6. **Caustic features do NOT improve the pipeline** - all 5 tested formulas performed worse than no_caustic baseline
7. **Entropy formula slightly helps HGB alone** (-0.5 MAE) but doesn't translate to pipeline improvement

See experiment scripts in `scripts/` for detailed results:
- `scripts/comprehensive_evaluation.py` - Full evaluation with caustic formula comparison
- `scripts/evaluate_caustic_formulas.py` - Focused caustic formula analysis
