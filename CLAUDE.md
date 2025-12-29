# Claude Context

## Project Goal

**The goal is NOT to accurately detect boom frames on ALL simulations.**

The actual goal is: **Produce high-quality animations with accurate boom detection for YouTube/social media.**

This means:
- We only care about HIGH-QUALITY simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we want accurate boom detection

**Current best (90 simulations, 3-seed evaluation)**:

*2-model pipeline (CNN + HGB):*
- Selective (sqrt/5): MAE 4.9 ± 1.2 frames at 22% coverage
- Balanced (sqrt/10): MAE 5.2 ± 1.2 frames at 37% coverage
- Permissive (sqrt/15): MAE 6.4 ± 0.5 frames at 43% coverage

*3-model pipeline (CNN + HGB + LSTM):*
- Selective (sqrt/5): MAE 5.6 ± 1.9 frames at 12% coverage
- Balanced (sqrt/12): MAE 5.2 ± 0.6 frames at 33% coverage
- Permissive (sqrt/15): MAE 5.1 ± 0.6 frames at 37% coverage

**Key finding**: The 3-model pipeline provides better accuracy at higher coverage levels. At ~37% coverage, 3-model achieves MAE 5.1 vs 2-model's 6.4 at 43%. For maximum selectivity, 2-model with scale=5 still provides the best MAE (4.9 at 22%).

**Key findings from comprehensive evaluation (Dec 2025)**:
- HGB alone: MAE 22.5 ± 0.6 frames (37% within 5 frames)
- LSTM alone: MAE 18.3 ± 1.6 frames (best individual model)
- CNN alone: MAE 20.2 ± 1.5 frames
- Caustic features do NOT improve the pipeline (no_caustic baseline is best)
- 3-model agreement (std-based) provides more stable confidence estimates

**Note**: Results using "oracle quality" (ground truth annotations) are NOT deployable. The above uses predicted quality, which is available at inference time.

## What is the Boom?

The boom is the moment of chaotic divergence: when nearly-identical pendulums suddenly spread apart due to sensitivity to initial conditions (the butterfly effect). The boom is the visually dramatic moment when chaos erupts. But the boom moment is NOT when pendulums slowly start diverging, the boom moment marks the explosion of chaotic divergence. There's a clear before/after. A boom with a high "boom quality score" is typically a boom that any human would trivially and objectively be able to find. Simulation with a low quality boom score (as annotated in the dataset) typically have a not-so-well-defined boom moment (it can be very ambiguous, there's not one clear "boom" moment, sometimes it can drags on, sometimes we can hesitate with different boom moments etc.). High quality boom typtically involves the pendulums slowly separating into 2+ distinct clusters (before boom) before accelerating and then meeting at high speed (boom moment), with caustic patterns emerging often at least a bit before the boom moment (and definitely right after it, since caustic-like patterns emerge from the chaotic divergence of such a simulation, and the boom moment marks the visually dramatic moment when true chaotic divergence begins).

## Key Patterns

### The Deployable Pipeline
```python
from boom_detection.deploy_pipeline import BoomDetectionPipeline

# 2-model pipeline (CNN + HGB) - default
pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb'),  # Default
    agreement_scale=5.0,          # Selective: lower = more selective
)

# 3-model pipeline (CNN + HGB + LSTM) - better at higher coverage
pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb', 'lstm'),
    primary_model='cnn',          # Use CNN prediction when accepted
    agreement_scale=15.0,         # Permissive: MAE 5.1 at 37% coverage
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
2. **Use CNN for final prediction** - CNN is more accurate than HGB or average
3. **3-model pipeline better at higher coverage** - MAE 5.1 at 37% vs 2-model's 6.4 at 43%
4. **2-model with scale=5 best for selectivity** - MAE 4.9 at 22% coverage
5. **LSTM is the best individual model** - MAE 18.3 (vs CNN 20.2, HGB 22.5)
6. **Accept threshold 0.60 compensates for overconfidence** - ECE improved from 0.15 to 0.06
7. **Caustic features do NOT improve the pipeline** - all tested formulas performed worse than no_caustic baseline

See experiment scripts in `scripts/` for detailed results:
- `scripts/evaluate_3model_pipeline.py` - 2-model vs 3-model comparison
- `scripts/evaluate_lstm.py` - Individual model (CNN/HGB/LSTM) evaluation
- `scripts/comprehensive_evaluation.py` - Full evaluation with caustic formula comparison
