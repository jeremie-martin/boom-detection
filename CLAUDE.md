# Claude Context

## Project Goal

**The goal is NOT to accurately detect boom frames on ALL simulations.**

The actual goal is: **Produce high-quality animations with accurate boom detection for YouTube/social media.**

This means:
- We only care about HIGH-QUALITY simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we want accurate boom detection

**Current best**: MAE 6.4 ± 0.5 frames with ~35% acceptance rate (robust 5-seed evaluation)

**Note**: Results using "oracle quality" (ground truth annotations) are NOT deployable. The above uses predicted quality, which is available at inference time.

## What is the Boom?

The boom is the moment of chaotic divergence: when nearly-identical pendulums suddenly spread apart due to sensitivity to initial conditions (the butterfly effect). The boom is the visually dramatic moment when chaos erupts. But the boom moment is NOT when pendulums slowly start diverging, the boom moment marks the explosion of chaotic divergence. There's a clear before/after. A boom with a high "boom quality score" is typically a boom that any human would trivially and objectively be able to find. Simulation with a low quality boom score (as annotated in the dataset) typically have a not-so-well-defined boom moment (it can be very ambiguous, there's not one clear "boom" moment, sometimes it can drags on, sometimes we can hesitate with different boom moments etc.). High quality boom typtically involves the pendulums slowly separating into 2+ distinct clusters (before boom) before accelerating and then meeting at high speed (boom moment), with caustic patterns emerging often at least a bit before the boom moment (and definitely right after it, since caustic-like patterns emerge from the chaotic divergence of such a simulation, and the boom moment marks the visually dramatic moment when true chaotic divergence begins).

## Key Patterns

### The Deployable Pipeline
```python
from boom_detection.deploy_pipeline import BoomDetectionPipeline

pipeline = BoomDetectionPipeline(
    agreement_threshold=5,
    quality_threshold=0.55,
)
pipeline.fit(sim_ids, boom_frames, qualities, cache)
result = pipeline.predict_one(features)

if result['accepted']:
    boom_frame = result['boom_frame']  # Uses CNN prediction (more accurate than HGB)
```

### Robust Evaluation (IMPORTANT!)
```python
# Always use multi-seed evaluation for honest results
from boom_detection.evaluation import CachedEvaluator

evaluator = CachedEvaluator(dataset, cache)
result = evaluator.cross_validate(
    lambda: MyModel(),  # Factory function!
    seeds=[42, 43, 44, 45, 46],  # 5 seeds
)
print(f"MAE: {result.mean_metrics['mae']:.2f} ± {result.std_metrics['mae']:.2f}")
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
2. **Use CNN, not HGB or average** - CNN is more accurate (MAE 7.1 vs 11.0) and has lower variance
3. **Quality threshold 0.55 works** - higher than initially expected
4. **CNN benefits from all features**, HistGBM benefits from feature selection

See `docs/RESULTS.md` for detailed results.
