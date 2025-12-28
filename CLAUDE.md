# Claude Context

## Project Goal

**The goal is NOT to accurately detect boom frames on ALL simulations.**

The actual goal is: **Produce high-quality animations with accurate boom detection for YouTube/social media.**

This means:
- We only care about HIGH-QUALITY simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we need MAE close to 5 frames

**Current best**: MAE 4.0 with 27% acceptance rate (deployable pipeline)

## What is the Boom?

The boom is when **two groups of pendulums converge**:
1. **Before boom**: Pendulums separate into 2+ distinct clusters
2. **At boom**: Clusters CONVERGE at a single point (collision effect)
3. **After boom**: Caustic patterns emerge, explosion

See `docs/EXPERIMENT_HISTORY.md` for detailed experimental findings.

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
    boom_frame = result['boom_frame']  # Use HGB prediction
```

### Fast Iteration
```python
# Use disk-cached features to skip 30s dataset loading
cache = FeatureCache(config, cache_dir='.feature_cache')
result = quick_cv(model, cache)  # ~0.2s to load features
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

## File Guide

| File | Purpose |
|------|---------|
| `deploy_pipeline.py` | **Start here** - production pipeline |
| `features.py` | Feature extraction + caching |
| `frame_models.py` | HistGBM classifier |
| `sequence_models.py` | CNN, LSTM, Transformer |
| `quality_models.py` | Quality prediction |
| `pipeline.py` | Multi-stage pipeline components |
| `run_baselines.py` | Baseline comparison |
| `evaluation.py` | Metrics |
| `ensemble.py` | Adaptive ensemble |

## Do

- Use `FeatureCache` with `cache_dir` for persistence
- Use `quick_cv()` for fast experiments
- Use `HistGradientBoosting*` (not `GradientBoosting*`) - 500x faster
- Split at simulation level (not frame level) to prevent data leakage

## Don't

- Don't use `metadata.json` boom_frame (unreliable auto-detection)
- Don't use oracle quality at inference (annotations not available)
- Don't commit `.feature_cache/`

## Key Findings

1. **Model agreement is the best confidence signal** - better than quality prediction alone
2. **Use HGB, not average** - when models agree, HGB alone is more accurate
3. **Quality threshold 0.55 works** - higher than initially expected
4. **CNN benefits from all features**, HistGBM benefits from feature selection

See `docs/RESULTS.md` for detailed results.
