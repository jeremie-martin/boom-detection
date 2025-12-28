# Claude Context

## What This Is

Predict the "boom" frame in double pendulum simulations. See README.md for full context.

**Current best**: CNN at MAE 17.1 frames (target: <5)

**Key finding**: Boom quality strongly predicts error. High-quality booms have MAE ~11, low-quality ~31.

## What is the Boom?

The boom is **NOT** just when pendulums diverge. It's when **two groups converge**:

1. **Before boom**: Pendulums separate into 2+ distinct clusters
2. **At boom**: Clusters CONVERGE at a single point (collision effect)
3. **After boom**: Caustic patterns emerge, explosion

See `BOOM_PHENOMENON.md` for detailed visual analysis and experimental findings.

**Key insight**: Direct convergence detection is hard. ML models work better by learning complex feature combinations.

## Key Patterns

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

### Adding New Features
Add to `features.py`, then update:
1. Add extraction function
2. Add to `FeatureConfig` (with `include_X` flag)
3. Add to `FEATURE_GROUPS` for names
4. Add to `transform()` method

### Resolution Invariance
All features must aggregate over pendulums (axis=1) so they work regardless of pendulum count.

## File Guide

| File | When to modify |
|------|----------------|
| `features.py` | Adding new features |
| `frame_models.py` | Adding sklearn-based models |
| `sequence_models.py` | Adding PyTorch models |
| `quality_models.py` | Quality prediction models |
| `pipeline.py` | Multi-stage quality-aware pipeline |
| `convergence.py` | Convergence/bimodality detection (experimental) |
| `changepoint.py` | CUSUM/BOCPD detectors |
| `ensemble.py` | Model ensembling |
| `run_baselines.py` | Adding evaluation utilities |
| `evaluation.py` | Changing metrics |

## Do

- Use `FeatureCache` with `cache_dir` for persistence
- Use `quick_cv()` for fast experiments
- Use `HistGradientBoosting*` (not `GradientBoosting*`) - 500x faster
- Split at simulation level (not frame level) to prevent data leakage

## Don't

- Don't use `metadata.json` boom_frame (unreliable auto-detection)
- Don't load full dataset for quick experiments (use `quick_cv`)
- Don't commit `.feature_cache/` (in .gitignore)
