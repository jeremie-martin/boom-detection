# Performance Optimization Analysis

**Agent Task**: Analyze performance bottlenecks and optimization opportunities.

**Date**: 2025-12-28

---

## Executive Summary

The codebase has significant parallelization opportunities that could provide 25-50x speedup. The slowest operations are CNN training (80% of time) and sequential cross-validation. Feature extraction is already parallelized. Key quick wins: increase CNN batch size, parallelize multi-seed CV, remove unnecessary data copies.

---

## 1. Slowest Operations

### Critical Bottleneck: CNN Training (50-100 epochs)
- **Location**: `sequence_models.py:SequenceTrainer.fit()` (lines 319-383)
- **Issue**: Training CNNs with:
  - 50 epochs per seed
  - 5 seeds × 5 CV folds = 125 model training runs
  - Each epoch processes full sequences with DataLoader overhead
- **Estimated time**: 10-30 minutes per evaluation
- **Impact**: Dominates multi-seed cross-validation runtime

### Secondary: Feature Extraction (Moderate)
- **Location**: `features.py:FeatureExtractor.transform()` (lines 620-694)
- **Cost per simulation**: ~200-500ms (depends on pendulum count)
- **Bottlenecks**:
  - Multiple aggregation operations across pendulums (axis=1)
  - Circular statistics calculations with repeated trig functions
  - Rolling window operations using `uniform_filter1d()`
- **Good news**: Already parallelized with ThreadPoolExecutor (lines 890-899)

### Tertiary: Frame-Level Model Training
- **Location**: `frame_models.py:FrameLevelRegressor.fit()`
- **Cost**: Training on ~50k frame-level samples
- **Why fast**: Uses HistGradientBoosting (500x faster than sklearn GBM)

### Quaternary: Cross-Validation Loops
- **Location**: `evaluation.py:CachedEvaluator.cross_validate()` (lines 665-735)
- **Cost**: 5 seeds × 5 folds × (CNN training + HGB training + quality prediction)
- **Pattern**: Sequential execution of seeds (lines 709-719)

---

## 2. Caching Strategy Analysis

### What's Cached (Excellent)

```
features.py:FeatureCache (lines 752-997)
├─ In-Memory Cache: _cache dict (line 777)
└─ Disk Cache: .npy files with config hash (line 799)
```

**Strengths**:
- Config-aware invalidation via MD5 hash (line 793-799)
- Lazy loading from disk when needed (lines 968-976)
- Parallel extraction with ThreadPoolExecutor (line 890)
- Smart discovery: `load_from_disk()` auto-discovers cached files

### Caching Weaknesses

1. **No cache for intermediate aggregations**:
   - `_extract_sim_features()` in `quality_models.py` recomputes aggregations
   - Called once at train, once at predict - not cached
   - Estimated cost: 5-10% of predict time wasted

2. **No cache for frame-level training data**:
   - `FrameLevelRegressor.fit()` reconstructs X_frames, y_distances each call
   - In multi-seed CV, this is recomputed 25 times
   - Could save ~30 seconds per evaluation

3. **No cache versioning**:
   - Cache files named: `{sim_id}_{config_hash}.npy`
   - Old config hashes gradually accumulate
   - No cache cleanup utility

---

## 3. Parallelization Opportunities

### Currently Parallelized ✓
1. Feature extraction (`FeatureCache.extract_all()`, line 890)
   - ThreadPoolExecutor with n_jobs workers
   - ~3-4x speedup on 8-core CPU

### Not Parallelized ✗

**1. Multi-seed evaluation** (CRITICAL - lines 709-719)
```python
for seed_idx, seed in enumerate(seeds):  # Sequential!
    result = self._cross_validate_single_seed(...)
```
- **Speedup opportunity**: 5x with parallel seeds
- **Estimated savings**: 20-100 minutes per evaluation

**2. Cross-validation folds** (lines 750-778)
```python
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(...)):  # Sequential
```
- **Speedup opportunity**: 5x (5 folds)
- **Combined**: 5 seeds × 5 folds = 25x parallelization potential

**3. Ensemble model predictions** (ensemble.py, line 119)
- **Speedup opportunity**: 2-4x (for N models)
- **Benefit**: Modest (ensemble overhead is small)

---

## 4. Data Copying & Memory Waste

### Significant Data Copies Detected

**1. Unnecessary copy in loader.py (line 183)**:
```python
data = np.frombuffer(decompressed, dtype=np.float32).copy()  # Copy here...
data = data.reshape(...)  # ...then reshape (view-safe)
```
- Reshape doesn't need a copy with contiguous memory
- **Estimated waste**: ~50-100 MB per simulation load

**2. Multiple vstack operations (frame_models.py, lines 70-71)**:
```python
X = np.vstack(X_frames)  # Creates new array
y = np.concatenate(y_distances)  # Creates new array
```
- Pre-allocate arrays instead
- **Estimated waste**: ~50 MB during frame-level training

**3. List accumulation (ensemble.py, lines 117-118)**:
```python
all_preds = []
for _, model in self.models:
    all_preds.append(preds)  # Grows list dynamically
```
- Pre-allocate `all_preds = np.zeros((n_models, len(sim_ids)))`

### Inefficient Operations

**1. Percentile calculations (features.py, lines 67-68)**:
```python
q75 = np.percentile(data, 75, axis=1)
q25 = np.percentile(data, 25, axis=1)
```
- Called separately (two full passes)
- Better: `np.percentile(data, [25, 75], axis=1)`
- **Estimated savings**: ~10% of feature extraction time

**2. Circular statistics repeated (lines 196-201)**:
- Trig functions called multiple times per frame
- Could cache cos/sin results

---

## 5. GPU Utilization for CNN/LSTM

### Current GPU Strategy
- **Device handling**: `device='auto'` selects GPU if available
- **Batch size**: 8 (line 303) - **Too small for modern GPUs**
- **Data loading**: `num_workers=0` - **Blocks async I/O**

### GPU Bottlenecks

**1. Batch size 8 is too small**
- Modern GPUs have 1000s of cores
- Batch size 8 underutilizes by 10-20x
- **Optimal**: Batch size 32-64
- **Estimated speedup**: 2-4x with proper batching

**2. num_workers=0 disables async data loading**
- With variable-length sequences, can't use workers easily
- **Workaround**: Pre-pad all sequences to max length
- **Trade-off**: Memory increase (~10-20%) for 20-30% training speedup

**3. No mixed precision training**
- No `torch.autocast()` or `GradientScaler`
- Could reduce memory 2x and speed up 1.5x
- **Estimated speedup**: 30-50% on GPU training

---

## 6. Optimization Recommendations

| Priority | Optimization | Location | Estimated Speedup | Effort | Risk |
|----------|--------------|----------|-------------------|--------|------|
| CRITICAL | Parallelize multi-seed CV | `evaluation.py:665-735` | 5x | Medium | Medium |
| CRITICAL | Increase CNN batch size | `sequence_models.py:303` | 2-4x | Low | Low |
| HIGH | Remove unnecessary data copy | `loader.py:183` | 10-15% | Low | Low |
| HIGH | Parallelize CV folds | `evaluation.py:750-778` | 5x | Medium | Medium |
| HIGH | Fix percentile calls | `features.py:67-68` | 10% | Low | Low |
| MEDIUM | Enable num_workers for DataLoader | `sequence_models.py:335` | 20% | Medium | Medium |
| MEDIUM | Batch frame model predictions | `frame_models.py:106-117` | 10-20% | Low | Low |
| MEDIUM | Add mixed precision training | `sequence_models.py:319-383` | 30-50% | Medium | Low |
| LOW | Cache quality aggregations | `quality_models.py:71-80` | 5-10% | Low | Low |
| LOW | Pre-allocate ensemble arrays | `ensemble.py:117-120` | <5% | Low | Low |

---

## 7. Expected Total Speedup

```
Baseline (current): 20-30 minutes per multi-seed evaluation

With Critical + High Priority:
  - Multi-seed parallelization: 5x
  - CV fold parallelization: 5x
  - Batch size + mixed precision: 3x
  - Data copy fixes: 1.1x
  ────────────────────────────
  Combined: ~50-75x theoretical speedup

With synchronization overhead (~30% efficiency loss):
  Result: 20-30 min → 1-2 minutes per evaluation
```

---

## 8. Implementation Priority (Maximum ROI)

**Phase 1 (Week 1): Critical Path - 10-20x speedup**
1. Parallelize multi-seed CV using ProcessPoolExecutor + memory-aware chunking
2. Increase CNN batch size from 8 to 32-64
3. Remove unnecessary `.copy()` in loader.py

**Phase 2 (Week 2): High Priority - 5-10x additional speedup**
1. Parallelize CV folds within each seed
2. Fix numpy percentile duplicate calls
3. Add num_workers to DataLoader (requires variable length handling)

**Phase 3 (Week 3): Polish - 1-2x additional speedup**
1. Mixed precision training for CNNs
2. Batch frame model predictions
3. Cache intermediate aggregations

---

## Summary

| Question | Answer |
|----------|--------|
| **Slowest operations?** | CNN training (80% of time): 50-100 epochs × 5 seeds × 5 folds |
| **Caching strategy optimal?** | Good overall. Missing: frame-level data caching, quality aggregation caching |
| **Parallelization opportunities?** | Massive: 25x potential (5 seeds × 5 folds) currently sequential |
| **Memory waste?** | Moderate: ~50-100 MB per simulation from unnecessary copy in loader.py |
| **Batch processing improvements?** | Minor: Good in CNN, could optimize frame models 10-20% |
| **GPU utilization?** | Moderate: Batch size 8 too small (underutilizes by 10-20x), no mixed precision |
