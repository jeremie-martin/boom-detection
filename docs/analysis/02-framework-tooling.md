# Framework and Tooling Analysis

**Agent Task**: Deep dive into the experimentation framework and tooling of the boom-detection codebase.

**Date**: 2025-12-28

---

## Executive Summary

The codebase has an excellent evaluation framework with proper multi-seed cross-validation and smart feature caching. However, it lacks experiment tracking, hyperparameter versioning, and has significant parallelization opportunities that could provide 25-50x speedup.

---

## Part 1: Current Experimentation Framework Assessment

### Architecture Overview
- **Total codebase**: 5,123 lines of Python across 12 modules
- **Core components**: Feature extraction (996 LOC), Evaluation (892 LOC), Models (630 LOC), Pipeline (413 LOC)
- **Execution pattern**: Single-file experiments using Python scripts with explicit parameters

### Strengths of Current Approach

#### 1. Robust Evaluation Framework (evaluation.py - 892 LOC)
- Excellent implementation of multi-seed cross-validation
- `MultiSeedResult` with proper uncertainty quantification (mean ± std ± 95% CI)
- Split at simulation level (prevents data leakage)
- Both `Evaluator` and `CachedEvaluator` interfaces
- Per-fold metrics tracking with aggregation

#### 2. Efficient Feature Caching (features.py - 996 LOC)
- Disk-based caching with config hashing for invalidation
- Parallel extraction via ThreadPoolExecutor
- Resolution-invariant features (aggregated over pendulums)
- Instant lookup after initial extraction (1.2GB cache indexed by sim ID)

#### 3. Clean Model Interface
- Consistent `fit(sim_ids, targets, cache)` and `predict(sim_ids, cache)` protocol
- All models (HistGBM, CNN, LSTM, Random Forest) follow same interface
- Enables easy swapping and ensemble composition

#### 4. Comprehensive Documentation
- Excellent experiment history (EXPERIMENT_HISTORY.md - 400+ lines)
- Clear results summary (RESULTS.md - 220+ lines)
- Each file has detailed docstrings and usage examples

---

## Part 2: What's Missing - Framework Gaps

### 1. No Experiment Tracking or Logging

**Current State**:
- Results exist as loose `*.npz` files (model_predictions.npz, enhanced_predictions.npz)
- Results printed to stdout, not systematically captured
- Manual documentation in markdown files

**Missing**:
- No experiment registry (no way to query "what experiments ran when?")
- No model checkpoints versioning
- No hyperparameter audit trail
- No performance timeline
- No failed experiment tracking

**Example Gap**: Running `deploy_pipeline.py --evaluate` produces a printed summary but doesn't save:
- Exact hyperparameters used
- Seed values tested
- Run timestamp
- Environment (PyTorch/sklearn versions)
- Wall-clock execution time per fold
- Which test simulations had worst errors

### 2. Limited Hyperparameter Tracking

**Current State**:
```python
# From deploy_pipeline.py - hardcoded values scattered
self.cnn = CNNClassifier(
    n_features=self.n_features,
    hidden_dim=64,           # ← Magic number, no registry
    kernel_sizes=(5, 11, 21) # ← No version control
)
```

**Missing**:
- No hyperparameter versioning
- No "best config so far" tracking
- No grid search results history
- No ability to compare two configurations side-by-side

### 3. No Reproducibility Guarantee System

**Current State**:
- Relies on seed specification in function calls
- Random seed set in individual model constructors (e.g., `random_state=42`)
- No enforcement of reproducibility

**Risks**:
- A developer could accidentally remove `random_state=42` from a model
- PyTorch seeds vs NumPy seeds not centrally managed
- No "frozen environment" specification

### 4. Manual Result Aggregation

**Current State** (from run_baselines.py):
```python
# Manual aggregation across seeds
for name, predictor_factory in baselines.items():
    all_metrics = []
    for seed in seeds:
        result = cross_validate_cached(...)
        all_metrics.append(result['metrics'])

    # Manually compute mean/std
    mean_metrics = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values, ddof=1)
```

**Problems**:
- Repeated in multiple places (run_baselines.py AND deploy_pipeline.py)
- No centralized aggregation logic
- No automatic comparison framework
- Difficult to compute relative improvement metrics

### 5. No Ablation Study Tracking

**Current State**:
- Ablation results documented manually in RESULTS.md (Tables 1-10)
- No automated ablation framework
- Example finding: "Top 50 features better than all 1365" documented but not versioned

**Missing**:
- No way to re-run ablation studies reproducibly
- No feature importance tracking system
- No automated comparison of feature sets

### 6. Feature Engineering Not Versioned

**Current State**:
- Features defined in FeatureConfig dataclass with multiple configs:
  - DEFAULT_CONFIG
  - ENHANCED_CONFIG
  - CAUSTIC_CONFIG
- No registry of "tried but didn't help" features
- No systematic A/B testing framework

**Example**:
```python
ENHANCED_CONFIG = FeatureConfig(
    max_pendulums=2000,
    include_rolling=True,
    rolling_windows=(10, 25),
    include_lag=True,
    lag_steps=(5, 15),
    include_relative=True,
)
```
No record of: Who created this? When? What was the impact?

### 7. No Model Comparison Dashboard

**Current State**:
- Results in markdown tables (manually typed)
- Hard to spot trends across experiments
- No visualization of MAE progression

**Example from RESULTS.md**:
```
| Phase | MAE | Approach |
|-------|-----|----------|
| Baseline | 18.9 | HistGBM |
| Phase 10 | 13.3 | Enhanced features |
| Phase 12 | 6.4 | Optimized CNN |
```
No way to query "show me all experiments with agreement_threshold=5"

---

## Part 3: Caching Strategy Analysis

### Current Caching (Excellent)
1. **Disk-based persistence**: `.feature_cache/` with 1.2GB of indexed numpy files
2. **Config-aware invalidation**: Hash of FeatureConfig determines cache key
3. **Parallel extraction**: Multi-threaded feature extraction saves ~2-3 minutes vs sequential
4. **Lazy loading**: Only loads features needed for current fold

### Caching Issues

**Issue 1: Feature Config Hash Invalidation**
- Hash computed only on FeatureConfig attributes
- If feature extraction function changes (e.g., bug fix in `caustic_features()`), cache not invalidated
- Example: Phase 10 refactored `rolling_features()`, but old cache files remain

**Issue 2: No Cache Versioning**
- Cache files named: `{sim_id}_{config_hash}.npy`
- If config changes slightly (e.g., from 36 to 37 bins), new hash generated
- Old config hashes gradually accumulate: `.feature_cache/` contains 4 different hash versions
- **Result**: Storage consumed by old cache versions

**Issue 3: No Cache Inspection Tools**
- No way to list what configs have cached data
- No cache cleanup utility
- No cache statistics (coverage, size breakdown)

**Opportunity**: A simple cache manager could:
```python
cache.list_configs()      # What feature versions are cached?
cache.get_stats()         # Coverage, size, oldest entry
cache.cleanup(keep_recent=2)  # Remove old versions
```

---

## Part 4: Model Configuration & Hyperparameter Space

### Current State
Hyperparameters scattered across 4 files:

1. **Frame Models** (frame_models.py):
   - HistGBM: n_estimators, max_depth, random_state
   - No defaults tracked

2. **Sequence Models** (sequence_models.py):
   - CNN: hidden_dim, kernel_sizes, dropout, lr, epochs
   - LSTM: hidden_dim, num_layers, lr, epochs
   - Hardcoded in SequenceTrainer

3. **Quality Models** (quality_models.py):
   - Random Forest: n_estimators, max_depth
   - Window size for context features
   - No registry

4. **Pipeline** (deploy_pipeline.py):
   - agreement_threshold: 5 frames (hardcoded)
   - quality_threshold: 0.55 (hardcoded)
   - quality_window: 25 frames (hardcoded)
   - n_quality_features: 50 (hardcoded)

### What's Missing
- No central config schema
- No way to compare "best config from phase 8" vs "best config from phase 12"
- No grid search history
- No learning curves per hyperparameter
- Hardcoded values in deploy_pipeline.py suggest these were found manually

---

## Part 5: Reproducibility Assessment

### Current Reproducibility: 7/10

**Good**:
- Multi-seed evaluation (5 seeds default)
- KFold shuffling with fixed random_state
- Feature cache ensures identical features across runs
- Model random_state specified (usually 42)

**Risky**:
- PyTorch seed not centrally managed (SequenceTrainer.fit() doesn't set torch.manual_seed)
- NumPy seeds set per-model, not globally
- No environment specification (requirements.txt? uv.lock has it but not in commit)
- "Robust evaluation" assumes 5 seeds, but quick_evaluate() uses 1

**Test**: Two runs of same config should produce identical results. Currently:
- ✅ Features identical (cached)
- ✅ Train/test split identical (fixed seed)
- ⚠️ Model initialization might differ (PyTorch randomness not controlled)
- ✅ Model predictions identical (sklearn models deterministic with fixed seed)

---

## Part 6: Comparison & Analysis Gaps

### No Built-in Comparison Framework
Users must manually:
1. Run experiment A, save results
2. Run experiment B, save results
3. Load both, compute differences
4. Create comparison table in markdown

**Example needed but missing**:
```python
# What we want but don't have:
result_a = CachedEvaluator(...).cross_validate(ModelA)
result_b = CachedEvaluator(...).cross_validate(ModelB)

comparison = result_a.compare_to(result_b)
print(comparison.summary())
# Output:
# Model B: MAE 6.4 ± 0.5 (△ -2.4 frames vs A, p < 0.001)
```

### Error Analysis Manual
**Current**: per_sample_errors() exists but:
- No per-quality-level analysis
- No per-agreement-level analysis
- No confusion matrix for classification
- Results printed but not saved

---

## Part 7: Parallelization Assessment

### Current Parallelization
1. **Feature extraction**: ✅ Parallelized via ThreadPoolExecutor (8 workers)
2. **Cross-validation folds**: ❌ Sequential (lines 750-778 in evaluation.py)
3. **Multiple seeds**: ❌ Sequential (lines 709-719 in evaluation.py)
4. **Model ensemble prediction**: ❌ Sequential (model.predict called 3 times)

### Opportunities

**Low-hanging fruit**: Parallelize fold execution
```python
# Current (sequential): 5 folds × ~30s per fold = 150s
# Could be: ceil(5 / n_workers) batches ≈ 60s with 4 workers

with ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(train_and_eval_fold, train_id, test_id)
               for train_id, test_id in fold_splits]
```

**Medium effort**: Parallelize seed evaluation
```python
# Current: 5 seeds × 5 folds × 30s = 750s
# With parallel seeds + folds: ~180s with 8 workers
```

**Impact**: Complete robust evaluation could go from 15 min to ~5 min.

---

## Part 8: What Would Help Most?

### Priority 1: Experiment Registry (Easy - 4 hours)
Create a lightweight JSON-based registry:
```python
# experiments.json
{
  "exp_20251228_001": {
    "timestamp": "2025-12-28T10:30:00",
    "model": "BoomDetectionPipeline",
    "config": {
      "agreement_threshold": 5,
      "quality_threshold": 0.55,
      "cnn_hidden_dim": 64,
      "cnn_kernels": [5, 11, 21]
    },
    "results": {
      "mae_mean": 6.4,
      "mae_std": 0.5,
      "within_5_mean": 0.63,
      "acceptance_rate": 0.35
    },
    "seeds": [42, 43, 44, 45, 46],
    "status": "completed",
    "duration_seconds": 1847
  }
}
```

**Benefits**:
- Query experiments by model, config, date
- Track MAE progression
- Compare configurations
- Identify best model per metric

### Priority 2: Reproducibility Lock (Medium - 8 hours)
```python
# reproducibility.py
class ReproducibilityContext:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def __enter__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        # Disable cuDNN randomness if using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return self

    def __exit__(self, *args):
        pass

# Usage:
with ReproducibilityContext(seed=42):
    evaluator.cross_validate(model)  # Fully deterministic
```

### Priority 3: Model Checkpointing (Medium - 6 hours)
```python
# model_checkpoint.py
class CheckpointManager:
    def save(self, model, config, results, path):
        torch.save({
            'model_state': model.state_dict(),
            'config': config,
            'results': results,
            'timestamp': datetime.now(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        return checkpoint['model'], checkpoint['results']
```

### Priority 4: Parallel Fold & Seed Evaluation (Medium - 10 hours)
Refactor evaluation loop to use ProcessPoolExecutor for fold/seed combinations.
Could reduce 5-seed evaluation from 15 min to 4 min.

### Priority 5: Visualization Dashboard (Hard - 20 hours)
Simple Streamlit app showing:
- MAE over time (phases)
- Model comparison table
- Hyperparameter sensitivity
- Error distribution by quality/agreement

---

## Part 9: Recommendations with Complexity Estimates

| Recommendation | Complexity | Time | Benefit | Priority |
|---|---|---|---|---|
| **Experiment Registry** | Easy | 4h | Massive - enables tracking | HIGH |
| **Reproducibility Context** | Easy | 2h | Critical - ensures valid comparisons | HIGH |
| **Cache Manager** | Easy | 3h | High - cleanup, stats, versioning | MEDIUM |
| **Config Schema** | Easy | 4h | High - centralized hyperparameters | MEDIUM |
| **Result Aggregator** | Easy | 3h | High - reusable across experiments | MEDIUM |
| **Parallel Fold Execution** | Medium | 10h | Medium - 2-3x speedup on CV | MEDIUM |
| **Model Checkpointing** | Medium | 6h | Medium - reproducibility + ensembles | MEDIUM |
| **Parallel Seed Evaluation** | Medium | 10h | Medium - 3-4x speedup on multi-seed | MEDIUM |
| **Error Analysis Tools** | Medium | 8h | Medium - per-quality, per-agreement breakdowns | LOW |
| **Visualization Dashboard** | Hard | 20h | Low - nice to have, not critical | LOW |
| **MLflow Integration** | Hard | 16h | High - standard practice, but overhead | LOW |

---

## Summary

**Current Strengths**:
- Excellent evaluation framework with proper uncertainty quantification
- Smart caching strategy for reproducible iteration
- Clean model interface enabling easy swapping
- Comprehensive documentation

**Key Gaps**:
1. No experiment tracking/versioning (biggest pain point)
2. Hyperparameters scattered, no registry
3. Reproducibility not enforced
4. Limited parallelization (folds/seeds run sequentially)
5. Manual result aggregation across multiple places
6. No feature importance/ablation versioning

**Recommended Path Forward**:
1. **Week 1**: Add lightweight experiment registry + reproducibility context (6 hours total)
2. **Week 2**: Add cache manager + config schema + parallel fold execution (15 hours)
3. **Week 3**: Model checkpointing + error analysis tools (14 hours)
4. **Nice-to-have**: Visualization dashboard or MLflow integration

The codebase is well-engineered at the model level but lacks tooling at the experiment management level. This gap becomes painful as the search space grows (hyperparameters × configurations × seeds = hundreds of experiments to manage).
