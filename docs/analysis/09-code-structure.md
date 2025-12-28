# Code Structure Analysis

**Agent Task**: Analyze the code structure, architecture, and engineering quality.

**Date**: 2025-12-28

---

## Executive Summary

The codebase is well-structured with 12 modules totaling 5,123 LOC. It has clean dependencies, consistent interfaces, and excellent type safety. However, it lacks tests (critical gap), has code duplication in model factories, and contains scattered magic numbers. Overall rating: **B+ (Good with clear improvement areas)**.

---

## 1. Architecture & Module Structure

### Overall Structure (12 modules, 5,123 LOC)

```
src/boom_detection/
├── loader.py              (322 LOC)  - Data I/O, annotations
├── features.py            (996 LOC)  - Feature extraction + caching
├── frame_models.py        (257 LOC)  - Frame-level regressors/classifiers
├── sequence_models.py     (448 LOC)  - PyTorch models (CNN, LSTM, Transformer)
├── quality_models.py      (378 LOC)  - Quality prediction
├── evaluation.py          (892 LOC)  - Metrics, CV, evaluation framework
├── pipeline.py            (385 LOC)  - Multi-stage pipelines
├── model_agreement.py     (256 LOC)  - Ensemble/agreement analysis
├── ensemble.py            (130 LOC)  - Adaptive ensemble
├── deploy_pipeline.py     (413 LOC)  - Production-ready pipeline
├── run_baselines.py       (502 LOC)  - Baseline comparisons
└── __init__.py            (144 LOC)  - Public API exports
```

### Dependency Graph

```
Healthy hierarchy:
  loader.py (foundation)
    ↓
  features.py (depends: loader)
    ↓
  {frame_models, sequence_models, quality_models, evaluation, pipeline}
    (depend: loader, features)
    ↓
  deploy_pipeline.py, run_baselines.py (top-level)
```

**Assessment**: Clean, acyclic dependencies. No circular imports detected.

---

## 2. Code Quality Strengths

### A. Consistent Interface Design

All model classes follow a standard protocol:
```python
class Model:
    def fit(self, sim_ids: list[str], targets: np.ndarray, cache: FeatureCache) -> None
    def predict(self, sim_ids: list[str], cache: FeatureCache) -> np.ndarray
```

- **46 model classes** all implement this interface
- Easy to add new models - just implement these two methods
- Models are compatible with evaluation framework (Protocol-based)

### B. Type Safety

- Modern Python type hints throughout (`from __future__ import annotations`)
- Protocol classes for interface definition (CachedPredictor protocol)
- Proper use of `Literal` types for constrained values
- No untyped parameters

### C. Comprehensive Feature Extraction

- **50+ feature functions** covering multiple aspects:
  - Variance/IQR/range statistics
  - Tip spread (position-based)
  - Angular spread (circular statistics)
  - Velocity features
  - Caustic/distribution features
  - Temporal derivatives
- **Resolution-invariant** design (aggregates over pendulums)
- **Caching system** for ~200x speedup

### D. Robust Evaluation Framework

- Multi-seed cross-validation with uncertainty estimates
- Protocol-based CachedPredictor interface
- Proper statistical reporting (mean ± std, 95% CI)
- Multiple metrics (MAE, median AE, RMSE, correlation, within-N)

---

## 3. Code Smells & Technical Debt

### Critical Issues

#### 1. Code Duplication in Feature Extraction (quality_models.py)
```python
# DUPLICATED in QualityRegressor.fit() AND QualityRegressor.predict()
def _extract_sim_features(self, features):  # Called in BOTH methods
def _extract_spike_features(self, features)  # Called in BOTH methods
```
**Impact**: Maintenance burden, inconsistency risk
**Fix**: Extract to single method, call from both

#### 2. Duplicate Model Creation Logic
Frame models and quality models both replicate sklearn model creation:
```python
# In frame_models.py (3 times)
if model == 'ridge':
    self._model = Ridge(...)
elif model == 'gbm':
    ...
# Same pattern in quality_models.py, run_baselines.py
```
**Impact**: 9+ duplicate model factory blocks
**Solution**: Single factory function for sklearn models

#### 3. Magic Numbers Scattered Throughout
```python
deploy_pipeline.py:93       kernel_sizes=(5, 11, 21)      # Why these values?
deploy_pipeline.py:96       lr=0.5e-3, epochs=30, patience=5
deploy_pipeline.py:108      max_iter=200, max_depth=7
quality_models.py:113       + 1e-8                         # Magic epsilon
```
**Fix**: Move to configuration dataclass or constants module

#### 4. Inconsistent Error Handling
- Some files check for errors (loader.py)
- Others silently fail or assume success (frame_models.py, quality_models.py)

---

## 4. Module-Level Issues

### Very Large Modules
- `features.py`: 996 LOC - Could split into statistical, spatial, temporal, caustic
- `evaluation.py`: 892 LOC - Could split into metrics, validation, evaluators
- `run_baselines.py`: 502 LOC - Experimental code, could be organized better

### Inconsistent Naming
- `fit()` and `predict()` sometimes take `sim_ids: list[str]` and cache
- Sometimes take `X: Sequence[Simulation]` and targets
- `predict()` returns different types:
  - `np.ndarray` (frame_models, sequence_models)
  - `list[dict]` (deploy_pipeline.predict())

### Missing Input Validation
```python
# No validation that:
# - cache contains required sim_ids
# - features have expected shape (frames, n_features)
# - boom_frames are in valid range [0, frame_count)
```

---

## 5. Testing Gap (Critical)

**Status**: No test directory exists. Critical functionality untested:
- Feature extraction correctness
- Model fitting/prediction consistency
- Evaluation metric calculations
- Cross-validation logic
- Edge cases (empty sequences, single frame, etc.)

**Minimal Test Suite Needed**:
```python
# tests/test_evaluation.py
def test_kfold_no_leakage(): ...

# tests/test_frame_models.py
def test_frame_regressor_fit_predict(): ...

# tests/test_pipeline.py
def test_pipeline_accepts_rejects_appropriately(): ...

# tests/test_edge_cases.py
def test_single_frame_handling(): ...
def test_nan_infinity_handling(): ...
```

---

## 6. Configuration Management Issues

### Problems
- **No config file format** (YAML, JSON, TOML)
- **Magic numbers in code**
- **No ConfigurationError** for invalid values

### Better Pattern
```python
@dataclass
class PipelineConfig:
    agreement_threshold: int = 5
    quality_threshold: float = 0.55
    cnn_kernel_sizes: tuple = (5, 11, 21)
    cnn_hidden_dim: int = 64

    def validate(self):
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError(f"Invalid quality_threshold")
```

---

## 7. Redundancies & Potential Dead Code

### Definite Redundancy
1. **Model factory blocks**: 9+ identical sklearn model creation patterns
2. **Feature extraction in quality models**: 2x duplication in fit/predict
3. **Model classes with same interface**: Could be consolidated

### Potential Dead Code
- `model_agreement.py:AgreementBasedSelector` (not imported elsewhere)
- `pipeline.py:ConditionalPipeline` (not used in deploy_pipeline.py)
- `run_baselines.py:VarianceThresholdPredictor` (experimental baseline)

---

## 8. Priority-Ranked Engineering Improvements

### P0: CRITICAL (Correctness/Functionality)
1. **Add comprehensive test suite** (50+ tests)
2. **Extract duplicate feature extraction logic**
3. **Add input validation** to all model classes
4. **Document feature normalization**
5. **Fix inconsistent return types**

### P1: HIGH (Maintainability/Scalability)
6. **Extract model factory function**
7. **Refactor large modules** into packages
8. **Create configuration dataclass**
9. **Create abstract base class** for all model types
10. **Add comprehensive error handling**
11. **Document data flow** and architecture

### P2: MEDIUM (Code Quality)
12. **Consolidate quality predictor classes**
13. **Remove dead code**
14. **Add logging** instead of silent failures
15. **Create factory functions** for common model combinations
16. **Document hyperparameter choices**

### P3: LOW (Nice-to-Have)
17. **Type checking**: Run mypy with strict mode
18. **Code coverage**: Set up pytest-cov, target 80%+
19. **Documentation**: Generate API docs from docstrings
20. **CI/CD pipeline**: GitHub Actions for tests + type checking

---

## 9. Positive Engineering Practices

The codebase does many things **right**:

✓ **Type safety** - Full type hints, Protocol-based interfaces
✓ **Consistent naming** - Snake_case, clear abbreviations
✓ **Documentation** - Docstrings on all public methods
✓ **Modularity** - Clear separation of concerns
✓ **Dependency management** - Clean acyclic graph
✓ **Feature caching** - Smart use of disk persistence
✓ **Evaluation rigor** - Multi-seed CV with uncertainty
✓ **Clean data model** - Dataclasses for domain objects
✓ **No premature optimization** - Readable code first
✓ **Git history** - Clear commit messages tracking evolution

---

## Summary

| Aspect | Rating | Key Issue |
|--------|--------|-----------|
| **Architecture** | A- | Clean, modular structure |
| **Interface Design** | A | Consistent 46-class protocol |
| **Type Safety** | A | Full annotations, Protocol classes |
| **Documentation** | B+ | Good docstrings, missing architecture docs |
| **Testing** | F | No tests exist |
| **Code Duplication** | C | 9+ duplicate model factories |
| **Configuration** | C- | Magic numbers scattered |
| **Error Handling** | C | Inconsistent across modules |

**Overall: B+ (Good with clear improvement areas)**

**Effort to Fix**:
- **Quick wins**: P0 items (5-10 hours)
- **Medium effort**: P1 refactoring (20-30 hours)
- **Long term**: P2-P3 improvements (10-20 hours)

**Recommendation**: Focus on P0 first (especially tests + validation), then P1 refactoring to reduce duplication.
