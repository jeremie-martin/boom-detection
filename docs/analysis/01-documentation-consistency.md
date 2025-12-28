# Documentation Consistency Audit

**Agent Task**: Analyze the boom-detection codebase for documentation consistency and accuracy.

**Date**: 2025-12-28

---

## Executive Summary

The documentation has several significant inconsistencies, particularly around which model to use (CNN vs HGB), reported MAE values (4.0 vs 6.4 vs 7.2), and whether results use oracle or predicted quality.

---

## Critical Inconsistencies

### 1. Which Model to Use: CNN vs HGB (SEVERITY: HIGH)

**Inconsistency Summary:**
- **EXPERIMENT_HISTORY.md (Phase 12)** claims the best config is "use **HGB**" with MAE 4.0
- **RESULTS.md (lines 10-26)** explicitly documents "use boom_cnn as final prediction"
- **Code (deploy_pipeline.py, line 192)** uses: `'boom_frame': cnn_pred if accepted else None`
- **CLAUDE.md (line 39)** shows HGB in example code

**Details:**
- EXPERIMENT_HISTORY.md states (line 352, 365, 391):
  - "use **HGB prediction** (not average!)"
  - "Use HGB, not average: When models agree, HGB alone is more accurate"
  - "best config is Agree≤5, PredQ≥0.55, use HGB"

- However, RESULTS.md (lines 21, 55-59) contradicts this:
  - "→ ACCEPT, use boom_cnn as final prediction"
  - Shows ablation study: CNN MAE 7.1±0.7 vs HGB MAE 11.0±4.5
  - Conclusion: "CNN is more accurate and has lower variance"

- Commit 33129e3 "Ablation study: CNN outperforms HGB, update pipeline to use CNN" confirms CNN is better
- Commit d3c7e31 (most recent) further optimizes CNN architecture

**Status:** Code correctly uses CNN, but EXPERIMENT_HISTORY.md is outdated and contradicts current implementation.

---

### 2. Performance Metrics: MAE Values (SEVERITY: HIGH)

**Inconsistency Summary:**
Multiple different "best" MAE values claimed across documentation:

| Document | Claimed Best | Context |
|----------|--------------|---------|
| CLAUDE.md | 7.2 ± 1.1 | "Current best" |
| README.md | 6.4 ± 0.5 | "Best Result" |
| RESULTS.md | 6.4 ± 0.5 | "Best Result" |
| EXPERIMENT_HISTORY Phase 12 | 4.0 | "Fully deployable" |
| deploy_pipeline.py docstring | 6.5 ± 0.3 | Code docstring (lines 10-13) |

**Details:**
- **CLAUDE.md (line 14):** "Current best: MAE 7.2 ± 1.1 frames with ~33% acceptance rate"
  - This appears to be an older baseline from the original pipeline before CNN optimization

- **README.md & RESULTS.md:** Both correctly state 6.4 ± 0.5
  - This reflects the most recent CNN optimization (commit d3c7e31)

- **deploy_pipeline.py docstring (lines 10-13):** Claims 6.5 ± 0.3
  - Close to 6.4 but slightly different; possibly from different evaluation

- **EXPERIMENT_HISTORY Phase 12 (line 379):** Claims MAE 4.0 on ~27% of sims
  - This is performance on ACCEPTED simulations only (after filtering)
  - Not the overall MAE, which is 6.4 ± 0.5
  - The Phase 12 configuration uses oracle quality (line 352) vs predicted quality used in current implementation

**Status:** CLAUDE.md is significantly outdated (7.2 vs 6.4). EXPERIMENT_HISTORY uses oracle quality results which conflict with "predicted quality" deployable approach.

---

### 3. Acceptance Rate Inconsistency (SEVERITY: MEDIUM)

**Values claimed:**
- CLAUDE.md: ~33% acceptance rate (line 14)
- README.md: 35% ± 5% (line 13)
- RESULTS.md: 35% ± 5% (line 36)
- NEXT_STEPS.md: 35% (line 5)
- EXPERIMENT_HISTORY Phase 12: 27% (line 358), 29% (line 359), 30% (line 361) depending on config

**Status:** The ~33% in CLAUDE.md matches the "original baseline" but the current best is 35% with CNN optimization. Phase 12 configs show 27-30% but use different thresholds. These reflect different hyperparameter configurations.

---

### 4. Within 5 Frames Metric (SEVERITY: MEDIUM)

**Values claimed:**
- README.md: 63% ± 6% (line 12)
- RESULTS.md: 63% ± 6% (line 35)
- deploy_pipeline.py: 60% (implied from code structure)

**Status:** Consistent between README and RESULTS (both 6.4 ± 0.5 results), but EXPERIMENT_HISTORY Phase 12 shows different values (77%, 79%) for the oracle quality configuration.

---

### 5. Feature Count in CLAUDE.md (SEVERITY: LOW)

**Issue:**
- CLAUDE.md doesn't mention feature selection or dimensionality
- Code uses top 50 quality features (deploy_pipeline.py, line 128)
- Docs mention "183 features" in EXPERIMENT_HISTORY (line 61) but this is the full set
- Quality model uses selected features: `window_feats_selected = window_feats[self.quality_feature_indices]`

**Status:** CLAUDE.md oversimplifies the feature handling by not mentioning that quality prediction uses feature selection.

---

### 6. Python File Paths in CLAUDE.md (SEVERITY: MEDIUM)

**Issue:**
CLAUDE.md File Guide (lines 80-90) references files that don't match actual module structure:

| CLAUDE.md says | Actual location |
|----------------|-----------------|
| `deploy_pipeline.py` | `src/boom_detection/deploy_pipeline.py` |
| `features.py` | `src/boom_detection/features.py` |
| `frame_models.py` | `src/boom_detection/frame_models.py` |
| All others | `src/boom_detection/*.py` |

**Status:** Paths are incomplete (missing `src/boom_detection/` prefix), though could be interpreted as relative imports.

---

### 7. Outdated Performance Progression (SEVERITY: MEDIUM)

**NEXT_STEPS.md Progress Summary (line 122-128):**
```
| Date | MAE | Acceptance | Key Change |
...
| **+CNN tuning** | **6.4** | **35%** | Larger kernels |
```

This table ends at 6.4 but doesn't show the recent improvements or acknowledge that EXPERIMENT_HISTORY Phase 12 achieved 4.0 on accepted simulations.

**Status:** Incomplete progress tracking; Phase 12 results not reflected in NEXT_STEPS progress summary.

---

### 8. Model Agreement in CLAUDE.md (SEVERITY: LOW)

**Issue:**
- CLAUDE.md (line 39) says "# Use HGB prediction" in code example
- Should be "# Use CNN prediction" based on current ablation study

```python
# CLAUDE.md (incorrect):
if result['accepted']:
    boom_frame = result['boom_frame']  # Use HGB prediction  ← WRONG

# Code is correct:
'boom_frame': cnn_pred if accepted else None  # CNN is more accurate
```

---

## Documentation Accuracy Issues

### 1. EXPERIMENT_HISTORY: Conflicting Approaches (SEVERITY: HIGH)

**Problem:**
- Phase 12 claims MAE 4.0 with "use HGB"
- But Phase 12 also says this uses ORACLE quality (line 342): "oracle (ground truth) quality which is NOT available at inference time"
- Then it claims it's "fully deployable in production" (line 352)

**Reality:**
- MAE 4.0 results are NOT deployable because they use oracle quality
- Current deployable approach (used in code) uses predicted quality and gets MAE 6.4
- The config in Phase 12 uses different thresholds (Agree≤5, PredQ≥0.55) which would be different

**Status:** EXPERIMENT_HISTORY is misleading by presenting Phase 12 as "fully deployable" when it uses oracle quality.

---

### 2. Resolution of Experiments (SEVERITY: MEDIUM)

**Issue:**
- EXPERIMENT_HISTORY describes extensive experimentation (Phase 1-12) with many intermediate results
- RESULTS.md only shows the final optimized pipeline
- NEXT_STEPS.md says work is "completed" but lists "optional future work"
- These seem to represent different phases of work

**Status:** Docs don't clearly articulate that:
- EXPERIMENT_HISTORY is exploratory work
- RESULTS.md is the final optimized pipeline
- Some Phase 12 results used oracle quality (not deployable)
- The actual deployed config is simpler than Phase 12 suggests

---

### 3. Data Leakage Discussion (SEVERITY: LOW)

**Issue:**
- EXPERIMENT_HISTORY Phase 12 uses oracle quality in results
- CLAUDE.md warns "Don't use oracle quality at inference"
- RESULTS.md is newer and correctly uses predicted quality
- But EXPERIMENT_HISTORY still presents oracle quality results as "fully deployable"

**Status:** EXPERIMENT_HISTORY needs clarification that Phase 12's MAE 4.0 is NOT the deployable result.

---

## Missing/Incomplete Documentation

### 1. No Clear Statement of Current Deployable Config (SEVERITY: MEDIUM)

The actual deployed configuration isn't clearly summarized. Need to state:
- Agreement threshold: 5
- Quality threshold: 0.55
- Quality window: ±25 frames
- Quality features: top 50 correlated
- Final prediction: CNN (not HGB, not average)
- Expected MAE: 6.4 ± 0.5
- Expected acceptance: 35% ± 5%

**Note:** This info is scattered across multiple files.

### 2. No TODO/FIXME Comments Found (POSITIVE)

- Grep for TODO|FIXME|XXX|HACK returned no results
- Code appears clean and complete

---

## Summary Table

| Issue | Severity | Files Affected | Status |
|-------|----------|-----------------|--------|
| CNN vs HGB model choice | HIGH | EXPERIMENT_HISTORY, CLAUDE.md | Code correct, docs outdated |
| MAE values inconsistent (7.2 vs 6.4 vs 4.0) | HIGH | CLAUDE.md, README, RESULTS, EXPERIMENT_HISTORY | Reflects older baseline + oracle quality + current best |
| EXPERIMENT_HISTORY Phase 12 uses oracle quality | HIGH | EXPERIMENT_HISTORY | Presented as deployable but isn't |
| Acceptance rate 33% vs 35% | MEDIUM | CLAUDE.md vs others | CLAUDE.md outdated |
| File paths missing prefixes | MEDIUM | CLAUDE.md | Incomplete relative paths |
| WITHIN 5 frames inconsistency | MEDIUM | Multiple docs | Varies with config/evaluation |
| Feature selection not mentioned in CLAUDE.md | MEDIUM | CLAUDE.md | Oversimplified |
| Model choice in CLAUDE.md code example | LOW | CLAUDE.md | Says HGB, should say CNN |
| NEXT_STEPS missing Phase 12 results | MEDIUM | NEXT_STEPS.md | Progress summary incomplete |
| No clear current deployable config summary | MEDIUM | All docs | Info scattered |

---

## Recommendations

1. **UPDATE CLAUDE.md** to reflect current state (6.4 ± 0.5, CNN prediction, 35% acceptance)
2. **CLARIFY EXPERIMENT_HISTORY Phase 12** that MAE 4.0 uses oracle quality (not deployable)
3. **ADD SECTION** to RESULTS.md or README explicitly listing "Current Deployable Configuration"
4. **UPDATE NEXT_STEPS.md** to note that Phase 12 results aren't the deployed config
5. **FIX CLAUDE.md code example** to use CNN instead of HGB
6. **ADD NOTES** that EXPERIMENT_HISTORY is exploratory; only RESULTS.md reflects deployed system
