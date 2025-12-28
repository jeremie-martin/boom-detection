# Results and Metrics Analysis

**Agent Task**: Analyze metrics, results reporting, and statistical rigor.

**Date**: 2025-12-28

---

## Executive Summary

The evaluation methodology is sound (multi-seed CV with proper CIs), but reporting has inconsistencies. Key issues: conflicting MAE values (4.0 vs 6.4 vs 7.2), missing significance testing, and incomplete error distribution analysis. The code computes comprehensive metrics but doesn't showcase them all in documentation.

---

## 1. Metric Appropriateness

### Current Metrics Used

- **MAE (Mean Absolute Error)** - primary metric
- **Median Absolute Error** - available but rarely reported
- **Within N frames (5, 10, 15, 30)** - percentage accuracy thresholds
- **RMSE, Max AE** - computed but rarely reported
- **Acceptance rate** - key deployment metric
- **Correlation** - for quality prediction

### Assessment: MAE is REASONABLE but INCOMPLETE

**Strengths:**
- MAE is interpretable (mean error in frames)
- Aligns with the goal (YouTubers care about timing accuracy)
- Consistent with historical reporting

**Critical Gaps:**

1. **Median Absolute Error under-reported**
   - The evaluation.py computes it, but RESULTS.md only mentions it in one table
   - Median AE is more robust to outliers
   - Recent results show MedAE=5.0 (much better than MAE=7.2)
   - This should be a primary metric, not secondary

2. **No error distribution analysis**
   - Only mean/std reported
   - Missing: percentile information (25th, 50th, 75th)
   - Missing: bimodality detection (there appear to be "easy" vs "hard" cases)
   - Missing: error histograms per quality tier

3. **Within-N-frames tracked inconsistently**
   - Some reports show "Within 5 frames: 63%"
   - Others show "Within 10 frames: 61%"
   - Should standardize on multiple thresholds: Within 3, 5, 10, 15 frames

4. **Per-simulation error analysis missing**
   - EXPERIMENT_HISTORY.md mentions "6 sims have error >20"
   - No systematic analysis of which simulations are hardest
   - No feature-error correlation analysis

### Recommendation: Comprehensive Error Reporting

```
For all results report:
- MAE ± std (primary)
- Median AE ± std (robustness check)
- RMSE ± std (penalizes outliers)
- Within 3/5/10 frames accuracy ± std
- Max error (worst case)
- Acceptance rate & n_accepted
- Per-quality-tier breakdown
```

---

## 2. Result Consistency

### Protocol Issues Identified

**Positive findings:**
- All recent results use 5-fold CV × 5 seeds = 25 evaluations (GOOD)
- Using t-distribution CIs for small sample size (correct)
- Using ddof=1 for sample std dev (correct)
- Simulation-level train/test splits (prevents leakage)

**Critical Inconsistencies:**

| Finding | Impact | Severity |
|---------|--------|----------|
| **RESULTS.md shows MAE 6.4 but EXPERIMENT_HISTORY shows MAE 4.0** | Unclear which is current best | HIGH |
| Confidence intervals computed but rarely reported | Hide actual uncertainty | MEDIUM |
| Some results use "oracle quality", others use "predicted quality" | Not directly comparable | HIGH |
| Different feature sets for CNN vs HGB not always clear | Hard to reproduce | MEDIUM |
| "Quick" vs "robust" evaluation can differ by 0.3-0.5 frames | Results context missing | LOW |

**Root cause of MAE 4.0 vs 6.4:**
- Phase 12 initially used TRUE quality (oracle) → MAE 4.0
- Later corrected to predicted quality → MAE 6.4-6.7
- This is documented but confusing for readers

---

## 3. Statistical Rigor Assessment

### 5-Seed Evaluation: APPROPRIATE but MINIMAL

Current setup:
- N=49 simulations
- 5-fold CV × 5 seeds = 25 evaluations
- Each test fold: 49/5 ≈ 10 samples
- Standard error: std / √25 ≈ std / 5

For a dataset of 49 sims:
- 25 evaluations = reasonable for reporting
- But bootstrap CI would be more reliable than t-distribution
- Consider 10 seeds (50 evaluations) for production claims

### Good Practices:
- Multi-seed evaluation with uncertainty quantification
- t-distribution CIs appropriate for n=5 seeds
- Separated train/test at simulation level
- Reports mean ± std (not just mean)

### Missing Practices:

| Issue | Impact | Fix |
|-------|--------|-----|
| **No paired testing** | Can't tell if improvements are significant | Use paired t-test (seed-by-seed) |
| **No effect sizes** | "0.3 frame improvement" - is this real? | Report Cohen's d |
| **No multiple comparison correction** | Tested many thresholds | Apply Bonferroni |
| **No power analysis** | How many sims needed? | Compute sample size |
| **Bootstrap vs t-dist not compared** | Which is more accurate for n=49? | Compare CIs empirically |

### Example of Unreported Significance

From RESULTS.md line 45-46:
```
| + Top 50 features, ±25 window | 6.7 ± 0.6 | 34% |
| **+ Optimized CNN** | **6.4 ± 0.5** | 35% |
Improvement: 0.3 frames, 1% acceptance
```

Is 0.3 frames significant? With std=0.5:
- SE = 0.5/√5 ≈ 0.22
- t = 0.3/0.22 ≈ 1.36
- p > 0.05 (**NOT significant**)

But it's reported as an improvement without caveat!

---

## 4. Progress Tracking

### Baseline Comparison: EXISTS but Scattered

| Phase | MAE | Method | Notes |
|-------|-----|--------|-------|
| **Baseline** | 18.9 | HistGBM only | Starting point |
| Best accepted | **6.4 ± 0.5** | CNN+HGB+Quality filter | Current champion |
| Improvement | **66% reduction** | - | - |

### Problems with Progress Tracking

1. **Two conflicting "best" results:**
   - RESULTS.md: "MAE 6.4 ± 0.5"
   - EXPERIMENT_HISTORY.md: "MAE 4.0" (on 27% of simulations)
   - Different acceptance rates make them non-comparable

2. **Missing ablations for recent work:**
   - Why CNN kernels (5,11,21) better than others? (only 1 configuration tested)
   - Why hidden_dim=64? (only 1 tested)
   - Why ±25 window for quality? (only 1 tested)

3. **Weak failure analysis:**
   - 51% rejection due to disagreement (what distinguishes these?)
   - 45% of high-quality sims rejected (why? what features help?)

### Recommended Progress Table

```markdown
| Date | Approach | MAE (all) | MAE (accepted) | Acceptance | Status |
|------|----------|-----------|----------------|------------|--------|
| Baseline | HGB | 18.9 | - | 100% | |
| Phase 6 | + Agreement | ~16 | ~8 | 50% | |
| Phase 11 | + Quality | ~13 | ~6 | 30% | |
| Phase 12 | Oracle Q | ~10 | ~4 | 27% | NOT DEPLOYABLE |
| Current | Pred Q | ~15 | ~6.4 | 33% | DEPLOYABLE |
```

---

## 5. Reporting Completeness

### What's Reported Well:
1. ✓ Acceptance rate & acceptance breakdown
2. ✓ Within-N-frames for some configurations
3. ✓ Feature importance (RESULTS.md lines 110-133)
4. ✓ Quality-error correlation analysis
5. ✓ Model agreement as confidence signal

### What's Missing:

**A. Error Distribution Analysis:**
```
Current gap: No per-quality-tier error analysis
Missing:
  - High quality (≥0.5): MAE = ?
  - Mid quality (0.3-0.5): MAE = ?
  - Low quality (<0.3): MAE = ?
```

From EXPERIMENT_HISTORY (Phase 11):
- High-Q MAE: 8.4-11.2 (varies by model)
- Low-Q MAE: 18.2-24.9

This should be in RESULTS.md prominently!

**B. Failure Case Analysis:**

RESULTS.md has good breakdown:
```
Rejection reasons:
- Agreement fail: 51%
- Quality fail: 15%
- Accepted: 34%
```

But missing:
- **Hard simulation analysis:** Which 6 sims have error >20?
- **Feature analysis:** What makes these hard?
- **Recovery strategies:** Can any features predict hardness?

**C. Reproducibility Gaps:**

| Missing Info | Impact |
|---|---|
| Random seeds in feature extraction? | Can't reproduce exactly |
| Feature normalization parameters? | Different inference might occur |
| PyTorch version/determinism flags? | Non-reproducible across runs |
| CNN convergence plots? | Can't verify training |

**D. Confidence Signals Under-Analyzed:**

Reported in RESULTS.md:
```
Quality strongly predicts error (r=-0.454)
```

But missing:
- Per-quality ROC curve
- Quality threshold optimization curve
- Trade-off between acceptance and accuracy

---

## 6. Recommendations

### Immediate (Fix inconsistencies):
1. **Clarify MAE 4.0 vs 6.4 discrepancy** - Document oracle vs predicted clearly
2. **Add Median AE to all results** - It's computed but hidden
3. **Show error distribution histograms** - One plot per configuration
4. **Add statistical significance testing** - t-tests for paired comparisons

### High Value (Better analysis):
5. **Per-quality error breakdown** - Show where models actually fail
6. **Hard case analysis** - Deep dive into 51% rejection due to disagreement
7. **Bootstrap CI comparison** - Verify t-distribution assumption
8. **Effect size reporting** - Cohen's d alongside p-values

### Nice to Have (Documentation):
9. **Reproducibility guide** - Random seeds, normalization parameters
10. **Convergence plots** - Show CNN training curves
11. **Acceptance-accuracy trade-off curve** - Visualization of quality thresholds
12. **Failure mode taxonomy** - Categorize the hard cases

---

## Summary Assessment

| Dimension | Rating | Status |
|-----------|--------|--------|
| **Metric Appropriateness** | 7/10 | Good but incomplete |
| **Evaluation Protocol** | 8/10 | Mostly consistent |
| **Statistical Rigor** | 6/10 | Basic but missing tests |
| **Progress Tracking** | 7/10 | Exists but scattered |
| **Reporting Completeness** | 6/10 | Good high-level, poor details |
| **Confidence Intervals** | 8/10 | Correct methodology |
| **Overall** | **7/10** | Solid foundation, needs polish |

**Key insight**: The `evaluation.py` implementation is **very good**. It computes all relevant metrics, has proper CI computation with t-distribution, and supports multi-seed evaluation. The gap is purely in **reporting**, not in methodology.
