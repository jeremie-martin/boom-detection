# Experiment Results

## Summary (December 2024)

Three incremental experiments were run to identify potential improvements to the current pipeline:

| Experiment | Finding | Actionable? |
|------------|---------|-------------|
| CNN Threshold | 0.60 slightly better than 0.50 | No (p=0.32) |
| Agreement Formulation | sqrt/15 achieves MAE 7.25 | Marginal (p=0.058) |
| Confidence Calibration | ECE=0.15, 12% overconfident | Yes - needs calibration |

## 1. CNN Threshold Sweep

**Question**: Is the default threshold of 0.5 for P(after_boom) optimal?

**Method**: 5-fold CV × 3 seeds, sweeping thresholds from 0.3 to 0.7

**Results**:
| Threshold | MAE | Within 5 | Within 10 |
|-----------|-----|----------|-----------|
| 0.30 | 19.19 | 38.7% | 56.7% |
| 0.35 | 17.21 | 36.7% | 56.0% |
| 0.40 | 16.31 | 35.3% | 56.7% |
| 0.45 | 15.99 | 36.0% | 55.3% |
| **0.50** | **14.85** | 37.3% | 56.7% |
| 0.55 | 14.65 | 36.0% | 58.0% |
| **0.60** | **14.57** | 37.3% | 56.0% |
| 0.65 | 14.69 | 40.0% | 57.3% |
| 0.70 | 14.72 | 38.7% | 56.0% |

**Conclusion**: Best threshold is 0.60 (MAE 14.57) vs 0.50 (MAE 14.85), but the difference is **not statistically significant** (paired t-test p=0.32). Keep default of 0.50.

## 2. Agreement Score Formulation

**Question**: Is the current formula `1 - min(|diff|/10, 1)` optimal?

**Method**: Test 4 functions × 4 scales, 5-fold CV × 3 seeds

**Results (best performers per function)**:
| Function | Scale | MAE | Coverage | Notes |
|----------|-------|-----|----------|-------|
| **sqrt** | **15** | **7.25** | 42.7% | Best MAE |
| sigmoid | 5 | 7.41 | 34.0% | Second best |
| linear | 5 | 8.17 | 34.0% | Most selective |
| **linear** | **10** | **8.54** | **50.7%** | **Current default** |
| quadratic | 10 | 9.10 | 58.0% | High coverage |

**Key Trade-off**: Lower scales = lower MAE but lower coverage.

**Conclusion**: sqrt with scale=15 achieves MAE 7.25 (vs 8.54 default), but p=0.058 just misses significance at α=0.05. The improvement is promising but not conclusive. Consider:
- Using sqrt/15 for production if lower coverage is acceptable
- Running more seeds to confirm significance

## 3. Confidence Calibration

**Question**: Is the accept_score well-calibrated?

**Method**: Compute ECE, MCE, Brier score across 5-fold CV × 3 seeds

**Results**:
| Metric | Value | Interpretation |
|--------|-------|----------------|
| ECE | 0.1495 | Poor (≥0.10) |
| MCE | 0.2836 | High maximum error |
| Brier | 0.2371 | Moderate |
| Avg Confidence | 0.4910 | - |
| Avg Accuracy | 0.3667 | - |
| Overconfidence | **+12.43%** | Significantly overconfident |

**Per-bin breakdown**:
| Confidence Bin | Count | Actual Accuracy | Gap |
|----------------|-------|-----------------|-----|
| 0.15 | 13 | 30.8% | +14.5% (underconfident) |
| 0.25 | 28 | 17.9% | -5.4% |
| 0.35 | 18 | 22.2% | -13.1% |
| 0.45 | 12 | 16.7% | **-28.4%** (most overconfident) |
| 0.55 | 22 | 50.0% | -6.2% |
| 0.65 | 25 | 44.0% | -21.3% |
| 0.75 | 24 | 54.2% | -20.3% |
| 0.85 | 8 | 62.5% | -21.0% |

**Conclusion**: The model is **consistently overconfident** above the 0.25 bin. When it claims 75% confidence, actual accuracy is only 54%. This suggests:
1. The current accept_threshold (0.53) should be **higher** to compensate
2. Post-hoc calibration (isotonic regression, temperature scaling) would help
3. The current quality calibration in the pipeline may not be sufficient

## Recommendations

### Short-term (low risk):
1. **Increase accept_threshold** from 0.53 to ~0.60 to compensate for overconfidence
2. **Consider sqrt/15 formulation** if targeting ~40% coverage instead of ~50%

### Medium-term (requires validation):
1. **Add temperature scaling** to accept_score before thresholding
2. **Use isotonic regression** on accept_score (similar to quality calibration)

### Long-term (research):
1. **Explore conformal prediction** for principled uncertainty quantification
2. **Multi-task learning** for boom + quality to get better shared representations

---

## Reproducibility

All experiments used:
- 5-fold cross-validation
- 3 seeds (42, 43, 44)
- FeatureConfig(max_pendulums=2000)
- Features cached in `.feature_cache/`

Run scripts:
```bash
uv run python scripts/threshold_sweep.py data --seeds 3
uv run python scripts/agreement_formulation.py data --seeds 3
uv run python scripts/calibration_check.py data --seeds 3
```
