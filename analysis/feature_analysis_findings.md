# Feature Analysis Findings

## Summary

Comprehensive feature analysis was conducted on 90 simulations to understand which features contribute to boom frame prediction.

## Key Results

### Feature Importance Ranking (Top 10)

| Rank | Feature | Importance | Group |
|------|---------|------------|-------|
| 1 | joint_concentration | 0.0081 | caustic |
| 2 | iqr_th1 | 0.0058 | iqr |
| 3 | var_th1 | 0.0049 | variance |
| 4 | std_th1 | 0.0045 | std |
| 5 | range_th1 | 0.0043 | range |
| 6 | th1_spread | 0.0043 | spread |
| 7 | iqr_th2 | 0.0035 | iqr |
| 8 | var_th2 | 0.0033 | variance |
| 9 | range_th2 | 0.0031 | range |
| 10 | th2_spread | 0.0031 | spread |

### Feature Group Summary

| Group | Mean Importance | Top Feature |
|-------|-----------------|-------------|
| caustic | 0.0043 | joint_concentration |
| iqr | 0.0024 | iqr_th1 |
| variance | 0.0020 | var_th1 |
| range | 0.0019 | range_th1 |
| spread | 0.0015 | th1_spread |
| std | 0.0013 | std_th1 |

## Velocity Dynamics Features (Removed)

The velocity dynamics and neighbor coherence features were evaluated and found harmful:

| Configuration | MAE | Std | Change |
|---------------|-----|-----|--------|
| Default (baseline) | 7.57 | 1.90 | - |
| With Dynamics | 8.51 | 0.78 | **+12.3% worse** |

**Action**: These features were removed from the codebase.

## Causticness Investigation (Major Finding!)

Investigation of causticness formulations revealed that **plain coverage outperforms coverage×gini by 26.6%**.

### Formula Comparison (single-feature regression)

| Formula | Angle | Correlation | MAE |
|---------|-------|-------------|-----|
| **coverage** | th1+th2 | **-0.718** | **109.1** |
| coverage | tip_dir | -0.733 | 112.8 |
| coverage | th2 | -0.743 | 114.5 |
| 1-entropy | tip_dir | +0.708 | 115.6 |
| 1-R (dispersion) | th2 | -0.724 | 116.2 |
| coverage | th1 | -0.721 | 117.9 |
| gini | th1 | +0.691 | 121.8 |
| coverage×gini | th1+th2 | +0.085 | 141.7 |
| coverage×gini | th1 | **-0.016** | **148.7** |

**Key insight**: The gini coefficient adds noise. Coverage alone has strong correlation (-0.718) while coverage×gini has near-zero correlation (-0.016).

### Changes Made

1. **Improved `joint_concentration`**: Now uses `coverage(th1) × coverage(th2)` instead of `(coverage×gini)²`
2. **Added coverage-only features**: `tip_coverage`, `th1_coverage`, `th2_coverage`
3. **Legacy features kept**: For backwards compatibility, the original causticness features remain

### New Caustic Features (9 total)

| Feature | Formula | Notes |
|---------|---------|-------|
| angular_causticness | coverage×gini of (θ1+θ2) | Legacy |
| tip_causticness | coverage×gini of atan2(x2,y2) | Legacy |
| joint_concentration | coverage(θ1)×coverage(θ2) | **Improved** |
| organization_causticness | (1-R1×R2)×coverage | Updated |
| th1_causticness | coverage×gini of θ1 | Legacy |
| th2_causticness | coverage×gini of θ2 | Legacy |
| tip_coverage | coverage of (θ1+θ2) | **New - best!** |
| th1_coverage | coverage of θ1 | **New** |
| th2_coverage | coverage of θ2 | **New** |

## Redundant Features

High correlations (r > 0.99) indicate redundant features:

| Feature Pair | Correlation |
|--------------|-------------|
| range_th1 ↔ th1_spread | 1.000 |
| range_th2 ↔ th2_spread | 1.000 |
| std_x2 ↔ tip_mean_dist | 0.992 |
| range_w1 ↔ range_abs_w1 | 0.991 |
| range_w2 ↔ range_abs_w2 | 0.991 |
| var_x1 ↔ iqr_x1 | 0.991 |

## Full Pipeline Evaluation Results

Comprehensive testing revealed a critical insight: **Caustic features help HGB but hurt the full pipeline**.

### HGB Classifier Results (single-seed)

| Configuration | Features | MAE | Change |
|---------------|----------|-----|--------|
| No caustic | 183 | 22.93 | baseline |
| Joint only | 186 | 22.48 | +2.0% better |
| All caustic (9) | 210 | **22.37** | **+2.4% better** |

### Full Pipeline Results (single-seed)

| Configuration | Features | Selective MAE | Coverage |
|---------------|----------|---------------|----------|
| **No caustic** | 183 | **5.92** | **40.0%** |
| Joint only | 186 | 6.53 | 42.2% |
| All caustic (9) | 210 | 6.26 | 34.4% |

### Analysis

The discrepancy occurs because:
1. **CNN is sensitive to input dimensions** - Adding features hurts the sequence model
2. **Quality prediction changes** - Different features affect acceptance decisions
3. **Model agreement shifts** - CNN/HGB agreement (used for confidence) changes

### Key Insight

While `joint_concentration` ranks #1 in HGB feature importance, the full pipeline performs best **without** caustic features because the CNN component is negatively affected.

## Recommendations

1. **For full pipeline (production)**: Use `include_caustic=False` - best selective MAE
2. **For HGB-only use cases**: Use `include_caustic=True` with all features
3. **For experimentation**: Use `caustic_subset` parameter to test specific features
4. **Focus on theta features**: th1 and th2 statistics dominate importance
5. **Consider feature reduction**: Many features are highly correlated

## Files

- Full statistics: `analysis/feature_stats.csv`
- Correlation pairs: `analysis/correlations.txt`
- Analysis script: `scripts/analyze_features.py`
- Causticness investigation: `scripts/investigate_causticness.py`
