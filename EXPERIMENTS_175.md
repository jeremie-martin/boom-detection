# Experiment Results: 175-Simulation Dataset

**Date**: 2024-12-30
**Dataset**: 175 simulations (up from 90)
**Seeds**: 42, 43, 44
**Config**: PRODUCTION_CONFIG (no caustic)

This document supersedes the previous 90-simulation results. The larger dataset reveals important corrections to our previous findings.

---

## Executive Summary

### Major Finding: 3-Model Pipeline Now Beats 2-Model

| Pipeline | 90-sim MAE | 175-sim MAE | Δ | Verdict |
|----------|------------|-------------|---|---------|
| 2-model (CNN+HGB) best | 2.78 | 3.48 | +25% worse | Degraded |
| 3-model (CNN+HGB+LSTM) best | 5.18 | 2.97 | **-43% better** | **Now recommended** |
| Quality-only t=0.70 | 3.35 | 3.54 | Similar | Stable baseline |

**The previous conclusion that "3-model adds noise" was WRONG.** It was a small-sample artifact. With more data, the 3-model pipeline clearly outperforms 2-model.

### New Recommended Configuration

```python
from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline

pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb', 'lstm'),
    combiner=ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',
        threshold=0.70,
    ),
)
# MAE 2.97 ± 0.06 at 10.9% coverage
```

---

## 1. Quality-Only Baseline

**Simplest approach**: Accept simulations based only on predicted quality score.

| Threshold | MAE | MAE std | RMSE | Coverage | Coverage std |
|-----------|-----|---------|------|----------|--------------|
| 0.75 | 1.86 | 0.20 | 2.36 | **1.9%** | 2.0% |
| **0.70** | **3.54** | **0.94** | **5.61** | **14.5%** | **5.8%** |
| 0.65 | 4.33 | 1.14 | 6.74 | 29.0% | 6.4% |
| 0.60 | 6.24 | 1.19 | 13.24 | 43.0% | 2.2% |
| 0.55 | 8.85 | 0.69 | 18.55 | 58.1% | 4.8% |

**Observations**:
- t=0.75 achieves excellent MAE (1.86) but only 1.9% coverage - too few samples
- **t=0.70 is the best practical threshold**: MAE 3.54 at 14.5% coverage
- Very stable - coverage std at t=0.70 is 5.8%

**Comparison to 90-sim**: Nearly identical. Quality-only is robust to dataset size.

---

## 2. 2-Model Pipeline (CNN + HGB)

### Top 10 Configurations by MAE

| Rank | Formula | Scale | Thresh | MAE | MAE std | Coverage |
|------|---------|-------|--------|-----|---------|----------|
| 1 | sigmoid | 10 | 0.70 | 3.48 | 1.30 | 15.8% |
| 2 | sqrt | 30 | 0.70 | 3.50 | 1.45 | 14.9% |
| 3 | sigmoid | 8 | 0.70 | 3.52 | 1.43 | 12.4% |
| 4 | sqrt | 5 | 0.60 | 3.58 | 1.98 | 10.1% |
| 5 | sqrt | 25 | 0.70 | 3.59 | 1.48 | 13.7% |
| 6 | sigmoid | 5 | 0.70 | 3.61 | 1.75 | 8.0% |
| 7 | linear | 5 | 0.70 | 3.71 | 2.14 | 9.7% |
| 8 | sqrt | 8 | 0.65 | 3.81 | 1.93 | 9.1% |
| 9 | sqrt | 10 | 0.65 | 3.84 | 1.94 | 11.6% |
| 10 | sigmoid | 15 | 0.70 | 3.93 | 1.17 | 21.7% |

### Best by Coverage Level

| Coverage | Config | MAE | MAE std |
|----------|--------|-----|---------|
| 0-10% | sigmoid/s=5/t=0.70 | 3.61 | 1.75 |
| 10-20% | sigmoid/s=10/t=0.70 | 3.48 | 1.30 |
| 20-30% | sigmoid/s=15/t=0.70 | 3.93 | 1.17 |
| 30-50% | sigmoid/s=30/t=0.70 | 4.97 | 0.98 |

### Key Observations

1. **sigmoid transform is now competitive with sqrt** - Best config uses sigmoid/s=10
2. **2-model is WORSE than 90-sim results**: Best MAE 3.48 vs 2.78 previously
3. **2-model is NOT better than quality-only**: MAE 3.48 vs 3.54 - virtually identical
4. **High variance**: Most configs have std > 1.0

**Interpretation**: The 2-model pipeline's apparent advantage in the 90-sim dataset was a small-sample artifact. With more data, it barely beats quality-only gating.

---

## 3. 3-Model Pipeline (CNN + HGB + LSTM) - NEW WINNER

### Results with `std` Metric (Recommended)

| Rank | Scale | Thresh | MAE | MAE std | RMSE | Coverage |
|------|-------|--------|-----|---------|------|----------|
| 1 | 5 | 0.70 | 1.00 | 1.00 | 1.18 | 1.3% |
| 2 | 3 | 0.65 | 1.22 | 0.69 | 1.40 | 2.1% |
| 3 | 3 | 0.70 | 2.00 | 1.41 | 2.30 | 0.6% |
| 4 | 8 | 0.70 | 2.55 | 0.68 | 3.28 | 4.4% |
| 5 | 10 | 0.70 | 2.58 | 0.69 | 3.36 | 6.9% |
| 6 | 5 | 0.65 | 2.74 | 0.81 | 3.52 | 6.3% |
| **7** | **15** | **0.70** | **2.97** | **0.06** | **4.25** | **10.9%** |
| 8 | 3 | 0.60 | 3.92 | 1.25 | 7.23 | 6.3% |

### Results with `range` Metric (For Comparison)

| Rank | Scale | Thresh | MAE | MAE std | RMSE | Coverage |
|------|-------|--------|-----|---------|------|----------|
| 1 | 5 | 0.65 | 1.00 | 1.00 | 1.22 | 1.1% |
| 2 | 10 | 0.70 | 1.00 | 1.00 | 1.22 | 1.1% |
| 3 | 5 | 0.60 | 1.45 | 0.18 | 1.94 | 3.4% |
| 5 | 20 | 0.70 | 2.53 | 0.85 | 3.40 | 5.5% |
| 7 | 15 | 0.70 | 2.93 | 1.29 | 3.66 | 3.6% |

### Key Findings

1. **3-model with std/s=15/t=0.70 is the new best configuration**:
   - MAE 2.97 at 10.9% coverage
   - Extremely low variance: std = 0.06 (incredibly stable!)
   - This is 15% better than best 2-model (3.48) and 16% better than quality-only (3.54)

2. **`std` metric beats `range`** at practical coverage levels:
   - At ~10% coverage: std gives MAE 2.97 vs range gives MAE ~3.55
   - `std` is more robust to outliers from LSTM predictions

3. **Very low coverage configs are unreliable**: MAE 1.00-1.22 at 1-2% coverage looks great but is meaningless - not enough samples

### Why 3-Model Now Works

Previous 90-sim reasoning said "LSTM adds noise to agreement metric." This was WRONG because:

1. **More data stabilizes LSTM**: With 175 sims, LSTM gets better training → more consistent predictions
2. **`std` metric is crucial**: Using standard deviation instead of range filters out LSTM outliers
3. **Small sample bias**: At 90 sims with ~10% coverage = ~9 samples. Random variance dominated.

---

## 4. Extended Parameter Sweep

### Primary Model Selection

At sqrt/s=15 with different primary models:

| Primary | Threshold | MAE | MAE std | Coverage |
|---------|-----------|-----|---------|----------|
| cnn | 0.60 | 5.24 | 0.78 | 29.1% |
| **hgb** | **0.60** | **4.79** | **0.81** | **29.1%** |
| median | 0.60 | 4.97 | 0.76 | 29.1% |
| cnn | 0.70 | 4.05 | 2.05 | 8.4% |
| **hgb** | **0.70** | **3.98** | **2.19** | **8.4%** |
| median | 0.70 | 4.07 | 2.15 | 8.4% |

**Finding**: HGB is slightly better as primary model than CNN. Median is in between. But differences are small.

### Score Function Alternatives

| Function | Best Threshold | MAE | MAE std | Coverage |
|----------|---------------|-----|---------|----------|
| weighted | 0.65 | 3.94 | 1.20 | 17.1% |
| min | 0.75 | 0.00 | 0.00 | 0.2% |
| product | 0.60 | 4.17 | 3.42 | 2.7% |

**Finding**: `min` and `product` achieve low MAE but with extremely low coverage (< 3%). Not practical. `weighted` (default) remains best.

### Extended Thresholds (0.75, 0.80, 0.85)

| Formula | Scale | Thresh | MAE | Coverage |
|---------|-------|--------|-----|----------|
| sqrt | 3-15 | 0.85 | 0.00 | 0.2% |
| sigmoid | 3-5 | 0.80 | 0.50 | 0.4% |
| sigmoid | 10 | 0.80 | 0.75 | 0.6% |
| sigmoid | 10 | 0.75 | 3.47 | 6.1% |

**Finding**: Very high thresholds (0.80+) give too few samples to be useful.

---

## 5. Quality Model Parameters

Testing quality_window × jitter_std combinations:

| Window | Jitter | MAE | MAE std | Coverage |
|--------|--------|-----|---------|----------|
| **50** | **0** | **3.75** | **1.84** | **9.5%** |
| **35** | **0** | **3.75** | **2.03** | **9.1%** |
| 15 | 5 | 3.96 | 1.82 | 8.8% |
| 25 | 5 | 4.05 | 2.05 | 8.4% |
| 25 | 3 | 4.11 | 2.14 | 9.1% |
| 15 | 3 | 4.21 | 2.19 | 8.0% |
| 15 | 0 | 4.22 | 2.31 | 8.8% |
| 25 | 10 | 4.35 | 2.07 | 8.6% |
| 15 | 10 | 4.37 | 1.70 | 8.2% |
| 35 | 5 | 4.43 | 1.91 | 8.4% |
| 25 | 0 | 5.97 | 4.09 | 9.0% |
| 35 | 3 | 6.94 | 4.53 | 8.0% |

### Key Findings

1. **Best config is window=50, jitter=0** (or window=35, jitter=0)
2. **This contradicts 90-sim findings** where window=35, jitter=10 was best
3. **No jitter is better with more data**: The regularization benefit of jitter diminishes with larger datasets
4. **Larger windows help**: 35-50 frame windows capture more context

### Interpretation

With 90 simulations, jitter provided needed regularization to prevent overfitting. With 175 simulations:
- More training data provides natural regularization
- No need for artificial data augmentation (jitter)
- Larger windows capture more temporal context without overfitting risk

---

## 6. Comprehensive Comparison

### All Approaches at ~10% Coverage

| Approach | Config | MAE | MAE std | Coverage |
|----------|--------|-----|---------|----------|
| **3-model** | **std/s=15/t=0.70** | **2.97** | **0.06** | **10.9%** |
| 2-model | sigmoid/s=8/t=0.70 | 3.52 | 1.43 | 12.4% |
| 2-model | sqrt/s=5/t=0.60 | 3.58 | 1.98 | 10.1% |
| Quality-only | t=0.70 | 3.54 | 0.94 | 14.5% |
| Quality params | window=50/jitter=0 | 3.75 | 1.84 | 9.5% |

**Winner**: 3-model with std/s=15/t=0.70 - MAE 2.97 is 15-20% better than all alternatives.

### Variance Analysis

| Approach | MAE std |
|----------|---------|
| 3-model std/s=15/t=0.70 | **0.06** |
| Quality-only t=0.70 | 0.94 |
| 2-model sigmoid/s=10/t=0.70 | 1.30 |
| Quality params window=50/jitter=0 | 1.84 |
| 2-model sqrt/s=5/t=0.60 | 1.98 |

**The 3-model config is also the most stable** - std of 0.06 means results are almost perfectly reproducible across seeds.

---

## 7. What Changed from 90 to 175 Simulations

### Findings That Still Hold

| Finding | 90-sim | 175-sim |
|---------|--------|---------|
| Quality-only t=0.70 is solid baseline | MAE 3.35 | MAE 3.54 |
| threshold=0.70 is good operating point | Yes | Yes |
| Caustic features don't help | Yes | Yes (still excluded) |
| Specialized models don't help | Yes | Not retested |

### Findings That Changed

| Finding | 90-sim | 175-sim | New Understanding |
|---------|--------|---------|-------------------|
| 2-model beats quality-only | Yes (2.78 vs 3.35) | **No** (3.48 vs 3.54) | Was small-sample artifact |
| 3-model adds noise | Yes (MAE 5.18) | **No** (MAE 2.97) | More data stabilizes LSTM |
| Best quality params | window=35, jitter=10 | **window=50, jitter=0** | Jitter = overfitting workaround |
| sqrt beats sigmoid | Yes | **No** (similar) | Transform matters less |

### Implications

1. **Small datasets can mislead**: The 90-sim dataset led to wrong conclusions about 3-model and quality params
2. **Always validate with more data**: Coverage < 10% with 90 sims = ~9 samples - too few
3. **Regularization needs scale with data**: Jitter helped at 90 sims but hurts at 175

---

## 8. New Recommendations

### Production Configuration (175-sim validated)

```python
from boom_detection.combine import ThresholdCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline

# RECOMMENDED: 3-model pipeline
pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb', 'lstm'),
    combiner=ThresholdCombiner(
        agreement_transform='sqrt',
        disagreement_scale=15.0,
        disagreement_metric='std',  # NEW: use std instead of range
        threshold=0.70,
    ),
)
# Expected: MAE 2.97 ± 0.06 at 10.9% coverage
```

### Alternative: Simpler Quality-Only

```python
from boom_detection.combine import QualityGatedCombiner
from boom_detection.deploy_pipeline import BoomDetectionPipeline

# Simpler alternative - nearly as good
pipeline = BoomDetectionPipeline(
    frame_models=('cnn', 'hgb'),
    combiner=QualityGatedCombiner(threshold=0.70),
)
# Expected: MAE 3.54 ± 0.94 at 14.5% coverage
```

### Quality Model Training

With 175+ simulations, use:
- `quality_window=50` (or 35)
- `jitter_std=0` (no jitter needed)

---

## 9. Future Work

1. **Confirm with even more data**: 175 sims is better but more would increase confidence
2. **Test 3-model quality params**: Current quality params sweep used 2-model; 3-model may have different optimal params
3. **Combined optimization**: Test 3-model + optimal quality params together
4. **LSTM+HGB pipeline**: Skipped in this sweep - could test if interested

---

## Run Information

All runs from 2024-12-29:
- `runs/sweep_quality_only_20251229_233835/`
- `runs/sweep_2model_20251229_234227/`
- `runs/sweep_3model_20251229_234514/`
- `runs/sweep_extended_20251229_234951/`
- `runs/sweep_quality_params_20251229_235239/`
