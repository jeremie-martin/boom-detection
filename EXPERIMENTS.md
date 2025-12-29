# Experiment Results Log

This document tracks experimental results that inform but don't yet change the production configuration.
Results here may have insufficient sample size or need validation with a larger dataset.

**Note**: Coverage < 10% means fewer than ~9 samples per seed on our 90-simulation dataset.
These results are interesting but require validation on a larger dataset before becoming the "gold standard".

---

## 2024-12-29: Extended Parameter Sweep

**Run**: `runs/sweep_extended_20251229_224142/`

Tested parameters never swept before: weights, primary_model, score_function, extended thresholds.

### Key Findings

#### 1. Primary Model Selection

At threshold=0.70 with sqrt/s=15:

| Primary Model | MAE | Std | Coverage |
|---------------|-----|-----|----------|
| median | 2.98 | 0.41 | 10.0% |
| hgb | 3.02 | 0.39 | 10.0% |
| cnn | 3.04 | 0.70 | 10.0% |

**Finding**: Using median of predictions is slightly better than any individual model.
The improvement is small but consistent across coverage levels.

#### 2. Score Function Alternatives

Fixed config: sqrt/s=15

| Function | Best Threshold | MAE | Std | Coverage | Notes |
|----------|---------------|-----|-----|----------|-------|
| weighted | 0.75 | 2.41 | 0.64 | 4.4% | Low coverage |
| **min** | **0.65** | **2.86** | **1.42** | **9.3%** | Good tradeoff |
| product | 0.60 | 2.75 | 0.75 | 3.7% | Low coverage |

**Finding**: `score_function='min'` (both agreement AND quality must exceed threshold)
gives a different accuracy/coverage tradeoff. At 9.3% coverage, it achieves MAE 2.86.
Higher variance suggests it may be more sensitive to the specific test samples.

#### 3. Extended Thresholds with Sigmoid

Sigmoid transform at high selectivity:

| Scale | Threshold | MAE | Std | Coverage |
|-------|-----------|-----|-----|----------|
| 10 | 0.75 | 2.40 | 0.40 | 7.0% |
| 5 | 0.70 | 2.71 | 1.30 | 10.0% |
| 3 | 0.70 | 2.62 | 0.96 | 4.8% |

**Finding**: Sigmoid with scale=5, threshold=0.70 achieves MAE 2.71 at 10% coverage,
which improves on sqrt baseline (MAE 2.78-3.04). However, higher std (1.30) suggests
this may be less stable.

#### 4. Weight Combinations

Best weight config: agreement=0.2, quality=0.7 -> MAE 1.92 at 2.2% coverage.
However, this is too selective to be useful. The default 0.4/0.6 split appears reasonable.

---

## 2024-12-29: LSTM+HGB Pipeline Test

**Run**: `runs/sweep_lstm_hgb_20251229_224354/`

**Hypothesis**: LSTM is documented as best individual model (MAE 18.3 vs CNN 20.2).
Does LSTM+HGB outperform CNN+HGB in a 2-model pipeline?

### Result: NEGATIVE

| Pipeline | Best Config | MAE | Std | Coverage |
|----------|-------------|-----|-----|----------|
| LSTM+HGB | sigmoid/s=20/t=0.75 | 4.51 | 1.52 | 10.0% |
| CNN+HGB (baseline) | sqrt/s=15/t=0.70 | ~3.0 | ~0.7 | 10.0% |

**LSTM+HGB is ~50% worse than CNN+HGB** despite LSTM being the best individual model.

### Why?

1. **Individual performance != ensemble synergy**: LSTM and HGB may make correlated errors
2. **CNN provides complementary signal**: CNN's different architecture catches cases HGB misses
3. **Agreement metric noise**: With PRODUCTION_CONFIG (max_pendulums=2000), LSTM predictions
   may be less stable, adding noise to the agreement calculation

**Conclusion**: Stick with CNN+HGB. Individual model performance doesn't predict ensemble performance.

---

## 2024-12-29: Quality Model Parameter Sweep

**Run**: `runs/sweep_quality_params_20251229_224629/`

Tested `quality_window` and `jitter_std` - parameters that affect quality model training.
Note: This requires retraining (not free like combiner sweeps).

### Results Summary

| Window | Jitter | MAE | Std | Coverage | Notes |
|--------|--------|-----|-----|----------|-------|
| **35** | **10** | **2.70** | **0.34** | 8.1% | **Best MAE, lowest std** |
| 15 | 0 | 2.75 | 0.86 | 10.4% | Good coverage |
| 15 | 5 | 2.91 | 0.77 | 8.9% | |
| 25 | 0 | 2.98 | 1.32 | 9.3% | |
| **25** | **5** | **3.04** | **0.70** | **10.0%** | **Current default** |
| 35 | 3 | 3.04 | 0.23 | 9.6% | Very stable |
| 25 | 3 | 3.06 | 0.27 | 11.5% | Stable, good coverage |

### Key Findings

1. **Best config (35/10)**: 11% better MAE than default (2.70 vs 3.04), with **half the variance**
2. **Coverage caveat**: Best config has 8.1% coverage (< 10% threshold)
3. **Stability sweet spot**: window=35, jitter=3 has very low std (0.23) at 9.6% coverage
4. **Window=50 is too large**: Performance degrades, especially with high jitter

### Interpretation

- Larger quality_window (35 vs 25) captures more context around the boom
- Higher jitter_std (10) provides regularization during training
- The combination (35/10) seems to make the quality model more robust

### Recommendation

**Pending validation on larger dataset**. The improvement is significant but:
- 8.1% coverage means ~7 samples per seed - need more data to confirm
- Should test on expanded dataset before changing defaults

---

## Summary: Promising Configurations to Validate

These configurations show improvement but need validation on a larger dataset:

### For High Accuracy (< 10% coverage)

| Config | MAE | Coverage | Notes |
|--------|-----|----------|-------|
| quality_window=35, jitter_std=10 | 2.70 | 8.1% | Best overall |
| sigmoid/s=10/t=0.75 | 2.40 | 7.0% | From combiner sweep |
| score_function='min', t=0.65 | 2.86 | 9.3% | Alternative approach |

### For Moderate Coverage (10-15%)

| Config | MAE | Coverage | Notes |
|--------|-----|----------|-------|
| sigmoid/s=5/t=0.70 | 2.71 | 10.0% | Transform alternative |
| window=25, jitter=3 | 3.06 | 11.5% | Very stable (std=0.27) |
| primary_model='median' | 2.98 | 10.0% | Use prediction median |

### Follow-up Experiments Needed

1. **Combine best findings**: quality_window=35, jitter_std=10 WITH sigmoid transform
2. **Test on larger dataset**: Validate low-coverage results
3. **score_function='min' deeper sweep**: Test more threshold values

---

## Negative Results (Don't Repeat)

| Experiment | Finding |
|------------|---------|
| LSTM+HGB pipeline | 50% worse than CNN+HGB |
| Specialized models (hgb_0.5) | Worse than baseline |
| 3-model pipeline | No improvement, adds noise |
| quality_window=50 | Too large, performance degrades |
