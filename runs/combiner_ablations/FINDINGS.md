# Combiner Ablation Experiment Findings

**Date**: December 2025
**Seeds**: 42, 43, 44 (3-seed evaluation)
**Simulations**: 90

## Executive Summary

Three major findings that change our understanding of the pipeline:

1. **3-model with `std` metric WORKS** - MAE 3.27 at 12.2% beats 2-model (MAE 3.38 at 13.7%)
2. **Quality is more important than agreement** - Weights 0.2/0.8 beat current 0.4/0.6
3. **Quality-only gating is surprisingly competitive** - Simpler and nearly as effective

---

## Experiment 1: std vs range Disagreement Metric

### 2-Model Results
For 2 models, `range = 2 × std` mathematically, so properly scaled configs are equivalent:

| Metric | Scale | MAE | Coverage |
|--------|-------|-----|----------|
| range | 5 | 8.37 ± 5.39 | 10.7% |
| range | 10 | 5.73 ± 2.99 | 21.5% |
| range | 15 | 7.28 ± 4.10 | 30.0% |
| std | 2.5 | 8.37 ± 5.39 | 10.7% |
| std | 5 | 5.73 ± 2.99 | 21.5% |
| std | 7.5 | 7.28 ± 4.10 | 30.0% |

**Conclusion**: For 2-model, std and range are mathematically equivalent when scaled appropriately.

### 3-Model Results - THE PARADOX RESOLVED!

| Metric | Scale | MAE | Coverage |
|--------|-------|-----|----------|
| **std** | **5** | **3.27 ± 0.56** | **12.2%** |
| std | 7.5 | 4.67 ± 0.31 | 24.8% |
| range | 10 | 3.77 ± 0.55 | 8.9% |
| range | 15 | 3.92 ± 0.73 | 19.3% |

**Key Finding**: 3-model with `std` scale=5 achieves **MAE 3.27** at 12.2% coverage!

This **beats the 2-model best** (MAE 3.38 ± 0.83 at 13.7%) while using the additional LSTM model.

**Why std works for 3-model**:
- `range` = max - min → inflates dramatically when LSTM disagrees (outlier effect)
- `std` = standard deviation → more robust to outliers
- The 3-model disagreement correlates better with error (r=0.56 vs r=0.44) - using std captures this without inflation

---

## Experiment 2: Simpler Baselines

### Results

| Strategy | MAE | Coverage | Notes |
|----------|-----|----------|-------|
| MedianCombiner (no rejection) | 20.75 | 100% | Baseline |
| **QualityGated(0.7)** | **4.38 ± 1.53** | **12.2%** | Very competitive! |
| QualityGated(0.6) | 7.00 ± 2.46 | 44.4% | |
| QualityGated(0.5) | 7.49 ± 1.96 | 53.3% | |
| AgreementGated(5) | 15.82 ± 1.37 | 56.7% | Poor |
| AgreementGated(10) | 16.74 ± 1.37 | 73.0% | Poor |
| MajorityVote | 20.26-20.33 | 100% | No rejection for 2 models |
| ThresholdCombiner(default) | 7.28 ± 4.10 | 30.0% | |

**Key Findings**:
1. **Quality-only gating is remarkably effective** - QualityGated(0.7) achieves MAE 4.38 at 12.2%
2. **Agreement-only gating is poor** - MAE ~16, worse than random selection
3. **Majority vote doesn't help** - 2 models always "agree" within any reasonable tolerance

**Insight**: Predicted quality is the dominant signal for acceptance. Agreement provides marginal benefit.

---

## Experiment 3: Weight Optimization

### Results

| Agreement Weight | Quality Weight | MAE | Coverage |
|-----------------|----------------|-----|----------|
| 0.0 | 1.0 | 7.14 ± 2.84 | 44.4% |
| **0.2** | **0.8** | **6.56 ± 3.23** | **38.9%** |
| 0.4 (current) | 0.6 | 7.28 ± 4.10 | 30.0% |
| 0.5 | 0.5 | 10.09 ± 4.48 | 28.5% |
| 0.6 | 0.4 | 10.95 ± 3.47 | 27.8% |
| 0.8 | 0.2 | 12.16 ± 3.90 | 26.7% |
| 1.0 | 0.0 | 13.32 ± 4.35 | 28.5% |

**Key Findings**:
1. **Quality dominates** - Performance degrades as agreement weight increases
2. **Current 0.4/0.6 is suboptimal** - Better to use 0.2/0.8 or pure quality
3. **Agreement-only (1.0/0.0) is worst** - Confirms agreement alone isn't useful

**Recommendation**: Consider changing default weights to 0.2/0.8 (or even 0.0/1.0).

---

## Recommendations

### Immediate Improvements

1. **For maximum accuracy**: Use 3-model with std metric
   - Config: `frame_models=('cnn', 'hgb', 'lstm')`, `disagreement_metric='std'`, `scale=5`
   - Expected: MAE 3.27 at 12.2% coverage

2. **For simplicity**: Use quality-only gating
   - Config: `QualityGatedCombiner(threshold=0.7)`
   - Expected: MAE 4.38 at 12.2% coverage
   - Benefit: Simpler, no model agreement needed

3. **For 2-model if keeping ThresholdCombiner**: Change weights to 0.2/0.8
   - Config: `agreement_weight=0.2, quality_weight=0.8`
   - Expected: Better MAE at similar coverage

### Updated Best Configurations

| Configuration | MAE | Coverage | Change from Current |
|---------------|-----|----------|---------------------|
| 3-model std/5/0.60 | 3.27 ± 0.56 | 12.2% | **NEW BEST** |
| 2-model range/5/0.60 (current best) | 3.38 ± 0.83 | 13.7% | Baseline |
| Quality-only 0.7 | 4.38 ± 1.53 | 12.2% | Simpler alternative |
| 2-model 0.2/0.8 weights | 6.56 ± 3.23 | 38.9% | Better than 0.4/0.6 |

---

## Why Quality Matters More Than Agreement

The results suggest that **predicted quality is a stronger signal** because:

1. Quality correlates with boom "clarity" - clear booms are easier to predict accurately
2. Both CNN and HGB benefit from the same underlying feature quality
3. When models disagree, the _reason_ may not always be predictive uncertainty but different model biases

Agreement helps most when:
- Models have independent error modes
- Disagreement indicates genuine uncertainty

But in our case, agreement adds noise because:
- LSTM's errors don't seem to complement CNN/HGB errors
- High disagreement seems to occur even with accurate individual predictions

---

## Future Work

1. **Learned combiner**: Train a model to directly predict acceptance from features
2. **Calibrated quality**: Improve quality prediction to be better calibrated
3. **Model-specific weights**: Weight each model's vote differently based on confidence
