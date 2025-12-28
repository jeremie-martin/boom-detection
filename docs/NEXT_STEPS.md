# Next Steps: Areas to Explore

## Current State

Best pipeline achieves **MAE 6.4 ± 0.5** on ~35% of simulations using:
1. CNN (hidden=64, kernels=(5,11,21)) + HGB agreement filter
2. Quality prediction using Random Forest on top 50 features
3. CNN prediction for final boom frame

## Completed Investigations

### CNN Architecture Optimization ✓
- **Finding**: Larger kernels (5,11,21) and hidden_dim=64 work best
- **Impact**: MAE improved from 6.7 to 6.4
- **Key insight**: Longer-range temporal patterns matter for boom detection

### Quality Prediction Improvement ✓
- **Finding**: Random Forest with top 50 features and ±25 window beats Ridge
- **Impact**: Improved acceptance (31% → 35%) while maintaining MAE
- **Key insight**: Derivative features (d1_*) predict quality, not variance/range

### Feature Importance Analysis ✓
- **Finding**: Different features matter for different tasks
  - Boom detection: var, std, range, tip features
  - Quality prediction: derivative features (d1_std_th1, d1_var_th1)
- **Impact**: Using specialized feature sets reduces overfitting

### CNN vs HGB Comparison ✓
- **Finding**: CNN (MAE 16.1) beats HGB (MAE 18.8) on all simulations
- **Finding**: CNN (MAE 7.1) beats HGB (MAE 11.0) on agreement cases
- **Impact**: Use CNN for final prediction, HGB only for confidence check

### Rejection Analysis ✓
- **Finding**: 51% fail agreement, 15% fail quality, 34% accepted
- **Finding**: 45% of agreement failures are HIGH quality simulations
- **Key insight**: Models genuinely struggle with some simulations (18+ frame error)
- **Impact**: Confirms filters work correctly; improving model accuracy helps most

---

## Priority Areas to Explore Next

### 1. Improve Individual Model Accuracy (HIGH PRIORITY)

**Why it matters**: Most rejections are due to model disagreement on genuinely hard cases. Better models would accept more simulations.

**Things to try**:
- [ ] Data augmentation for CNN (currently disabled)
- [ ] Different CNN architectures (deeper, attention)
- [ ] Better HGB features or tuning
- [ ] Train on only high-quality simulations
- [ ] Semi-supervised learning (use rejected predictions as pseudo-labels)

**Expected impact**: Could increase acceptance from 35% to 50%+ while maintaining MAE.

---

### 2. Feature Engineering (MEDIUM PRIORITY)

**Why it matters**: We use 183 features. There might be better ones.

**Things to try**:
- [ ] Velocity/acceleration features (not just position)
- [ ] Phase space features (angle-velocity pairs)
- [ ] Fourier transform features (frequency content)
- [ ] Entropy/complexity measures
- [ ] Cross-correlation between pendulums

**Key question**: What physical properties signal the boom?

---

### 3. Ensemble Strategies (MEDIUM PRIORITY)

**Why it matters**: Current simple agreement check might be suboptimal.

**Things to try**:
- [ ] Weighted average based on prediction confidence
- [ ] Stacking (meta-learner on top of CNN + HGB)
- [ ] Train multiple CNNs with different seeds
- [ ] Add more diverse base models

---

### 4. Understanding Hard Cases (LOW PRIORITY)

**Why it matters**: Some simulations are fundamentally hard. Understanding why could help.

**Things to try**:
- [ ] Visualize hard vs easy cases
- [ ] What features distinguish them?
- [ ] Is there a physical explanation?

---

## Lower Priority (Optimization)

These are fine-tuning and should be done last:

- [ ] Optimize agreement threshold (currently 5)
- [ ] Optimize quality threshold (currently 0.55)
- [ ] Optimize quality window (currently ±25)
- [ ] Optimize number of quality features (currently 50)
- [ ] Grid search on CNN hyperparameters

---

## Principles

- Always use multi-seed evaluation (5 seeds minimum)
- Check for data leakage before trusting any result
- Document findings in RESULTS.md
- Commit regularly with clear messages
- Focus on understanding, not just trying things blindly
- Prefer simple changes that give clear wins

---

## Progress Summary

| Date | MAE | Acceptance | Key Change |
|------|-----|------------|------------|
| Baseline | 18.9 | 100% | HistGBM only |
| +Agreement | ~10 | ~50% | CNN+HGB filter |
| +Quality | ~7 | ~30% | Quality filter |
| +CNN pred | 7.5 | 31% | Use CNN not HGB |
| +RF quality | 6.7 | 34% | Better quality model |
| **+CNN tuning** | **6.4** | **35%** | Larger kernels |
