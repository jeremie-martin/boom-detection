# Next Steps: Areas to Explore

## Current State

Best pipeline achieves **MAE 6.7 ± 0.6** on ~34% of simulations using:
1. CNN + HGB agreement filter (reject if |CNN - HGB| > 5)
2. Quality prediction filter using Random Forest on top 50 features
3. CNN prediction for final boom frame

## Completed Investigations

### Quality Prediction Improvement ✓
- **Finding**: Random Forest with top 50 features and ±25 window beats Ridge
- **Impact**: Improved acceptance (31% → 34%) while maintaining MAE
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

---

## Priority Areas to Explore Next

### 1. Alternative Boom Detection Models (HIGH PRIORITY)

**Why it matters**: Current CNN is good but might not be optimal.

**Things to try**:
- [ ] Deeper/wider CNN architectures
- [ ] Attention mechanism on top of CNN
- [ ] Different optimizers/learning rates
- [ ] Training with augmentation (currently disabled)
- [ ] Different prediction strategy (regression vs classification)

**Key constraint**: Must be fast to train (< 2 minutes).

---

### 2. Feature Engineering (MEDIUM PRIORITY)

**Why it matters**: We use 183 features. There might be better ones.

**Things to try**:
- [ ] Velocity/acceleration features (not just position)
- [ ] Phase space features (angle-velocity pairs)
- [ ] Fourier transform features
- [ ] Entropy/complexity measures
- [ ] Cross-correlation between pendulums

**Key question**: What physical properties signal the boom?

---

### 3. Understanding Hard Cases (MEDIUM PRIORITY)

**Why it matters**: We reject 66% of simulations. Why?

**Things to try**:
- [ ] Analyze feature distributions for easy vs hard cases
- [ ] What makes CNN and HGB disagree?
- [ ] Can we predict which simulations will be hard?
- [ ] Is there a pattern in the worst predictions?

**Key question**: Is there a fundamental limit to detection accuracy?

---

### 4. Ensemble Strategies (LOW PRIORITY)

**Why it matters**: Current simple agreement check might be suboptimal.

**Things to try**:
- [ ] Weighted average based on confidence
- [ ] Stacking (meta-learner on top of CNN + HGB)
- [ ] More diverse base models (Random Forest, etc.)

---

### 5. Pipeline Optimization (LOW PRIORITY)

**Why it matters**: Once individual components are good, tune how they work together.

**Things to try**:
- [ ] Optimize agreement threshold (currently 5)
- [ ] Optimize quality threshold (currently 0.55)
- [ ] Optimize quality window (currently ±25)
- [ ] Optimize number of quality features (currently 50)

**Note**: This is optimization, not fundamental improvement.

---

## Principles

- Always use multi-seed evaluation (5 seeds minimum)
- Check for data leakage before trusting any result
- Document findings in RESULTS.md
- Commit regularly with clear messages
- Focus on understanding, not just trying things blindly
- Prefer simple changes that give clear wins
