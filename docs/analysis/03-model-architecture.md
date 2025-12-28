# Model Architecture Analysis

**Agent Task**: Analyze the model architectures in the boom-detection codebase.

**Date**: 2025-12-28

---

## Executive Summary

The CNN architecture is well-designed for this temporal detection task, achieving the best results (MAE 6.4). HistGBM serves well as a confidence check through model agreement. LSTM and Transformer are underexplored and likely need architectural adjustments. Model agreement is a solid heuristic but could be improved with soft confidence scores.

---

## 1. Frame-Level Models (HistGBM) - Appropriateness & Design

### Current Implementation
- `FrameLevelClassifier` and `FrameLevelRegressor` wrap scikit-learn's HistGradientBoosting
- Treats each frame independently: P(after_boom | features_t)
- Finds 0.5 probability crossing as boom prediction

### Assessment: GOOD FIT, but with caveats

**Strengths:**
- Fast training (500x faster than GradientBoosting)
- Handles high-dimensional features well (~183 features)
- Robust to missing data
- Good feature importance extraction
- Appropriate for frame-level binary classification

**Weaknesses & Opportunities:**
1. **Temporal context ignored**: Each frame treated in isolation. The boom is fundamentally a *temporal event* (sharp variance spike), so this is suboptimal
2. **Simple aggregation**: Using 0.5 probability crossing is reasonable, but doesn't account for confidence
3. **Hyperparameters**: Default max_depth=5 is conservative; results show better performance with depth=7
4. **No uncertainty quantification**: Binary predictions don't indicate confidence

**Recommendations:**
- Add temporal context to frame features (sliding window statistics, momentum)
- Use probability calibration to quantify confidence
- Consider frame ordering constraint (monotonicity: once P=1, stay 1)
- Alternative: CatBoost (handles categorical features better, similar speed to HistGBM)

---

## 2. Sequence Models (CNN, LSTM, Transformer) - Architecture Assessment

### CNN Architecture (Best Performer: MAE 6.4)

```python
Multi-scale branches: kernels (5, 11, 21)
hidden_dim=64, 2 conv layers per branch, BatchNorm + ReLU
```

**Assessment: WELL-DESIGNED, with minor improvements possible**

**Strengths:**
- Multi-scale kernels capture temporal patterns at different ranges
- 5 frames ≈ ~100ms (local convergence signature)
- 11 & 21 frames ≈ longer-range momentum
- BatchNorm prevents internal covariate shift
- Achieves best results on the task

**Weaknesses:**
1. **Limited receptive field**: Largest kernel=21, receptive field grows slowly with depth
   - Only 2 conv layers means limited stacking of abstractions
   - Simulations are ~100-600 frames; could benefit from larger context

2. **No explicit temporal modeling**: Convolutions operate locally; long-range dependencies rely on increasing depth

3. **Positional information lost**: Unlike Transformer, no explicit position encoding

4. **Data augmentation disabled**: In production (augment=False), missing regularization benefits

### LSTM Architecture

```python
Input projection → Bidirectional LSTM (2 layers, hidden_dim=128)
```

**Assessment: REASONABLE but underperforms CNN**

- Bidirectional LSTM is theoretically perfect for this (can look ahead to confirm boom)
- 2 layers = good depth for temporal modeling
- BUT: Results show worse performance than CNN

**Why might LSTM underperform?**
- LSTMs struggle with very long sequences (100-600 frames) without better initialization
- Vanishing/exploding gradients despite LSTM cells
- May need layer normalization (more modern than BatchNorm for RNNs)
- Difficulty learning which temporal features matter most

### Transformer Architecture

```python
Input projection → Positional encoding → Transformer (4 heads, 2 layers)
```

**Assessment: PROMISING but likely under-tuned**

**Strengths:**
- Self-attention directly models all temporal dependencies
- Position encoding handles sequential nature
- Parallel computation (faster than LSTM)
- Should excel at finding sharp transitions

**Weaknesses:**
- No results comparing Transformer to CNN/LSTM in the codebase
- Default 4 heads may be too few for 183-dimensional input
- No learnable positional encoding (fixed random initialization)
- May overfit on small dataset (49 simulations × ~200 frames = ~10k effective samples)

---

## 3. Model Agreement as Confidence Metric

### Current Use
Agreement threshold ≤5 frames between CNN and HistGBM

### Assessment: GOOD HEURISTIC, but limited theoretical basis

**Strengths:**
- Empirically works: improves from 16.1 MAE (all) → 7.8 MAE (agreement ≤5)
- Captures that independent models converging is a strong signal
- Simple, interpretable, cheap to compute
- Effectively identifies "hard cases" where models genuinely struggle

**Weaknesses:**
1. **Correlated errors**: CNN and HistGBM may fail on similar challenging cases
   - Both trained on same features → similar failure modes likely
   - No proof they're independent

2. **Arbitrary threshold**: ≤5 frames chosen empirically, not theoretically justified

3. **One-bit confidence**: Either agree or don't; no soft confidence score

4. **Asymmetric confidence**: Doesn't account for which model is more trustworthy
   - Results show CNN is more accurate (MAE 14.3 vs 17.0 standalone)
   - But used as 50-50 confidence check

### Better Approaches

1. **Prediction uncertainty**:
   - CNN: Use softmax entropy over probability curve
   - HistGBM: Use prediction variance across tree votes
   - Ensemble: Weight by individual model uncertainties

2. **Jackknife/Bootstrap confidence intervals**:
   - Train multiple CNN/HistGBM variants
   - Compute prediction variance → confidence

3. **Calibrated probability estimates**:
   - Isotonic regression on validation probabilities
   - Platt scaling

4. **Learned confidence weighting** (Adaptive Ensemble approach):
   - Instead of binary agreement check, learn soft weights
   - Cross-validation shows this works well in practice

5. **Model disagreement direction**:
   - When CNN > HGB: boom detection slower (different signal)
   - When CNN < HGB: boom detection faster
   - Could indicate error direction, not just magnitude

---

## 4. Ensemble Strategy Analysis

### Current: AdaptiveEnsemble
- Learns weights from CV performance
- Weighted average of predictions

**Weakness**: Not used in production pipeline
- Production uses hardcoded agreement check + selective prediction (CNN vs HGB)
- Learned ensemble doesn't beat this approach

### Assessment: Simple but suboptimal

**Issues:**
1. **Averaging frame predictions**: If CNN=50, HGB=55, average=52.5 - rounds to 52
   - Loss of model-specific signal

2. **No soft filtering**: Accepts all simulations, uses weights for all predictions
   - When agreement poor, weights can't rescue bad predictions

3. **Weight computation**: Uses validation MAE
   - HistGBM typically has higher MAE but complements CNN
   - Simple inverse MAE weights may not be optimal

### Better Ensemble Strategies

1. **Conditional prediction selection** (Current best: 6.4 MAE):
   - Use CNN when available (more accurate)
   - Fallback to HGB only if CNN unavailable
   - This is better than averaging → suggests asymmetric confidence is key

2. **Learned stacking regressor**:
   - Train meta-model on [CNN_pred, HGB_pred, confidence_scores] → final_prediction
   - Uses expert knowledge of when each model works well

3. **Mixture of Experts**:
   - Learn gating network P(CNN | features) from validation data
   - Dynamically weight predictions per simulation

4. **Disagreement-based selection**:
   - When |CNN - HGB| ≤ threshold: use CNN (higher accuracy)
   - When |CNN - HGB| > threshold: reject (current behavior) or use quality-predicted outcome

---

## 5. Hyperparameter Choices - Critical Assessment

### Frame Models (HistGBM)
- max_depth=7 ✓ (better than default 5)
- max_iter=200 ✓ (reasonable)
- Early stopping not used ✗ (could add)

### CNN
- hidden_dim=64 ✓ (good balance)
- kernel_sizes=(5, 11, 21) ✓ (multi-scale sensible)
- dropout=0.3 ✓ (moderate regularization)
- **epochs=30, patience=5** ✗ (likely early stopping too aggressive)
  - 49 samples = ~400 training iterations
  - Patience=5 means ~20 iterations = very early stopping
  - Recommendation: patience=10-15 or use validation-based early stopping

- **batch_size=4** - Marginal (49 samples / 4 = ~12 batches; very small)
  - Recommendation: Try 8-16 for more stable gradients

### Quality Model (Random Forest)
- n_estimators=50 ✓ (reasonable)
- max_depth=5 ✓ (prevents overfitting)
- Feature selection (top 50 by correlation) ✓ (good)
- Window ±25 frames ✓ (empirically validated better than ±50)

### LSTM/Transformer
- LSTM hidden_dim=128 ✗ (too large for 49 samples, likely overfitting)
- Transformer n_heads=4 ✗ (too few; could use 8-16 with smaller d_model)
- No ablation studies shown for these architectures

---

## 6. Modern Architectures Not Tried

### 1. Temporal Convolutional Networks (TCN)
- 1D dilated convolutions for long-range temporal modeling
- Exponential receptive field growth (5→9→17→33... with dilation 1,2,4,8)
- Better than standard CNN for capturing boom event structure
- Example: WaveNet-style architecture

### 2. Transformer with positional encoding improvements
- Learnable positional encoding (instead of fixed random)
- Relative positional bias (more robust to sequence length variation)
- ALiBi (Attention with Linear Biases) for extrapolation

### 3. Attention-augmented CNN
- Combine CNN's efficiency with Transformer's modeling power
- Add self-attention modules to CNN features
- Better for detecting sharp transitions (boom signature)

### 4. 1D ConvLSTM
- Hybrid: convolutional gates + recurrent cells
- Captures both local spatial patterns and temporal dynamics

### 5. Temporal Fusion Transformer (TFT)
- Specifically designed for time-series forecasting
- Variable selection network to learn which features matter
- Decoder-encoder structure for multi-step predictions

### 6. Neural ODE
- Continuous-time dynamics modeling
- Better for understanding physical processes (pendulum physics)
- Can model variable frame rates

**Why CNN wins**: Boom is a *sharp, local event* (variance spike). CNN's inductive bias of local feature extraction is well-matched. LSTM assumes slowly-changing dynamics (more suitable for smooth processes). Transformer needs more data to beat CNN on small datasets.

---

## 7. Critical Issues & Quick Wins

**Current Best: MAE 6.4 ± 0.5 (35% acceptance)**

### Immediate improvements (estimated impact)

1. **Add temporal consistency loss to CNN**:
   - Encourage P(t) >= P(t-1) monotonically
   - Should reduce prediction variance, improve calibration
   - Est. MAE improvement: 5-10%

2. **Fix LSTM/Transformer underfitting**:
   - Add layer normalization
   - Reduce model size (hidden_dim → 32)
   - Use mixup or cutmix augmentation
   - Estimate: Could match CNN

3. **Implement soft confidence weighting**:
   - Instead of binary agreement, learn P(accept | |CNN-HGB|, quality_score)
   - Soft filtering should outperform hard thresholds
   - Est. acceptance improvement: +5% with similar MAE

4. **Probabilistic output**:
   - Add prediction intervals to CNN/HistGBM
   - Use ensemble variance as uncertainty
   - Filter based on confidence, not hard agreement
   - Est. MAE improvement: 5-15%

5. **Feature engineering**:
   - Add explicit temporal derivative features (rate of change)
   - Add curvature features (second derivatives)
   - Currently derivatives ignored (importance=0.02) - may be poorly engineered
   - Est. improvement: 5-10%

---

## Summary Matrix

| Component | Rating | Why | Recommendation |
|-----------|--------|-----|---|
| HistGBM | B+ | Fast, robust; but temporal-blind | Add temporal context features |
| CNN | A | Multi-scale, works well | Add monotonicity constraint, larger kernels |
| LSTM | C+ | Underperforming | Reduce size, add layer norm |
| Transformer | B | Underexplored | Learnable positional encoding, scale up |
| Agreement filter | B- | Works but hard-coded | Learn soft weighting |
| Ensemble strategy | B | Averaging suboptimal | Use selective prediction (CNN priority) |
| Quality predictor | A- | Random Forest good | Good feature selection |

**Highest ROI improvements**: (1) Soft confidence weighting, (2) Temporal feature engineering, (3) Fix LSTM/Transformer, (4) Add prediction uncertainty
