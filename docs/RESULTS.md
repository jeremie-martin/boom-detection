# Boom Detection: Results Summary

## Best Result: MAE 6.4 ± 0.5 frames (Robust Evaluation)

The goal is producing **high-quality animations for YouTube/social media**, not detecting boom frames on all simulations. This means:
- We only care about high-quality simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we want accurate boom detection

### Deployable Pipeline

```
At inference:
1. CNN → boom_cnn
2. HistGBM → boom_hgb
3. IF |boom_cnn - boom_hgb| ≤ 5:
   4. Extract features around avg(boom_cnn, boom_hgb) [±25 frames]
   5. Select top 50 quality-correlated features
   6. Predict quality using Random Forest
   7. IF predicted_quality ≥ 0.55:
      → ACCEPT, use boom_cnn as final prediction
   ELSE:
      → REJECT
ELSE:
   → REJECT
```

### Performance (Robust Multi-Seed Evaluation)

Results from 5-fold CV × 5 random seeds = 25 evaluations:

| Metric | Mean ± Std |
|--------|-----------|
| MAE | **6.4 ± 0.5 frames** |
| Within 5 frames | 63% ± 6% |
| Acceptance rate | 35% ± 5% |

### Key Improvements

| Change | MAE | Acceptance | Notes |
|--------|-----|------------|-------|
| Baseline (HGB prediction) | 7.2 ± 1.1 | 33% | Original pipeline |
| + Use CNN prediction | 7.5 ± 0.6 | 31% | CNN more accurate |
| + RF for quality | 6.8 ± 0.4 | 35% | Better quality prediction |
| + Top 50 features, ±25 window | **6.7 ± 0.6** | **34%** | Less overfitting |

---

## Ablation Study Results

### Which prediction to use on agreement cases?

| Prediction | MAE | Variance | Acceptance |
|------------|-----|----------|------------|
| **CNN** | **7.1 ± 0.7** | Low | 29% |
| HGB | 11.0 ± 4.5 | High | 32% |
| Average | 7.9 ± 2.6 | Medium | 32% |

**Conclusion**: CNN is more accurate and has lower variance.

### Impact of each filter

| Configuration | MAE | Acceptance |
|---------------|-----|------------|
| No filtering | 16.1 | 100% |
| Agreement≤5 only | 10.5 ± 3.3 | 47% |
| Agreement≤5 + Quality≥0.55 | **6.7 ± 1.0** | 31% |
| Agreement≤3 only | 8.4 ± 1.2 | 34% |

**Conclusion**: Both filters contribute. Agreement is the main driver; quality adds ~3 frame improvement.

### CNN vs HGB standalone (all simulations)

| Model | MAE |
|-------|-----|
| CNN | 16.1 ± 2.0 |
| HGB | 18.8 ± 1.3 |
| Ensemble | 17.6 ± 1.3 |

**Conclusion**: CNN outperforms HGB on average. HGB's value is as a confidence check.

---

## Quality Prediction Analysis

### Top features for quality (Spearman correlation)

| Feature | Correlation | Description |
|---------|-------------|-------------|
| d1_std_th1 | +0.649 | 1st derivative std of theta1 |
| d1_var_th1 | +0.569 | 1st derivative var of theta1 |
| d1_var_x1 | +0.531 | 1st derivative var of x1 |
| kurt_th1 | -0.469 | Kurtosis of theta1 |

**Key insight**: Derivative features predict quality, but NOT boom timing. This suggests quality and timing use different signals.

### Quality model comparison

| Model | MAE | Spearman r |
|-------|-----|------------|
| Ridge (original) | 0.244 | 0.294 |
| Lasso | 0.239 | 0.168 |
| **Random Forest** | **0.176** | **0.454** |
| HGB | 0.224 | -0.173 |

**Best config**: Random Forest with top 50 features and ±25 frame window.

---

## Feature Importance for Boom Detection

### Top features (Random Forest importance)

| Feature | Importance | Description |
|---------|------------|-------------|
| std_th1 | 0.080 | Std of theta1 |
| var_th1 | 0.080 | Variance of theta1 |
| var_th2 | 0.079 | Variance of theta2 |
| range_w2 | 0.057 | Range of omega2 |
| tip_area | 0.056 | Convex hull area of tips |

### Feature group importance

| Group | Total Importance | # Features |
|-------|-----------------|------------|
| range | 0.31 | 10 |
| var | 0.25 | 10 |
| std | 0.16 | 8 |
| iqr | 0.11 | 8 |
| tip | 0.09 | 3 |
| derivatives (d1, d2) | 0.02 | 122 |

**Key insight**: Derivatives are NOT important for boom detection, but ARE important for quality. Variance and range features dominate boom detection.

---

## Individual Model Performance

Without filtering (all simulations):

| Model | MAE | MedAE | Within 10 |
|-------|-----|-------|-----------|
| Variance threshold | 31.2 | - | - |
| HistGBM | 17.0 | 8.0 | 61% |
| CNN | 14.3 | 8.0 | 55% |
| Mean(CNN+HGB) | 13.3 | 7.5 | 59% |

---

## Key Discoveries

### 1. Quality-Error Correlation

Boom quality strongly predicts detection error (Spearman = -0.454):

| Quality Group | MAE | Within 10 |
|---------------|-----|-----------|
| High (≥0.5) | 11.2 | 67% |
| Low (<0.5) | 31.1 | 26% |

### 2. Model Agreement as Confidence

When CNN and HistGBM agree within 5 frames:
- ~50% of simulations
- MAE drops to 7-8
- Strong correlation with quality

### 3. Different Features for Different Tasks

| Task | Important Features |
|------|-------------------|
| Boom detection | var, std, range, tip |
| Quality prediction | derivatives (d1_*) |

This suggests we should use specialized feature sets.

---

## Progress History

| Phase | MAE | Approach |
|-------|-----|----------|
| Baseline | 18.9 | HistGBM |
| + Hyperparameters | 16.2 | Tuned HistGBM |
| + Ensemble | 14.0 | CNN+HGB mean |
| + Local context | 13.3 | Enhanced features |
| + Agreement filter | ~10 | Accept when models agree |
| + Quality filter | ~7 | Reject low predicted quality |
| + Use CNN | 7.5 | CNN > HGB when agree |
| + Improved quality model | 6.7 | RF, top 50, ±25 window |
| **+ Optimized CNN** | **6.4 ± 0.5** | hidden=64, kernels=(5,11,21) |

---

## Running the Pipeline

```bash
# Evaluate with cross-validation (5 seeds, ~5 min)
uv run python -m boom_detection.deploy_pipeline data --evaluate

# Quick single-seed evaluation (for development)
uv run python -m boom_detection.deploy_pipeline data --evaluate --quick

# Train final models
uv run python -m boom_detection.deploy_pipeline data --train --output models/
```

See `src/boom_detection/deploy_pipeline.py` for implementation details.
