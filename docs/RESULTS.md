# Boom Detection: Results Summary

## Best Result: MAE 7.5 ± 0.6 frames (Robust Evaluation)

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
   4. Extract features around avg(boom_cnn, boom_hgb)
   5. Predict quality
   6. IF predicted_quality ≥ 0.55:
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
| MAE | **7.5 ± 0.6 frames** |
| Within 5 frames | 59% ± 6% |
| Acceptance rate | 31% ± 4% |

### Key Insights

1. **Model agreement is the primary filter** - when CNN and HistGBM disagree, predictions are unreliable
2. **Use CNN prediction, not HGB** - CNN is more accurate when models agree (see ablation study)
3. **Quality filter helps** - reduces MAE by ~2-3 frames but lowers acceptance
4. **Low variance** - switching from HGB to CNN reduced std from 1.1 to 0.6

---

## Ablation Study Results

### Which prediction to use on agreement cases?

| Prediction | MAE | Variance | Acceptance |
|------------|-----|----------|------------|
| **CNN** | **7.1 ± 0.7** | Low | 29% |
| HGB | 11.0 ± 4.5 | High | 32% |
| Average | 7.9 ± 2.6 | Medium | 32% |

**Conclusion**: CNN is more accurate and has lower variance. Use CNN.

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

**Conclusion**: CNN outperforms HGB on average. HGB's value is as a confidence check (agreement filter).

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

### 3. Complementary Model Strengths

| Model | High-Q MAE | Low-Q MAE |
|-------|-----------|-----------|
| CNN | **8.4** | 24.9 |
| HGB | 11.2 | **18.2** |

CNN excels on high-quality, HGB on low-quality.

---

## Feature Importance

Top features for HistGBM (feature indices):
- 1298, 662, 616, 1189, 28, 979, 657, 70

Key finding: Top 20-50 features outperform all 1365 for HistGBM, but CNN benefits from all features.

| Features | HistGBM MAE | CNN MAE |
|----------|-------------|---------|
| All 1365 | 18.9 | **15.7** |
| Top 50 | **16.2** | 20.4 |
| Top 20 | 17.1 | 20.2 |

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
| **+ Use CNN (ablation)** | **7.5 ± 0.6** | Final pipeline |

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
