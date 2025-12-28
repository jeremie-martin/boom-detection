# Boom Detection: Results Summary

## Best Result: MAE 4.0 (Deployable)

The goal is producing **high-quality animations for YouTube/social media**, not detecting boom frames on all simulations. This means:
- We only care about high-quality simulations
- Rejecting low-quality simulations is acceptable (can generate more)
- For accepted simulations, we need MAE close to 5 frames

### Deployable Pipeline

```
At inference:
1. CNN → boom_cnn
2. HistGBM → boom_hgb
3. IF |boom_cnn - boom_hgb| ≤ 5:
   4. Extract features around avg(boom_cnn, boom_hgb)
   5. Predict quality
   6. IF predicted_quality ≥ 0.55:
      → ACCEPT, use boom_hgb as final prediction
   ELSE:
      → REJECT
ELSE:
   → REJECT
```

### Performance

| Configuration | Accepted | MAE | Within 5 | Within 3 |
|--------------|----------|-----|----------|----------|
| Agree≤5, PredQ≥0.55 | 27% | **4.0** | 77% | 62% |
| Agree≤9, PredQ≥0.6 | 29% | 4.1 | 79% | 57% |
| Agree≤5, PredQ≥0.6 | 24% | 4.25 | 75% | 58% |

### Key Insights

1. **Model agreement is the primary filter** - when CNN and HistGBM disagree, predictions are unreliable
2. **Use HGB prediction, not average** - HGB alone is more accurate than CNN+HGB average when they agree
3. **Predicted quality helps** - adds ~1-2 frame improvement over agreement-only
4. **Quality threshold ~0.55 works best** - higher than initially expected

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

When CNN and HistGBM agree within 10 frames:
- 60% of simulations
- MAE drops to 6.8-7.2
- Strong correlation with quality (Spearman = -0.498)

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
| + Agreement filter | 7.0 | Accept when models agree |
| **+ Predicted quality** | **4.0** | Full deployable pipeline |

---

## Running the Pipeline

```bash
# Evaluate with cross-validation
uv run python -m boom_detection.deploy_pipeline data --evaluate

# Train final models
uv run python -m boom_detection.deploy_pipeline data --train --output models/
```

See `src/boom_detection/deploy_pipeline.py` for implementation details.
