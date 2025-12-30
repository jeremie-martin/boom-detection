# B3: Model Agreement Analysis

## Objective
When models disagree, which one is correct? Understand model behavior
under different disagreement conditions.

## Configuration
- Seeds: [42, 43, 44]
- Total samples: 525 (5-fold CV x 3 seeds)
- Models: CNN + HGB + LSTM

## Overall Model Accuracy

| Model | MAE | Median Error | Win Rate |
|-------|-----|--------------|----------|
| CNN | 19.58 | 9.0 | 32.2% |
| HGB | 20.15 | 7.0 | 41.7% |
| LSTM | 18.50 | 8.0 | 26.1% |

## Accuracy by Disagreement Level

| Disagreement | N | CNN MAE | HGB MAE | LSTM MAE | Best Model |
|--------------|---|---------|---------|----------|------------|
| Low (<3) | 247 | 11.43 | 10.70 | 11.30 | hgb (46%) |
| Medium (3-8) | 156 | 20.77 | 20.41 | 20.40 | hgb (42%) |
| High (>=8) | 122 | 34.57 | 38.94 | 30.66 | cnn (34%) |

## LSTM Outlier Analysis

| Model | Times as Outlier | Percentage |
|-------|------------------|------------|
| cnn | 177 | 33.7% |
| hgb | 207 | 39.4% |
| lstm | 141 | 26.9% |

**When LSTM is the outlier (141 cases):**
- LSTM is correct: 59 (41.8%)
- LSTM is wrong: 82 (58.2%)
- LSTM MAE when outlier: 21.52
- CNN+HGB avg MAE when LSTM outlier: 23.10

## Key Insights

1. **Overall best model: LSTM** with lowest MAE
2. **At high disagreement: CNN** is most reliable
3. **LSTM outlier usually wrong** (58%) - using `std` metric helps

## Recommendations

- Continue using `std` metric for 3-model (robust to outliers)
- Consider switching `primary_model` to LSTM for better accuracy

## Plots
- `wins_by_disagreement.png`: Which model wins at each disagreement level
- `mae_by_disagreement.png`: Model error at each disagreement level
- `lstm_outlier_accuracy.png`: When LSTM is outlier, is it correct?
