# D4: Ensemble Prediction Methods

## Objective
Test different methods for combining model predictions after acceptance.

## Background
The ThresholdCombiner decides acceptance based on quality + model agreement.
Once accepted, it returns a prediction using the `primary_model` setting:
- **cnn**: Return CNN's prediction (default)
- **hgb**: Return HGB's prediction
- **lstm**: Return LSTM's prediction
- **median**: Return median of all model predictions

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)
- Combiner: ThresholdCombiner (std/s=15/t=0.70)

## Results

| Primary Model | MAE | RMSE | Coverage |
|---------------|-----|------|----------|
| hgb | 2.18 ± 0.51 | 3.41 | 10.9% |
| median | 2.58 ± 0.23 | 3.87 | 10.9% |
| lstm | 2.81 ± 0.22 | 4.17 | 10.9% |
| cnn | 2.97 ± 0.06 | 4.25 | 10.9% |

## Best Configuration

**hgb**: MAE 2.18 ± 0.51

## Analysis

**hgb outperforms CNN** by 0.79 frames.

Recommendation: Switch to primary_model='hgb'.

### Sensitivity Analysis

MAE range across methods: 0.79 frames

Primary model choice has **significant impact** on performance.

### Why Models Differ

Even with high agreement (std metric), models can differ:
1. CNN: Good at local patterns, may miss global context
2. HGB: Frame-by-frame, no temporal modeling
3. LSTM: Long-range dependencies, but sometimes outliers
4. Median: Robust to outliers, compromise between models
