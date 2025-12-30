# E3: Quality Feature Selection

## Objective
Test different n_quality_features values for quality prediction.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- Total features available: 183
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)
- Combiner: ThresholdCombiner (std/s=15/t=0.70)

## Parameters Tested
- n_quality_features: [20, 30, 50, 75, 100, all (183)]

## Results

| n_features | MAE | Coverage |
|------------|-----|----------|
| n=20 | 2.90 ± 0.35 | 10.9% |
| n=50 | 2.97 ± 0.06 | 10.9% |
| n=30 | 3.85 ± 1.16 | 10.9% |
| n=75 | 4.11 ± 1.41 | 9.5% |
| n=100 | 4.22 ± 0.90 | 10.9% |
| all | 4.25 ± 1.31 | 10.5% |

## Best Configuration

**n=20**: MAE 2.90 ± 0.35

## Analysis

**n_quality_features changes have minimal impact** (within 0.2 of baseline).
Recommendation: Keep current n_quality_features=50.

### Trend Analysis

Fewer features generally work better (feature selection helps).
