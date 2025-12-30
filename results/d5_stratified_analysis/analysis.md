# D5: Stratified Analysis by Quality

## Objective
Analyze performance across different quality strata to understand:
1. How performance varies by true quality
2. Whether low-quality simulations are correctly rejected
3. If acceptance is well-calibrated to quality

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)
- Combiner: ThresholdCombiner (std/s=15/t=0.70)

## Quality Distribution

- **low (0-0.3)**: 41 simulations (23%)
- **med-low (0.3-0.5)**: 27 simulations (15%)
- **med-high (0.5-0.7)**: 54 simulations (31%)
- **high (0.7-1.0)**: 53 simulations (30%)

## Results by Stratum

| Quality Stratum | N | Accepted | Coverage | MAE |
|-----------------|---|----------|----------|-----|
| low (0-0.3) | 123 | 0 | 0.0% | N/A |
| med-low (0.3-0.5) | 81 | 2 | 2.5% | 8.50 |
| med-high (0.5-0.7) | 162 | 16 | 9.9% | 4.06 |
| high (0.7-1.0) | 159 | 39 | 24.5% | 2.26 |

## Analysis

### Rejection Behavior

**Good calibration:** Low-quality simulations are rejected at higher rates.
- Low quality rejection: 100.0%
- High quality rejection: 75.5%

### Error by Quality

**High quality predictions are accurate:** MAE = 2.26

### Source of Accepted Predictions

- low (0-0.3): 0 (0% of accepted)
- med-low (0.3-0.5): 2 (4% of accepted)
- med-high (0.5-0.7): 16 (28% of accepted)
- high (0.7-1.0): 39 (68% of accepted)

### Recommendations

1. Consider tuning threshold to better reject low-quality simulations
2. Quality prediction may need improvement
