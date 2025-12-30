# E1: CNN Architecture Variations

## Objective
Test different CNN architecture configurations.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)
- Combiner: ThresholdCombiner (std/s=15/t=0.70)

## Parameters Tested
- hidden_dim: [32, 64, 128]
- kernel_sizes: [(3, 7, 15), (5, 11, 21), (7, 15, 31)]
- dropout: [0.2, 0.3, 0.4]

## Results

| Configuration | MAE | Coverage |
|---------------|-----|----------|
| dropout=0.4 | 2.53 ± 0.15 | 10.9% |
| hidden=128 | 2.90 ± 0.63 | 10.7% |
| hidden=64 | 2.97 ± 0.06 | 10.9% |
| kernels=(5,11,21) | 2.97 ± 0.06 | 10.9% |
| dropout=0.3 | 2.97 ± 0.06 | 10.9% |
| kernels=(7,15,31) | 3.02 ± 0.03 | 9.1% |
| kernels=(3,7,15) | 3.07 ± 0.37 | 10.9% |
| dropout=0.2 | 3.57 ± 1.12 | 11.6% |
| hidden=32 | 4.05 ± 1.10 | 12.2% |

## Best Configuration

**dropout=0.4**: MAE 2.53 ± 0.15

## Analysis

**dropout=0.4 improves over baseline** by 0.44 frames.
Recommendation: Consider updating CNN architecture.

### Per-Parameter Sensitivity

**hidden_dim**: MAE range = 1.15 frames
**kernel_sizes**: MAE range = 0.10 frames
**dropout**: MAE range = 1.03 frames
