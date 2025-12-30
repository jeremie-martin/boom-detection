# D3: GRU vs LSTM Comparison

## Objective
Compare GRU as an alternative to LSTM for the recurrent sequence model.

## Background
GRU (Gated Recurrent Unit) has fewer parameters than LSTM:
- LSTM: 3 gates (input, forget, output) + cell state
- GRU: 2 gates (reset, update) + no separate cell state

GRU often trains faster and can be similarly effective.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- Both use hidden_dim=64, n_layers=2, dropout=0.3
- 3-model combiner: ThresholdCombiner (std/s=15/t=0.70)
- Single model combiner: QualityGatedCombiner (t=0.70)

## Results

| Configuration | MAE | RMSE | Coverage |
|---------------|-----|------|----------|
| 3model-lstm | 2.97 ± 0.06 | 4.25 | 10.9% |
| 3model-gru | 3.83 ± 1.01 | 5.86 | 11.6% |
| gru-only | 4.32 ± 0.96 | 5.94 | 15.2% |
| lstm-only | 4.74 ± 1.04 | 6.23 | 14.5% |

## 3-Model Comparison

**LSTM outperforms GRU** by 0.85 frames.

Recommendation: Keep using LSTM.

## Single Model Comparison

- LSTM-only: MAE 4.74 at 14.5% coverage
- GRU-only: MAE 4.32 at 15.2% coverage
- Difference: -0.42 frames

## GRU Characteristics

- Fewer parameters: ~25% fewer than LSTM
- Faster training: Fewer computations per step
- Simpler architecture: No separate cell state
- Similar expressiveness for many tasks

## Comparison with Baseline

**Baseline (3-model LSTM)**: MAE 2.97 ± 0.06 at 10.9% coverage

GRU 3-model is 0.86 frames worse than baseline (likely variance).
