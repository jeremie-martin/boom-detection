# Acceptance Formula Characterization Analysis

Generated: $(date)

## Dataset
- 90 simulations
- 3-seed evaluation (seeds: 42, 43, 44)
- 5-fold cross-validation

## Key Findings

### 1. MAE vs Coverage Tradeoff

The relationship between selectivity and accuracy is **monotonic but non-linear**:

| Coverage | Best MAE | Configuration |
|----------|----------|---------------|
| 3-5% | 2.5 ± 1.1 | sigmoid/s=3/t=0.70 |
| 5-10% | 2.6 ± 0.7 | sigmoid/s=4/t=0.70 |
| 10-15% | 2.8 ± 0.9 | sqrt/s=18/t=0.70 |
| 15-20% | 3.0 ± 0.8 | sigmoid/s=8/t=0.70 |
| 20-25% | 4.3 ± 1.3 | sqrt/s=40/t=0.70 |
| 25-30% | 4.6 ± 1.4 | linear/s=15/t=0.70 |
| 30-35% | 4.6 ± 1.3 | linear/s=18/t=0.70 |
| 35-40% | 4.9 ± 1.3 | sqrt/s=18/t=0.60 |

**Insight**: Diminishing returns at extreme selectivity (<5% coverage) due to small sample sizes.

### 2. Formula Comparison

All formulas show similar performance when properly tuned:

| Formula | Best MAE | Coverage | Config |
|---------|----------|----------|--------|
| sqrt | 2.77 ± 0.91 | 11.9% | s=18, t=0.70 |
| linear | 2.81 ± 1.28 | 4.4% | s=3, t=0.70 |
| sigmoid | 2.53 ± 1.10 | 4.1% | s=3, t=0.70 |
| quadratic | 2.96 ± 0.75 | 9.6% | s=2, t=0.70 |

**Insight**: sqrt formula provides best stability across coverage range.

### 3. Parameter Effects

**Scale parameter**:
- Lower scale = more selective (requires tighter model agreement)
- sqrt with scale=5 vs scale=15: MAE 3.4→4.9 but coverage 14%→30%

**Threshold parameter**:
- Higher threshold = more selective
- Threshold 0.70 gives best MAE but lowest coverage
- Threshold 0.60 is a good balance

### 4. The 3-Model Puzzle

Adding LSTM (the best individual model) **does NOT improve** the pipeline:

| Finding | Value |
|---------|-------|
| 2-model mean disagreement | 11.3 frames |
| 3-model mean disagreement | 17.1 frames (1.52x higher) |
| Cases where LSTM increases disagreement | 65.6% |
| When LSTM causes disagreement, it's correct | 45.2% (no better than chance) |

**Root cause**: Range-based disagreement (max-min) naturally increases with more models.
The additional model inflates disagreement without providing better discrimination.

**Interesting**: 3-model disagreement has BETTER correlation with error (r=0.558 vs r=0.439),
suggesting the signal is useful but needs different formula (std instead of range?).

### 5. Recommended Operating Points

| Use Case | Config | MAE | Coverage |
|----------|--------|-----|----------|
| Maximum accuracy | sqrt/s=15/t=0.70 | ~2.8 | ~12% |
| Conservative selective | sqrt/s=5/t=0.60 | ~3.4 | ~14% |
| Balanced | sqrt/s=15/t=0.60 | ~4.9 | ~30% |
| High coverage | sqrt/s=40/t=0.60 | ~7.9 | ~50% |

## Files

- `sweep_results.json` - Raw results (480 configurations)
- `sweep_results.csv` - Same data in CSV format
