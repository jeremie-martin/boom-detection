# C5: Disagreement Metric MAD

## Objective
Test MAD (median absolute deviation) as an alternative to std and range.
MAD is more robust to outliers than std.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)

## Results

| Configuration | MAE | RMSE | Coverage |
|---------------|-----|------|----------|
| range/s=30/t=0.75 | 0.60 ± 0.53 | 0.63 | 1.3% |
| std/s=10/t=0.75 | 0.67 ± 0.58 | 0.67 | 1.0% |
| range/s=20/t=0.75 | 0.67 ± 0.58 | 0.67 | 1.0% |
| range/s=25/t=0.75 | 0.67 ± 0.58 | 0.67 | 1.0% |
| std/s=5/t=0.7 | 1.00 ± 1.00 | 1.18 | 1.3% |
| std/s=5/t=0.75 | 1.00 ± 0.00 | 1.00 | 0.4% |
| std/s=8/t=0.75 | 1.00 ± 0.00 | 1.00 | 0.6% |
| range/s=10/t=0.7 | 1.00 ± 1.00 | 1.22 | 1.1% |
| range/s=10/t=0.75 | 1.00 ± 0.00 | 1.00 | 0.4% |
| range/s=15/t=0.75 | 1.00 ± 0.00 | 1.00 | 0.4% |
| std/s=15/t=0.75 | 2.13 ± 0.73 | 2.71 | 2.7% |
| std/s=20/t=0.75 | 2.49 ± 1.07 | 3.13 | 5.0% |
| range/s=20/t=0.7 | 2.53 ± 0.85 | 3.40 | 5.5% |
| std/s=8/t=0.7 | 2.55 ± 0.68 | 3.28 | 4.4% |
| range/s=10/t=0.65 | 2.56 ± 0.79 | 3.43 | 5.1% |
| std/s=10/t=0.7 | 2.58 ± 0.69 | 3.36 | 6.9% |
| range/s=25/t=0.7 | 2.66 ± 0.43 | 3.49 | 8.2% |
| range/s=30/t=0.7 | 2.70 ± 0.35 | 3.59 | 9.1% |
| std/s=5/t=0.65 | 2.74 ± 0.81 | 3.52 | 6.3% |
| range/s=15/t=0.7 | 2.93 ± 1.29 | 3.66 | 3.6% |
| std/s=15/t=0.7 | 2.97 ± 0.06 | 4.25 | 10.9% |
| range/s=15/t=0.65 | 3.55 ± 1.18 | 6.45 | 9.1% |
| std/s=20/t=0.7 | 4.23 ± 0.57 | 8.64 | 14.9% |
| range/s=20/t=0.65 | 4.35 ± 0.79 | 8.95 | 13.7% |
| std/s=10/t=0.65 | 4.36 ± 0.71 | 8.76 | 15.2% |
| std/s=8/t=0.65 | 4.49 ± 0.56 | 9.12 | 12.8% |
| range/s=25/t=0.65 | 4.53 ± 0.89 | 8.58 | 17.1% |
| range/s=30/t=0.65 | 4.63 ± 0.76 | 8.30 | 20.2% |
| std/s=15/t=0.65 | 4.73 ± 0.98 | 8.17 | 22.5% |
| std/s=20/t=0.65 | 5.12 ± 0.65 | 8.43 | 27.6% |
| mad/s=10/t=0.75 | 5.44 ± 0.69 | 9.37 | 10.7% |
| mad/s=2/t=0.75 | 5.55 ± 0.83 | 9.50 | 10.5% |
| mad/s=3/t=0.75 | 5.55 ± 0.83 | 9.50 | 10.5% |
| mad/s=5/t=0.75 | 5.55 ± 0.83 | 9.50 | 10.5% |
| mad/s=8/t=0.75 | 5.55 ± 0.83 | 9.50 | 10.5% |
| mad/s=10/t=0.7 | 5.68 ± 1.24 | 9.76 | 18.5% |
| mad/s=5/t=0.65 | 6.00 ± 1.90 | 10.30 | 19.2% |
| mad/s=8/t=0.7 | 6.04 ± 1.16 | 10.16 | 16.2% |
| mad/s=10/t=0.65 | 6.12 ± 1.88 | 10.03 | 28.2% |
| mad/s=8/t=0.65 | 6.19 ± 2.16 | 10.25 | 25.3% |
| mad/s=5/t=0.7 | 6.51 ± 0.91 | 11.00 | 13.1% |
| mad/s=2/t=0.7 | 6.60 ± 0.88 | 11.10 | 13.0% |
| mad/s=3/t=0.7 | 6.60 ± 0.88 | 11.10 | 13.0% |
| mad/s=3/t=0.65 | 6.84 ± 1.82 | 11.52 | 14.1% |
| mad/s=2/t=0.65 | 6.91 ± 1.76 | 11.60 | 13.9% |

## Best per Metric

- **MAD**: mad/s=10/t=0.75 - MAE 5.44, Coverage 10.7%
- **STD**: std/s=10/t=0.75 - MAE 0.67, Coverage 1.0%
- **RANGE**: range/s=30/t=0.75 - MAE 0.60, Coverage 1.3%

## Analysis

### Comparison at threshold=0.70

| Metric | Best Scale | MAE | Coverage |
|--------|------------|-----|----------|
| mad | 10 | 5.68 | 18.5% |
| std | 5 | 1.00 | 1.3% |
| range | 10 | 1.00 | 1.1% |

**Best metric**: `std` with scale=5

MAD/std/range alternatives **improve** over baseline by 1.97 frames.

### Recommendations

- **std outperforms MAD** - keep using std metric
