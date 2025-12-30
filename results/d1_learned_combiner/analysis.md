# D1: Learned Combiner (Logistic/MLP)

## Objective
Replace hand-tuned ThresholdCombiner with learned decision boundary.

## Configuration
- Seeds: [42, 43, 44]
- Dataset: 175 simulations
- Error threshold for accept label: 5.0 frames
- Nested 5-fold cross-validation
- 3-model pipeline (CNN + HGB + LSTM)

## Features for Classifier
- CNN predicted frame
- HGB predicted frame
- LSTM predicted frame
- Predicted quality
- Std disagreement
- Range disagreement

## Results

| Classifier | MAE | Coverage |
|------------|-----|----------|
| LogisticRegression | 11.39 ± 1.19 | 55.8% |
| MLP-medium | 11.84 ± 0.97 | 51.0% |
| MLP-small | 12.34 ± 0.51 | 54.9% |

## Comparison with Baseline

**Baseline**: ThresholdCombiner (std/s=15/t=0.70) - MAE 2.97 @ 10.9% coverage

**Best learned combiner**: LogisticRegression
- MAE: 11.39 ± 1.19
- Coverage: 55.8%

**ThresholdCombiner outperforms** learned combiner by 8.42 frames.
Recommendation: Keep using ThresholdCombiner.

## Analysis

The learned combiner attempts to learn the accept/reject decision from data.
This allows it to potentially capture complex decision boundaries that
a simple threshold-based approach cannot.

However, the hand-tuned ThresholdCombiner performs better, likely because:
1. The decision boundary is actually simple (threshold-based)
2. Limited training data for the classifier
3. The features already capture the relevant information
