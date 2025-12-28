# Boom Detection

Predict the "boom" frame in chaotic double pendulum simulations - the moment when pendulums visually diverge into a caustic pattern.

## Best Result: MAE 6.4 ± 0.5 frames

Using model agreement + predicted quality filtering (robust 5-seed evaluation):

| Metric | Value |
|--------|-------|
| MAE | **6.4 ± 0.5 frames** |
| Within 5 frames | 63% ± 6% |
| Acceptance rate | 35% ± 5% |

**Key improvements from ablation/tuning:**
- CNN prediction (not HGB) - more accurate when models agree
- Random Forest for quality (not Ridge) - better correlation
- Top 50 quality features with smaller window (±25) - less overfitting
- Larger CNN kernels (5,11,21) - capture longer-range temporal patterns
- hidden_dim=64 - more capacity without overfitting

## Quick Start

```bash
# Install
uv sync --extra ml

# Evaluate the deployable pipeline
uv run python -m boom_detection.deploy_pipeline data --evaluate

# Train and save models
uv run python -m boom_detection.deploy_pipeline data --train --output models/
```

## How It Works

The pipeline uses two models (CNN and HistGBM) as a confidence filter:

1. **Run both models** on the simulation
2. **Check agreement**: If predictions differ by >5 frames → reject
3. **Predict quality**: Random Forest on top 50 correlated features (±25 window)
4. **Filter**: If predicted quality < 0.55 → reject
5. **Accept**: Use CNN prediction (ablation study showed CNN > HGB)

This achieves MAE 6.7 on ~34% of simulations. For video production, we simply generate more simulations and use only the accepted ones.

## Project Structure

```
src/boom_detection/
├── deploy_pipeline.py  # Production pipeline (start here!)
├── loader.py           # Load simulations and annotations
├── features.py         # Feature extraction + caching
├── evaluation.py       # Metrics and cross-validation
├── frame_models.py     # HistGBM classifier
├── sequence_models.py  # CNN, LSTM, Transformer (PyTorch)
├── quality_models.py   # Boom quality prediction
├── pipeline.py         # Multi-stage pipeline components
└── run_baselines.py    # Baseline comparison script
```

## Data

```bash
# Dataset not in git (~20GB). Copy from source:
cp -r /path/to/double-pendulum/output/eval2 data
```

- 49 valid simulations
- Boom frames: 204-933 (annotated)
- Quality scores: 0.1-0.92 (annotated)

## Documentation

- **[docs/RESULTS.md](docs/RESULTS.md)** - Detailed results and findings
- **[docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md)** - Full experiment history
- **[DATA_FORMAT.md](DATA_FORMAT.md)** - Binary data format specification
- **[CLAUDE.md](CLAUDE.md)** - AI assistant context

## Development

```bash
uv sync --extra ml                                    # Install
uv run pytest                                         # Run tests
uv run python -m boom_detection.run_baselines data    # Run baselines
uv run python -m boom_detection.deploy_pipeline data --evaluate  # Best pipeline
```

## Key Insights

1. **Model agreement = confidence**: When CNN and HistGBM agree, predictions are reliable
2. **Quality predicts error**: High-quality booms have MAE ~11, low-quality ~31
3. **Different features for different tasks**: Derivatives predict quality; variance/range predict boom
4. **CNN > HGB**: Ablation study showed CNN is more accurate when models agree
5. **Feature selection matters**: Top 50 features with smaller window reduces overfitting

See [docs/RESULTS.md](docs/RESULTS.md) for detailed analysis.
