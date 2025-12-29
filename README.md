# Boom Detection

Predict the "boom" frame in chaotic double pendulum simulations - the moment when pendulums visually diverge into a caustic pattern.

## Pipeline

```
                            ┌───────────┐
  Simulation                │    CNN    │──► pred: 546
  (2000 pendulums)          │ (PyTorch) │              ╲
        │                   └───────────┘               ╲    ┌─────────────────┐
        │     ┌──────────┐                               ══► │ Accept/Reject   │
        └────►│ Features │                               ══► │                 │
              │(183/frame)│                              ╱   │ score ≥ 0.60 ?  │
              └──────────┘  ┌───────────┐               ╱    └────────┬────────┘
                            │    HGB    │──► pred: 546                │
                            │ (sklearn) │                    ┌────────┴────────┐
                            └───────────┘                    ▼                 ▼
                                                          ACCEPT            REJECT
                                                       boom_frame=546    boom_frame=null
```

**Accept score** = `0.4 × model_agreement + 0.6 × predicted_quality`

## Results (90 simulations, 3-seed evaluation)

| Config | MAE | Coverage | Use Case |
|--------|-----|----------|----------|
| Selective (sqrt/5) | **3.4 ± 0.8** | 14% | Best accuracy, more generation needed |
| Balanced (sqrt/15) | 4.9 ± 1.3 | 30% | Good tradeoff |
| Default (linear/10) | 7.3 ± 1.7 | 39% | More coverage, less accuracy |

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
