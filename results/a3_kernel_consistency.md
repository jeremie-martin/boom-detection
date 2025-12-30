# A3: Kernel Size Consistency Verification

## Objective
Check whether the kernel size inconsistency between `deploy_pipeline.py` and
`sequence_models.py` is intentional or a bug.

## Findings

### Current Configuration

| Location | Kernel Sizes | Notes |
|----------|--------------|-------|
| `sequence_models.py` (default) | (3, 7, 15) | Default parameter in CNNClassifier.__init__ |
| `deploy_pipeline.py` | (5, 11, 21) | Explicitly set when creating CNN |
| `scripts/evaluate_lstm.py` | (5, 11, 21) | Explicitly set |
| `tests/test_smoke.py` | (5, 11, 21) | Explicitly set |

### Analysis

**This is INTENTIONAL, not a bug.**

The `deploy_pipeline.py` and production scripts explicitly override the default
kernel sizes with `(5, 11, 21)`:

```python
# deploy_pipeline.py line 255-258
model = CNNClassifier(
    n_features=self.n_features,
    hidden_dim=64,
    kernel_sizes=(5, 11, 21)
)
```

The larger kernel sizes `(5, 11, 21)` provide wider receptive fields which is
likely better for capturing the boom transition pattern across multiple time scales.
The smaller default `(3, 7, 15)` exists for basic/lightweight use cases but is not
used in production.

### Code Quality Note

The current situation is fine but could be improved:
1. The explicit override makes the production configuration clear
2. Tests use the same values as production
3. The default is smaller for flexibility

### Recommendation

**No changes needed.** The inconsistency is intentional:
- Default kernel sizes `(3, 7, 15)` are smaller/simpler for general use
- Production explicitly uses `(5, 11, 21)` for better boom detection
- The explicit override is more explicit than changing the default

If desired, could add a comment in `deploy_pipeline.py` explaining why production
uses different kernel sizes than the default, but this is optional.

## Conclusion

**Not a bug.** The kernel sizes are intentionally different:
- Library default: small kernels for general use
- Production config: larger kernels explicitly set for boom detection
