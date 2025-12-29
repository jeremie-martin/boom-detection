# Archived Scripts

These scripts were used during development and research but are no longer actively maintained.
They are preserved for reference and historical context.

## Research Scripts (Conclusions Documented)

### `analyze_features.py`
Feature importance analysis for boom detection models.
**Conclusion**: Core features identified and documented in main codebase.

### `investigate_causticness.py`
Investigation of caustic patterns as features for boom detection.
**Conclusion**: Caustic features do NOT improve the pipeline - see CLAUDE.md.

### `explore_caustic_variants.py`
Exploration of different caustic feature formulations.
**Conclusion**: No variant improved over no_caustic baseline.

### `evaluate_caustic_formulas.py`
Systematic evaluation of caustic formula variations.
**Conclusion**: All caustic formulas performed worse than no_caustic.

### `evaluate_coverage_features.py`
Testing coverage-based features for quality prediction.
**Conclusion**: Coverage features did not improve quality prediction.

## Superseded Scripts

### `experiment_combined_best.py`
Early experiment combining different approaches.
**Superseded by**: `experiment_combiner_ablations.py` with more rigorous methodology.

### `experiment_quality_gated_model.py`
Testing quality-gated model selection.
**Results documented in**: `runs/combiner_ablations/FINDINGS.md`

## Utility Scripts

### `summarize_runs.py`
Utility to summarize experiment run directories.
**Note**: Rarely needed; results are now documented in CLAUDE.md.
