#!/usr/bin/env python3
"""
Summarize all saved runs and generate a results table.

Usage:
    python scripts/summarize_runs.py [runs_dir]
    python scripts/summarize_runs.py --markdown  # Output as markdown table
    python scripts/summarize_runs.py --csv       # Output as CSV
    python scripts/summarize_runs.py --update-docs  # Update docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class RunSummary:
    """Summary of a single run."""
    path: Path
    timestamp: str
    git_commit: str
    mae_mean: float
    mae_std: float
    within_5_mean: float
    within_5_std: float
    coverage_mean: float
    coverage_std: float
    agreement_threshold: int
    quality_threshold: float
    n_seeds: int
    quick_mode: bool

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> 'RunSummary | None':
        """Load a run summary from a directory."""
        try:
            # Load metrics
            metrics_path = run_dir / 'metrics.json'
            if not metrics_path.exists():
                return None
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Load config
            config_path = run_dir / 'config.json'
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

            # Load environment
            env_path = run_dir / 'environment.json'
            env = {}
            if env_path.exists():
                with open(env_path) as f:
                    env = json.load(f)

            return cls(
                path=run_dir,
                timestamp=env.get('timestamp', 'unknown'),
                git_commit=env.get('git_commit', 'unknown')[:8],
                mae_mean=metrics.get('mae_mean', float('nan')),
                mae_std=metrics.get('mae_std', 0.0),
                within_5_mean=metrics.get('within_5_mean', float('nan')),
                within_5_std=metrics.get('within_5_std', 0.0),
                coverage_mean=metrics.get('acceptance_rate_mean', 0.0),
                coverage_std=metrics.get('acceptance_rate_std', 0.0),
                agreement_threshold=config.get('agreement_threshold', -1),
                quality_threshold=config.get('quality_threshold', -1.0),
                n_seeds=config.get('n_seeds', 0),
                quick_mode=env.get('quick_mode', False),
            )
        except Exception as e:
            print(f"Warning: Failed to load {run_dir}: {e}", file=sys.stderr)
            return None


def load_all_runs(runs_dir: Path) -> list[RunSummary]:
    """Load all runs from a directory."""
    runs = []
    if not runs_dir.exists():
        return runs

    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir():
            run = RunSummary.from_run_dir(run_path)
            if run is not None:
                runs.append(run)

    return runs


def format_as_table(runs: list[RunSummary], include_quick: bool = False) -> str:
    """Format runs as a plain text table."""
    # Filter out quick mode unless requested
    if not include_quick:
        runs = [r for r in runs if not r.quick_mode]

    if not runs:
        return "No runs found."

    # Sort by MAE (best first)
    runs = sorted(runs, key=lambda r: r.mae_mean if not r.quick_mode else float('inf'))

    lines = [
        "=" * 100,
        f"{'Run':<30} {'MAE':<15} {'Within 5':<15} {'Coverage':<15} {'Commit':<10} {'Seeds'}",
        "-" * 100,
    ]

    for run in runs:
        name = run.path.name[:28]
        mae = f"{run.mae_mean:.2f} ± {run.mae_std:.2f}"
        within5 = f"{run.within_5_mean:.1f}% ± {run.within_5_std:.1f}%"
        coverage = f"{run.coverage_mean:.1f}% ± {run.coverage_std:.1f}%"
        quick_marker = " (QUICK)" if run.quick_mode else ""
        lines.append(
            f"{name:<30} {mae:<15} {within5:<15} {coverage:<15} {run.git_commit:<10} {run.n_seeds}{quick_marker}"
        )

    lines.append("=" * 100)

    # Show best result
    non_quick = [r for r in runs if not r.quick_mode]
    if non_quick:
        best = non_quick[0]
        lines.append("")
        lines.append(f"Best: {best.path.name}")
        lines.append(f"  MAE: {best.mae_mean:.2f} ± {best.mae_std:.2f} frames")
        lines.append(f"  Coverage: {best.coverage_mean:.1f}% ± {best.coverage_std:.1f}%")
        lines.append(f"  Commit: {best.git_commit}")

    return "\n".join(lines)


def format_as_markdown(runs: list[RunSummary], include_quick: bool = False) -> str:
    """Format runs as a markdown table."""
    if not include_quick:
        runs = [r for r in runs if not r.quick_mode]

    if not runs:
        return "No runs found."

    # Sort by MAE (best first)
    runs = sorted(runs, key=lambda r: r.mae_mean)

    lines = [
        "| Run | MAE | Within 5 | Coverage | Commit | Seeds |",
        "|-----|-----|----------|----------|--------|-------|",
    ]

    for run in runs:
        name = run.path.name[:25]
        mae = f"{run.mae_mean:.2f} ± {run.mae_std:.2f}"
        within5 = f"{run.within_5_mean:.1f}% ± {run.within_5_std:.1f}%"
        coverage = f"{run.coverage_mean:.1f}% ± {run.coverage_std:.1f}%"
        lines.append(
            f"| {name} | {mae} | {within5} | {coverage} | {run.git_commit} | {run.n_seeds} |"
        )

    return "\n".join(lines)


def format_as_csv(runs: list[RunSummary]) -> str:
    """Format runs as CSV."""
    import io
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        'run_name', 'timestamp', 'git_commit',
        'mae_mean', 'mae_std',
        'within_5_mean', 'within_5_std',
        'coverage_mean', 'coverage_std',
        'agreement_threshold', 'quality_threshold',
        'n_seeds', 'quick_mode'
    ])

    for run in runs:
        writer.writerow([
            run.path.name, run.timestamp, run.git_commit,
            run.mae_mean, run.mae_std,
            run.within_5_mean, run.within_5_std,
            run.coverage_mean, run.coverage_std,
            run.agreement_threshold, run.quality_threshold,
            run.n_seeds, run.quick_mode
        ])

    return output.getvalue()


def update_results_doc(runs: list[RunSummary], doc_path: Path) -> None:
    """Update docs/RESULTS.md with the latest results."""
    non_quick = [r for r in runs if not r.quick_mode]
    if not non_quick:
        print("No non-quick runs to report.")
        return

    # Sort by MAE
    non_quick = sorted(non_quick, key=lambda r: r.mae_mean)
    best = non_quick[0]

    content = f"""# Results

This file is auto-generated by `scripts/summarize_runs.py`.

## Best Result

**MAE: {best.mae_mean:.2f} ± {best.mae_std:.2f} frames** at {best.coverage_mean:.1f}% acceptance rate

- Run: `{best.path.name}`
- Commit: `{best.git_commit}`
- Seeds: {best.n_seeds}
- Agreement threshold: {best.agreement_threshold}
- Quality threshold: {best.quality_threshold}

## All Runs

{format_as_markdown(non_quick)}

---

*Last updated from run artifacts. See individual run directories for full details.*
"""

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, 'w') as f:
        f.write(content)

    print(f"Updated {doc_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize saved runs")
    parser.add_argument('runs_dir', type=Path, nargs='?', default=Path('runs'),
                        help='Directory containing runs')
    parser.add_argument('--markdown', action='store_true', help='Output as markdown')
    parser.add_argument('--csv', action='store_true', help='Output as CSV')
    parser.add_argument('--include-quick', action='store_true',
                        help='Include quick-mode runs in output')
    parser.add_argument('--update-docs', action='store_true',
                        help='Update docs/RESULTS.md')
    args = parser.parse_args()

    runs = load_all_runs(args.runs_dir)

    if not runs:
        print(f"No runs found in {args.runs_dir}")
        return 1

    print(f"Found {len(runs)} runs", file=sys.stderr)

    if args.update_docs:
        update_results_doc(runs, Path('docs/RESULTS.md'))
    elif args.csv:
        print(format_as_csv(runs))
    elif args.markdown:
        print(format_as_markdown(runs, args.include_quick))
    else:
        print(format_as_table(runs, args.include_quick))

    return 0


if __name__ == '__main__':
    sys.exit(main())
