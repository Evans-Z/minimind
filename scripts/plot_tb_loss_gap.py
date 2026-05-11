#!/usr/bin/env python3
"""Plot TensorBoard scalar gaps against a baseline run.

Examples:
    python scripts/plot_tb_loss_gap.py \
      --tb-root ../runs/pretrain \
      --baseline "baseline/*" \
      --baseline-label "Baseline" \
      --compare "use_hc/*" \
      --compare "use_mhc/*" \
      --compare-label "HC" \
      --compare-label '$m$HC' \
      --scalar-tag train/loss \
      --y-min -0.06 \
      --save-pdf \
      --output model/loss_gap.png
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def _read_scalar_from_run(run_dir: str, scalar_tag: str) -> ScalarSeries:
    """Load scalar series from a single TensorBoard run directory."""
    accumulator = EventAccumulator(run_dir)
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if scalar_tag not in available:
        raise ValueError(
            f"Tag '{scalar_tag}' not found in run '{run_dir}'. Available tags: {available}"
        )
    scalars = accumulator.Scalars(scalar_tag)
    steps = np.array([item.step for item in scalars], dtype=np.int64)
    values = np.array([item.value for item in scalars], dtype=np.float64)
    return ScalarSeries(steps=steps, values=values)


def _resolve_run_dir(tb_root: str, run_spec: str) -> str:
    """Resolve a run directory from an explicit path or a glob pattern."""
    expanded = os.path.expanduser(run_spec)

    if os.path.isdir(expanded):
        return os.path.abspath(expanded)

    candidate = os.path.join(tb_root, expanded)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)

    # Support flexible selection under TensorBoard root, e.g. "baseline/*".
    matches = glob.glob(os.path.join(tb_root, "**", expanded), recursive=True)
    matches = [os.path.abspath(path) for path in matches if os.path.isdir(path)]
    if not matches:
        raise FileNotFoundError(
            f"No run directory matches '{run_spec}' under '{tb_root}'."
        )
    if len(matches) > 1:
        preview = ", ".join(matches[:5])
        raise ValueError(
            f"Run spec '{run_spec}' matched multiple directories; please be more specific.\n"
            f"Matched (showing up to 5): {preview}"
        )
    return matches[0]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with edge-safe fallback."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _interp_to_steps(source: ScalarSeries, target_steps: np.ndarray) -> np.ndarray:
    """Interpolate source values onto target steps."""
    return np.interp(
        target_steps.astype(np.float64),
        source.steps.astype(np.float64),
        source.values.astype(np.float64),
    )


def _collect_runs(tb_root: str, specs: list[str]) -> list[str]:
    return [_resolve_run_dir(tb_root, spec) for spec in specs]


def _list_runs(tb_root: str) -> list[str]:
    root = Path(tb_root).expanduser().resolve()
    runs = []
    for event_path in root.glob("**/events.out.tfevents.*"):
        runs.append(str(event_path.parent))
    return sorted(set(runs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scalar gap vs. training steps from TensorBoard runs."
    )
    parser.add_argument(
        "--tb-root",
        type=str,
        default="../runs/pretrain",
        help="Root directory that contains TensorBoard run folders.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Only list discovered run directories under --tb-root and exit.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=False,
        help="Baseline run dir/path or pattern under --tb-root.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline",
        help="Legend label for baseline reference line.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        action="append",
        default=[],
        help="Compare run dir/path or pattern under --tb-root. Can be repeated.",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        action="append",
        default=[],
        help="Legend labels for compare runs. Must match --compare count when set.",
    )
    parser.add_argument(
        "--scalar-tag",
        type=str,
        default="train/loss",
        help="TensorBoard scalar tag to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title. Leave empty to omit title.",
    )
    parser.add_argument("--xlabel", type=str, default="Steps", help="X-axis label.")
    parser.add_argument(
        "--ylabel", type=str, default="Absolute Loss Gap", help="Y-axis label."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/loss_gap_vs_steps.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save a PDF version (good for LaTeX).",
    )
    parser.add_argument(
        "--pdf-output",
        type=str,
        default="",
        help="Optional explicit PDF path. Defaults to output path with .pdf suffix.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving average window for smoothing (1 disables smoothing).",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=None,
        help="Optional max training step to display.",
    )
    parser.add_argument("--y-min", type=float, default=None, help="Optional y-axis min.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional y-axis max.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Line alpha for compare curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tb_root = str(Path(args.tb_root).expanduser().resolve())

    if args.list_runs:
        runs = _list_runs(tb_root)
        if not runs:
            print(f"No TensorBoard runs found under: {tb_root}")
            return
        print(f"Discovered {len(runs)} run(s) under: {tb_root}")
        for run_dir in runs:
            print(run_dir)
        return

    if not args.baseline:
        raise ValueError("--baseline is required unless --list-runs is used.")
    if not args.compare:
        raise ValueError("At least one --compare run is required.")
    if args.compare_label and len(args.compare_label) != len(args.compare):
        raise ValueError(
            "When provided, --compare-label count must equal --compare count."
        )

    baseline_run = _resolve_run_dir(tb_root, args.baseline)
    compare_runs = _collect_runs(tb_root, args.compare)
    compare_labels = (
        args.compare_label
        if args.compare_label
        else [Path(run).name for run in compare_runs]
    )

    baseline_series = _read_scalar_from_run(baseline_run, args.scalar_tag)
    if args.x_max is not None:
        keep_mask = baseline_series.steps <= args.x_max
        baseline_series = ScalarSeries(
            steps=baseline_series.steps[keep_mask],
            values=baseline_series.values[keep_mask],
        )
    if len(baseline_series.steps) == 0:
        raise ValueError("Baseline series is empty after filtering.")

    baseline_values = _moving_average(baseline_series.values, args.smooth_window)
    steps = baseline_series.steps

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline line sits at 0 gap.
    ax.axhline(0.0, color="#4C4C4C", linewidth=1.8, label=args.baseline_label, zorder=3)

    for run_dir, label in zip(compare_runs, compare_labels):
        compare_series = _read_scalar_from_run(run_dir, args.scalar_tag)
        if args.x_max is not None:
            keep_mask = compare_series.steps <= args.x_max
            compare_series = ScalarSeries(
                steps=compare_series.steps[keep_mask],
                values=compare_series.values[keep_mask],
            )
        compare_interp = _interp_to_steps(compare_series, steps)
        compare_interp = _moving_average(compare_interp, args.smooth_window)
        gap = compare_interp - baseline_values
        ax.plot(steps, gap, linewidth=2.2, label=label, alpha=args.alpha)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title.strip():
        ax.set_title(args.title)
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)
    if args.x_max is not None:
        ax.set_xlim(steps.min(), args.x_max)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.25)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    pdf_path = None
    if args.save_pdf:
        if args.pdf_output.strip():
            pdf_path = Path(args.pdf_output).expanduser().resolve()
        else:
            pdf_path = out_path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep vector text/curves for high-quality LaTeX inclusion.
        fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved figure to: {out_path}")
    if pdf_path is not None:
        print(f"Saved PDF to   : {pdf_path}")
    print(f"Baseline run : {baseline_run}")
    for run_dir, label in zip(compare_runs, compare_labels):
        print(f"Compare run  : {run_dir} (label={label})")


if __name__ == "__main__":
    main()
