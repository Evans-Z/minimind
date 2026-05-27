#!/usr/bin/env python3
"""Plot TensorBoard loss curves against consumed tokens.

This script is intended for pretraining runs that log either:
  - ``train_by_tokens/loss`` with token count stored as the TensorBoard step, or
  - ``train/loss`` plus ``train/tokens_seen`` with both stored by global step.

Examples:
    python scripts/plot_tb_loss_vs_tokens.py \
      --tb-root ../runs/pretrain_scale \
      --run "baseline/*" \
      --run "use_mhc/*" \
      --label "Baseline" \
      --label '$m$HC' \
      --token-center 10B \
      --token-window 1B \
      --smooth-method ema \
      --ema-alpha 0.03 \
      --zoom \
      --zoom-fraction 0.25 \
      --zoom-inset-bounds 0.50,0.48,0.45,0.38 \
      --output model/loss_vs_tokens.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    STORE_EVERYTHING_SIZE_GUIDANCE,
    EventAccumulator,
)


@dataclass
class ScalarSeries:
    tokens: np.ndarray
    values: np.ndarray


@dataclass
class RunCurve:
    label: str
    series: ScalarSeries
    source_dirs: list[str]


_NUMBER_WITH_SUFFIX_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([kKmMbBtT]?)\s*$"
)


def _parse_token_count(value: str | None) -> float | None:
    """Parse token counts such as 10000000000, 10B, 9.5b, or 800M."""
    if value is None:
        return None
    match = _NUMBER_WITH_SUFFIX_RE.match(value)
    if not match:
        raise ValueError(f"Invalid token count '{value}'. Try values like 10B or 800M.")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    scale = {
        "": 1.0,
        "k": 1e3,
        "m": 1e6,
        "b": 1e9,
        "t": 1e12,
    }[suffix]
    return number * scale


def _format_tokens(value: float) -> str:
    abs_value = abs(value)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs_value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _read_scalar_from_dir(run_dir: str, scalar_tag: str) -> ScalarSeries:
    accumulator = EventAccumulator(
        run_dir, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE
    )
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    if scalar_tag not in available:
        raise ValueError(
            f"Tag '{scalar_tag}' not found in run '{run_dir}'. Available tags: {available}"
        )
    scalars = accumulator.Scalars(scalar_tag)
    tokens = np.array([item.step for item in scalars], dtype=np.float64)
    values = np.array([item.value for item in scalars], dtype=np.float64)
    return ScalarSeries(tokens=tokens, values=values)


def _read_loss_with_token_tag(
    run_dir: str, loss_tag: str, token_tag: str
) -> ScalarSeries:
    """Read loss by global step and map it to the nearest logged token count."""
    accumulator = EventAccumulator(
        run_dir, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE
    )
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    missing = [tag for tag in (loss_tag, token_tag) if tag not in available]
    if missing:
        raise ValueError(
            f"Tag(s) {missing} not found in run '{run_dir}'. Available tags: {available}"
        )

    losses = accumulator.Scalars(loss_tag)
    token_events = accumulator.Scalars(token_tag)
    loss_steps = np.array([item.step for item in losses], dtype=np.float64)
    values = np.array([item.value for item in losses], dtype=np.float64)
    token_steps = np.array([item.step for item in token_events], dtype=np.float64)
    token_values = np.array([item.value for item in token_events], dtype=np.float64)
    if len(loss_steps) == 0 or len(token_steps) == 0:
        return ScalarSeries(
            tokens=np.array([], dtype=np.float64),
            values=np.array([], dtype=np.float64),
        )

    tokens = np.interp(loss_steps, token_steps, token_values)
    return ScalarSeries(tokens=tokens, values=values)


def _resolve_run_dirs(tb_root: str, run_spec: str) -> list[str]:
    """Resolve one logical run spec to one or more TensorBoard event directories."""
    expanded = os.path.expanduser(run_spec)
    candidates: list[str] = []

    if os.path.isdir(expanded):
        candidates.append(os.path.abspath(expanded))
    else:
        candidates.extend(
            os.path.abspath(path) for path in glob.glob(expanded) if os.path.isdir(path)
        )
        joined = os.path.join(tb_root, expanded)
        if os.path.isdir(joined):
            candidates.append(os.path.abspath(joined))
        candidates.extend(
            os.path.abspath(path)
            for path in glob.glob(os.path.join(tb_root, "**", expanded), recursive=True)
            if os.path.isdir(path)
        )

    event_dirs: list[str] = []
    for candidate in sorted(set(candidates)):
        path = Path(candidate)
        if any(path.glob("events.out.tfevents.*")):
            event_dirs.append(str(path.resolve()))
        else:
            event_dirs.extend(
                str(event_path.resolve())
                for event_path in path.glob("**/events.out.tfevents.*")
            )

    event_dirs = sorted(
        set(str(Path(path).parent if Path(path).is_file() else path) for path in event_dirs)
    )
    if not event_dirs:
        raise FileNotFoundError(
            f"No TensorBoard event directories match '{run_spec}' under '{tb_root}'."
        )
    return event_dirs


def _list_runs(tb_root: str) -> list[str]:
    root = Path(tb_root).expanduser().resolve()
    return sorted(
        set(str(event_path.parent) for event_path in root.glob("**/events.out.tfevents.*"))
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) == 0:
        return values
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"ema alpha must be in (0, 1], got {alpha}")
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def _smooth_values(
    values: np.ndarray, method: str, window: int, ema_alpha: float
) -> np.ndarray:
    if method == "none":
        return values
    if method == "moving_average":
        return _moving_average(values, window)
    if method == "ema":
        return _ema(values, ema_alpha)
    raise ValueError(f"Unknown smooth method: {method}")


def _merge_series(series_list: list[ScalarSeries], dedupe: str) -> ScalarSeries:
    tokens = np.concatenate([series.tokens for series in series_list])
    values = np.concatenate([series.values for series in series_list])
    finite_mask = np.isfinite(tokens) & np.isfinite(values)
    tokens = tokens[finite_mask]
    values = values[finite_mask]
    if len(tokens) == 0:
        return ScalarSeries(tokens=tokens, values=values)

    order = np.argsort(tokens, kind="mergesort")
    tokens = tokens[order]
    values = values[order]

    if dedupe == "none":
        return ScalarSeries(tokens=tokens, values=values)

    unique_tokens, inverse = np.unique(tokens, return_inverse=True)
    if dedupe == "last":
        unique_values = np.empty_like(unique_tokens, dtype=np.float64)
        for idx in range(len(unique_tokens)):
            unique_values[idx] = values[np.flatnonzero(inverse == idx)[-1]]
    elif dedupe == "mean":
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)
        unique_values = sums / counts
    else:
        raise ValueError(f"Unknown dedupe mode: {dedupe}")
    return ScalarSeries(tokens=unique_tokens, values=unique_values)


def _filter_by_tokens(
    series: ScalarSeries, token_min: float | None, token_max: float | None
) -> ScalarSeries:
    keep = np.ones(len(series.tokens), dtype=bool)
    if token_min is not None:
        keep &= series.tokens >= token_min
    if token_max is not None:
        keep &= series.tokens <= token_max
    return ScalarSeries(tokens=series.tokens[keep], values=series.values[keep])


def _load_curve(
    tb_root: str,
    run_spec: str,
    label: str,
    scalar_tag: str,
    token_tag: str,
    x_mode: str,
    dedupe: str,
    token_min: float | None,
    token_max: float | None,
    smooth_method: str,
    smooth_window: int,
    ema_alpha: float,
) -> RunCurve:
    run_dirs = _resolve_run_dirs(tb_root, run_spec)
    raw_series = []
    for run_dir in run_dirs:
        if x_mode == "scalar_step":
            raw_series.append(_read_scalar_from_dir(run_dir, scalar_tag))
        elif x_mode == "token_tag":
            raw_series.append(_read_loss_with_token_tag(run_dir, scalar_tag, token_tag))
        else:
            raise ValueError(f"Unknown x mode: {x_mode}")

    merged = _merge_series(raw_series, dedupe=dedupe)
    selected = _filter_by_tokens(merged, token_min=token_min, token_max=token_max)
    if len(selected.tokens) == 0:
        raise ValueError(
            f"Run '{run_spec}' is empty after token filtering. "
            f"Matched {len(run_dirs)} event dir(s)."
        )
    smoothed_values = _smooth_values(
        selected.values, smooth_method, smooth_window, ema_alpha
    )
    return RunCurve(
        label=label,
        series=ScalarSeries(tokens=selected.tokens, values=smoothed_values),
        source_dirs=run_dirs,
    )


def _resolve_token_bounds(args: argparse.Namespace) -> tuple[float | None, float | None]:
    token_min = _parse_token_count(args.token_min)
    token_max = _parse_token_count(args.token_max)
    token_center = _parse_token_count(args.token_center)
    token_window = _parse_token_count(args.token_window)

    if token_center is not None:
        if token_window is None:
            raise ValueError("--token-window is required when --token-center is set.")
        half_window = token_window / 2.0
        center_min = token_center - half_window
        center_max = token_center + half_window
        token_min = center_min if token_min is None else max(token_min, center_min)
        token_max = center_max if token_max is None else min(token_max, center_max)

    if token_min is not None and token_max is not None and token_min >= token_max:
        raise ValueError("--token-min must be smaller than --token-max.")
    return token_min, token_max


def _apply_axis_style(ax, xlabel: str, ylabel: str, token_scale: float):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.25)
    formatter = plt.FuncFormatter(lambda value, _: f"{value / token_scale:g}")
    ax.xaxis.set_major_formatter(formatter)


def _parse_inset_bounds(value: str) -> tuple[float, float, float, float]:
    """Parse matplotlib inset bounds: left,bottom,width,height in axes fraction."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "--zoom-inset-bounds must have four comma-separated values: "
            "left,bottom,width,height"
        )
    bounds = tuple(float(part) for part in parts)
    left, bottom, width, height = bounds
    if width <= 0.0 or height <= 0.0:
        raise ValueError("--zoom-inset-bounds width and height must be positive.")
    if left < 0.0 or bottom < 0.0 or left + width > 1.0 or bottom + height > 1.0:
        raise ValueError("--zoom-inset-bounds must fit inside [0, 1] axes space.")
    return bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard loss curves against token count."
    )
    parser.add_argument(
        "--tb-root",
        type=str,
        default="../runs/pretrain_scale",
        help="Root directory that contains TensorBoard run folders.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Only list discovered TensorBoard event directories and exit.",
    )
    parser.add_argument(
        "--run",
        type=str,
        action="append",
        default=[],
        help=(
            "Run dir/path or glob pattern under --tb-root. Can match multiple event "
            "directories, which will be merged as one resumed run. Repeat for curves."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=[],
        help="Legend label for each --run. Must match --run count when set.",
    )
    parser.add_argument(
        "--scalar-tag",
        type=str,
        default="train_by_tokens/loss",
        help="TensorBoard loss scalar tag to plot.",
    )
    parser.add_argument(
        "--token-tag",
        type=str,
        default="train/tokens_seen",
        help="Token scalar tag, used only with --x-mode token_tag.",
    )
    parser.add_argument(
        "--x-mode",
        type=str,
        default="scalar_step",
        choices=["scalar_step", "token_tag"],
        help=(
            "Use scalar steps as tokens, or align --scalar-tag to --token-tag by "
            "global step. For train_by_tokens/loss, use scalar_step."
        ),
    )
    parser.add_argument(
        "--token-min",
        type=str,
        default=None,
        help="Minimum token count to show, e.g. 9.5B.",
    )
    parser.add_argument(
        "--token-max",
        type=str,
        default=None,
        help="Maximum token count to show, e.g. 10.5B.",
    )
    parser.add_argument(
        "--token-center",
        type=str,
        default=None,
        help="Center of a token window to show, e.g. 10B.",
    )
    parser.add_argument(
        "--token-window",
        type=str,
        default=None,
        help="Width around --token-center, e.g. 1B means center +/- 0.5B.",
    )
    parser.add_argument(
        "--smooth-method",
        type=str,
        default="none",
        choices=["none", "moving_average", "ema"],
        help="Smoothing method for noisy loss curves.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving-average window size, used when --smooth-method=moving_average.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA alpha in (0,1]; smaller means smoother trend.",
    )
    parser.add_argument(
        "--dedupe",
        type=str,
        default="last",
        choices=["last", "mean", "none"],
        help="How to handle duplicate token counts after merging resumed logs.",
    )
    parser.add_argument("--title", type=str, default="", help="Figure title.")
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Tokens (billions)",
        help="X-axis label.",
    )
    parser.add_argument("--ylabel", type=str, default="Loss", help="Y-axis label.")
    parser.add_argument(
        "--token-scale",
        type=str,
        default="1B",
        help="Scale used for x tick labels; default shows tokens in billions.",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Add an inset window zooming into the tail of the selected token range.",
    )
    parser.add_argument(
        "--zoom-fraction",
        type=float,
        default=0.25,
        help="Fraction of selected x-range shown in the zoom inset.",
    )
    parser.add_argument(
        "--zoom-token-min",
        type=str,
        default=None,
        help="Explicit zoom min token. Overrides --zoom-fraction lower bound.",
    )
    parser.add_argument(
        "--zoom-token-max",
        type=str,
        default=None,
        help="Explicit zoom max token. Defaults to selected range max.",
    )
    parser.add_argument(
        "--zoom-inset-bounds",
        type=str,
        default="0.50,0.48,0.45,0.38",
        help=(
            "Inset position as left,bottom,width,height in main axes fraction. "
            "Move this when the default overlaps curves."
        ),
    )
    parser.add_argument("--y-min", type=float, default=None, help="Main plot y-axis min.")
    parser.add_argument("--y-max", type=float, default=None, help="Main plot y-axis max.")
    parser.add_argument(
        "--zoom-y-min", type=float, default=None, help="Zoom plot y-axis min."
    )
    parser.add_argument(
        "--zoom-y-max", type=float, default=None, help="Zoom plot y-axis max."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Line alpha for plotted curves.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/loss_vs_tokens.png",
        help="Output PNG figure path. A PDF is saved beside it by default.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI.")
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Deprecated; kept for compatibility. PDF is always saved.",
    )
    parser.add_argument(
        "--pdf-output",
        type=str,
        default="",
        help="Optional explicit PDF path. Defaults to output path with .pdf suffix.",
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

    if not args.run:
        raise ValueError("At least one --run is required unless --list-runs is used.")
    if args.label and len(args.label) != len(args.run):
        raise ValueError("When provided, --label count must equal --run count.")

    token_min, token_max = _resolve_token_bounds(args)
    token_scale = _parse_token_count(args.token_scale)
    if token_scale is None or token_scale <= 0:
        raise ValueError("--token-scale must be positive.")

    labels = args.label if args.label else [Path(spec.rstrip("/")).name for spec in args.run]
    curves = [
        _load_curve(
            tb_root=tb_root,
            run_spec=run_spec,
            label=label,
            scalar_tag=args.scalar_tag,
            token_tag=args.token_tag,
            x_mode=args.x_mode,
            dedupe=args.dedupe,
            token_min=token_min,
            token_max=token_max,
            smooth_method=args.smooth_method,
            smooth_window=args.smooth_window,
            ema_alpha=args.ema_alpha,
        )
        for run_spec, label in zip(args.run, labels)
    ]

    all_tokens = np.concatenate([curve.series.tokens for curve in curves])
    selected_min = token_min if token_min is not None else float(np.min(all_tokens))
    selected_max = token_max if token_max is not None else float(np.max(all_tokens))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for curve in curves:
        ax.plot(
            curve.series.tokens,
            curve.series.values,
            linewidth=2.2,
            label=curve.label,
            alpha=args.alpha,
        )

    _apply_axis_style(ax, args.xlabel, args.ylabel, token_scale)
    if args.title.strip():
        ax.set_title(args.title)
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)
    ax.set_xlim(selected_min, selected_max)
    ax.legend(loc="best", frameon=True)

    if args.zoom:
        zoom_min = _parse_token_count(args.zoom_token_min)
        zoom_max = _parse_token_count(args.zoom_token_max)
        if zoom_max is None:
            zoom_max = selected_max
        if zoom_min is None:
            if args.zoom_fraction <= 0.0 or args.zoom_fraction > 1.0:
                raise ValueError("--zoom-fraction must be in (0, 1].")
            zoom_width = (selected_max - selected_min) * args.zoom_fraction
            zoom_min = zoom_max - zoom_width
        zoom_min = max(zoom_min, selected_min)
        zoom_max = min(zoom_max, selected_max)
        if zoom_min >= zoom_max:
            raise ValueError("Zoom token range is empty.")

        zoom_ax = ax.inset_axes(_parse_inset_bounds(args.zoom_inset_bounds))
        zoom_ax.set_facecolor("white")
        for curve in curves:
            keep = (curve.series.tokens >= zoom_min) & (curve.series.tokens <= zoom_max)
            if np.any(keep):
                zoom_ax.plot(
                    curve.series.tokens[keep],
                    curve.series.values[keep],
                    linewidth=2.2,
                    label=curve.label,
                    alpha=args.alpha,
                )

        _apply_axis_style(
            ax=zoom_ax,
            xlabel="",
            ylabel="",
            token_scale=token_scale,
        )
        zoom_ax.set_xlim(zoom_min, zoom_max)
        if args.zoom_y_min is not None or args.zoom_y_max is not None:
            zoom_ax.set_ylim(args.zoom_y_min, args.zoom_y_max)
        zoom_ax.set_title("Tail zoom", fontsize=10)
        zoom_ax.tick_params(axis="both", labelsize=8)
        for spine in zoom_ax.spines.values():
            spine.set_edgecolor("#404040")
            spine.set_linewidth(1.0)
        ax.axvspan(zoom_min, zoom_max, color="#808080", alpha=0.10, zorder=0)

    png_path = Path(args.output).expanduser().resolve()
    if png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    pdf_path = (
        Path(args.pdf_output).expanduser().resolve()
        if args.pdf_output.strip()
        else png_path.with_suffix(".pdf")
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved PNG to   : {png_path}")
    print(f"Saved PDF to   : {pdf_path}")
    print(
        f"Token range   : {_format_tokens(selected_min)} - {_format_tokens(selected_max)}"
    )
    for curve in curves:
        print(f"Run '{curve.label}' merged {len(curve.source_dirs)} event dir(s):")
        for run_dir in curve.source_dirs:
            print(f"  {run_dir}")


if __name__ == "__main__":
    main()
