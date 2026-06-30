import argparse
import os
import sys
from dataclasses import dataclass

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model_mhc import HyperConnection


@dataclass(frozen=True)
class ProjectionMetrics:
    name: str
    shape: tuple[int, ...]
    min_value: float
    max_value: float
    row_mae: float
    row_max_err: float
    col_mae: float
    col_max_err: float
    total_mean: float
    diag_mean: float
    offdiag_mean: float
    zero_fraction: float
    active_entries_mean: float
    identity_mae: float
    input_delta_mae: float
    grad_mean_abs: float | None


def _build_connection(projector: str, hc_mult: int, hidden_size: int, iters: int, eps: float) -> HyperConnection:
    return HyperConnection(
        hc_mult=hc_mult,
        hidden_size=hidden_size,
        hc_iters=iters,
        hc_eps=eps,
        rms_norm_eps=1e-6,
        hc_projector=projector,
    )


def _case_scale(shape: tuple[int, ...], hc_mult: int, *, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    view_shape = (1,) * len(shape) + (hc_mult, 1)
    row_scale = torch.linspace(0.2, 2.0, hc_mult, device=device, dtype=dtype).view(*view_shape)
    col_scale = torch.linspace(2.0, 0.2, hc_mult, device=device, dtype=dtype).view(*view_shape[:-2], 1, hc_mult)
    return row_scale, col_scale


def _make_cases(hc_mult: int, *, batch_shape: tuple[int, ...], device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    eye = torch.eye(hc_mult, device=device, dtype=dtype)
    rand = torch.rand(*batch_shape, hc_mult, hc_mult, device=device, dtype=dtype)

    row_scale, col_scale = _case_scale(batch_shape, hc_mult, device=device, dtype=dtype)

    return {
        "uniform_random": rand + 1e-3,
        "identity_biased": 0.85 * eye.expand(*batch_shape, hc_mult, hc_mult) + 0.15 * rand + 1e-3,
        "skewed_positive": rand * row_scale * col_scale + 1e-3,
    }


def _offdiag_mean(matrix: torch.Tensor) -> torch.Tensor:
    hc_mult = matrix.shape[-1]
    offdiag_mask = ~torch.eye(hc_mult, device=matrix.device, dtype=torch.bool)
    return matrix[..., offdiag_mask].mean()


def _zero_fraction(matrix: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    return (matrix.abs() <= threshold).to(torch.float32).mean()


def _active_entries_mean(matrix: torch.Tensor, threshold: float = 1e-8) -> torch.Tensor:
    return (matrix.abs() > threshold).to(torch.float32).sum(dim=(-2, -1)).mean()


def _project(projector: str, comb: torch.Tensor, *, iters: int, eps: float) -> torch.Tensor:
    connection = _build_connection(projector, comb.shape[-1], hidden_size=8, iters=iters, eps=eps)
    if projector == "sinkhorn":
        return connection._project_comb_sinkhorn(comb)
    if projector == "balm":
        return connection._project_comb_balm(comb)
    raise ValueError(f"Unsupported projector: {projector}")


def _gradient_mean_abs(projector: str, comb: torch.Tensor, *, iters: int, eps: float) -> float | None:
    work = comb.detach().clone().requires_grad_(True)
    projected = _project(projector, work, iters=iters, eps=eps)
    if not projected.requires_grad:
        return None

    projected.square().mean().backward()
    if work.grad is None:
        return None
    return work.grad.abs().mean().item()


def _metrics(projector: str, name: str, comb: torch.Tensor, *, iters: int, eps: float) -> ProjectionMetrics:
    with torch.no_grad():
        projected = _project(projector, comb.detach().clone(), iters=iters, eps=eps)
        hc_mult = projected.shape[-1]
        row_err = projected.sum(dim=-1) - 1.0
        col_err = projected.sum(dim=-2) - 1.0
        eye = torch.eye(hc_mult, device=projected.device, dtype=projected.dtype)

        metrics = {
            "shape": tuple(projected.shape),
            "min_value": projected.min().item(),
            "max_value": projected.max().item(),
            "row_mae": row_err.abs().mean().item(),
            "row_max_err": row_err.abs().max().item(),
            "col_mae": col_err.abs().mean().item(),
            "col_max_err": col_err.abs().max().item(),
            "total_mean": projected.sum(dim=(-2, -1)).mean().item(),
            "diag_mean": projected.diagonal(dim1=-2, dim2=-1).mean().item(),
            "offdiag_mean": _offdiag_mean(projected).item(),
            "zero_fraction": _zero_fraction(projected).item(),
            "active_entries_mean": _active_entries_mean(projected).item(),
            "identity_mae": (projected - eye).abs().mean().item(),
            "input_delta_mae": (projected - comb).abs().mean().item(),
        }

    return ProjectionMetrics(
        name=f"{name}/{projector}",
        grad_mean_abs=_gradient_mean_abs(projector, comb, iters=iters, eps=eps),
        **metrics,
    )


def collect_projection_metrics(
    *,
    hc_mult: int = 4,
    iters: int = 20,
    eps: float = 1e-6,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    batch_shape: tuple[int, ...] = (2, 3),
) -> list[ProjectionMetrics]:
    torch.manual_seed(seed)
    cases = _make_cases(hc_mult, batch_shape=batch_shape, device=device, dtype=dtype)

    metrics = []
    for case_name, comb in cases.items():
        for projector in ("sinkhorn", "balm"):
            metrics.append(_metrics(projector, case_name, comb, iters=iters, eps=eps))
    return metrics


def _format_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.3e}"


def print_projection_report(metrics: list[ProjectionMetrics]) -> None:
    headers = [
        "case/projector",
        "row_mae",
        "row_max",
        "col_mae",
        "col_max",
        "total",
        "diag",
        "offdiag",
        "zero%",
        "active",
        "id_mae",
        "delta",
        "grad",
    ]
    rows = [
        [
            m.name,
            _format_float(m.row_mae),
            _format_float(m.row_max_err),
            _format_float(m.col_mae),
            _format_float(m.col_max_err),
            f"{m.total_mean:.4f}",
            f"{m.diag_mean:.4f}",
            f"{m.offdiag_mean:.4f}",
            f"{100.0 * m.zero_fraction:.1f}",
            f"{m.active_entries_mean:.1f}",
            _format_float(m.identity_mae),
            _format_float(m.input_delta_mae),
            _format_float(m.grad_mean_abs),
        ]
        for m in metrics
    ]

    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def print_projection_matrices(
    *,
    hc_mult: int,
    iters: int,
    eps: float,
    seed: int,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    torch.manual_seed(seed)
    cases = _make_cases(hc_mult, batch_shape=(), device=device, dtype=dtype)

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    for case_name, comb in cases.items():
        sinkhorn = _project("sinkhorn", comb.detach().clone(), iters=iters, eps=eps)
        balm = _project("balm", comb.detach().clone(), iters=iters, eps=eps)

        print(f"\n[{case_name}] input comb")
        print(comb)
        print("row sums:", comb.sum(dim=-1))
        print("col sums:", comb.sum(dim=-2))

        print("\nsinkhorn projected")
        print(sinkhorn)
        print("row sums:", sinkhorn.sum(dim=-1))
        print("col sums:", sinkhorn.sum(dim=-2))
        print(f"zero fraction: {_zero_fraction(sinkhorn).item():.2%}")

        print("\nbalm projected")
        print(balm)
        print("row sums:", balm.sum(dim=-1))
        print("col sums:", balm.sum(dim=-2))
        print(f"zero fraction: {_zero_fraction(balm).item():.2%}")


def test_projection_methods_return_finite_nonnegative_outputs():
    metrics = collect_projection_metrics(hc_mult=4, iters=20)
    for metric in metrics:
        assert metric.shape == (2, 3, 4, 4)
        assert metric.min_value >= 0.0
        assert torch.isfinite(torch.tensor(metric.max_value))


def test_sinkhorn_projection_is_nearly_doubly_stochastic():
    metrics = collect_projection_metrics(hc_mult=4, iters=20)
    sinkhorn_metrics = [metric for metric in metrics if metric.name.endswith("/sinkhorn")]

    for metric in sinkhorn_metrics:
        assert metric.row_max_err < 5e-3, metric
        assert metric.col_max_err < 5e-3, metric
        assert metric.grad_mean_abs is not None and metric.grad_mean_abs > 0.0


def test_balm_projection_is_sparse_and_nearly_doubly_stochastic():
    metrics = collect_projection_metrics(hc_mult=4, iters=100)
    balm_metrics = [metric for metric in metrics if metric.name.endswith("/balm")]
    sinkhorn_by_case = {
        metric.name.removesuffix("/sinkhorn"): metric
        for metric in metrics
        if metric.name.endswith("/sinkhorn")
    }

    for metric in balm_metrics:
        assert metric.row_max_err < 2e-2, metric
        assert metric.col_max_err < 2e-2, metric
        assert metric.grad_mean_abs is not None and metric.grad_mean_abs > 0.0

        case_name = metric.name.removesuffix("/balm")
        assert metric.zero_fraction > sinkhorn_by_case[case_name].zero_fraction, metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mHC comb projection behavior.")
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metrics = collect_projection_metrics(
        hc_mult=args.hc_mult,
        iters=args.iters,
        eps=args.eps,
        seed=args.seed,
        device=args.device,
        batch_shape=(),
    )
    print_projection_matrices(
        hc_mult=args.hc_mult,
        iters=args.iters,
        eps=args.eps,
        seed=args.seed,
        device=args.device,
    )
    print("\nSummary metrics")
    print_projection_report(metrics)


if __name__ == "__main__":
    main()
