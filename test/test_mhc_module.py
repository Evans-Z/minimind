import math
import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.model_mhc import HyperConnection


def _assert_close(name: str, actual: torch.Tensor, expected: float, atol: float = 1e-4, rtol: float = 1e-3):
    target = torch.full_like(actual, expected)
    if not torch.allclose(actual, target, atol=atol, rtol=rtol):
        max_abs_err = (actual - target).abs().max().item()
        raise AssertionError(f"{name} mismatch: expected~{expected}, max_abs_err={max_abs_err:.6e}")


def test_shapes_and_dtypes():
    hc, d = 4, 16
    m = HyperConnection(hc_mult=hc, hidden_size=d, hc_iters=8, hc_eps=1e-6, rms_norm_eps=1e-6)

    x = torch.randn(2, 5, hc, d, dtype=torch.float32, requires_grad=True)
    post, comb, collapsed = m(x)

    assert post.shape == (2, 5, hc), f"post shape wrong: {post.shape}"
    assert comb.shape == (2, 5, hc, hc), f"comb shape wrong: {comb.shape}"
    assert collapsed.shape == (2, 5, d), f"collapsed shape wrong: {collapsed.shape}"
    assert collapsed.dtype == x.dtype, f"collapsed dtype should match input dtype ({x.dtype}), got {collapsed.dtype}"


def test_initial_targets():
    hc, d = 4, 8
    eps = 1e-6
    m = HyperConnection(hc_mult=hc, hidden_size=d, hc_iters=10, hc_eps=eps, rms_norm_eps=1e-6)

    # fn should start at zero as intended.
    assert torch.count_nonzero(m.fn).item() == 0, "fn is not initialized to zero"

    # With fn == 0, mix == 0; so pre/post are controlled only by base.
    pre0 = torch.sigmoid(m.base[:hc]) + eps
    post0 = torch.sigmoid(m.base[hc : 2 * hc]) + eps

    _assert_close("pre0", pre0, 1.0 / hc, atol=2e-5, rtol=1e-3)
    _assert_close("post0", post0, 0.95, atol=2e-4, rtol=2e-3)

    # comb should be identity-biased after Sinkhorn projection.
    x = torch.randn(1, 2, hc, d)
    _, comb, _ = m(x)
    diag = torch.diagonal(comb, dim1=-2, dim2=-1).mean()
    off = (comb.sum(dim=(-2, -1)) - torch.diagonal(comb, dim1=-2, dim2=-1).sum(dim=-1)).mean() / (hc * hc - hc)
    assert diag > off, f"comb is not identity-biased: diag={diag.item():.6f}, off={off.item():.6f}"


def test_sinkhorn_near_doubly_stochastic():
    hc, d = 4, 8
    m = HyperConnection(hc_mult=hc, hidden_size=d, hc_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6)

    x = torch.randn(3, 7, hc, d)
    _, comb, _ = m(x)
    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)

    _assert_close("comb row sums", row_sums, 1.0, atol=5e-3, rtol=5e-3)
    _assert_close("comb col sums", col_sums, 1.0, atol=5e-3, rtol=5e-3)


def test_backward_flow():
    hc, d = 4, 8
    m = HyperConnection(hc_mult=hc, hidden_size=d, hc_iters=8, hc_eps=1e-6, rms_norm_eps=1e-6)

    x = torch.randn(2, 4, hc, d, requires_grad=True)
    post, comb, collapsed = m(x)
    loss = post.mean() + comb.mean() + collapsed.mean()
    loss.backward()

    for name, p in [("fn", m.fn), ("base", m.base), ("scale", m.scale)]:
        assert p.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(p.grad).all(), f"{name}.grad has NaN/Inf"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "input grad invalid"


def main():
    torch.manual_seed(42)
    tests = [
        test_shapes_and_dtypes,
        test_initial_targets,
        test_sinkhorn_near_doubly_stochastic,
        test_backward_flow,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print("All mHC module tests passed.")


if __name__ == "__main__":
    main()
