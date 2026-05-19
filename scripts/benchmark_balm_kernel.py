#!/usr/bin/env python3
import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.balm_triton import is_triton_balm_available
from model.model_mhc import HyperConnection


def make_module(args, kernel):
    return HyperConnection(
        hc_mult=args.hc_mult,
        hidden_size=args.hidden_size,
        hc_iters=args.hc_iters,
        hc_eps=args.hc_eps,
        rms_norm_eps=args.rms_norm_eps,
        hc_projector="balm",
        hc_balm_r=args.hc_balm_r,
        hc_balm_delta=args.hc_balm_delta,
        hc_balm_diag_cost=args.hc_balm_diag_cost,
        hc_balm_offdiag_cost=args.hc_balm_offdiag_cost,
        hc_balm_cost_scale=args.hc_balm_cost_scale,
        hc_balm_kernel=kernel,
    ).to(args.device)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_projector(module, comb, warmup, repeat, device):
    for _ in range(warmup):
        module._project_comb_balm(comb)
    sync(device)
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            module._project_comb_balm(comb)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat

    start = time.perf_counter()
    for _ in range(repeat):
        module._project_comb_balm(comb)
    sync(device)
    return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs Triton BALM projector.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=340)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--hc-iters", type=int, default=20)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--hc-balm-r", type=float, default=1.0)
    parser.add_argument("--hc-balm-delta", type=float, default=1e-6)
    parser.add_argument("--hc-balm-diag-cost", type=float, default=0.0)
    parser.add_argument("--hc-balm-offdiag-cost", type=float, default=0.0)
    parser.add_argument("--hc-balm-cost-scale", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-grad", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    if args.device == "cuda":
        args.device = "cuda:0"

    print("== BALM Kernel Benchmark ==")
    print(f"device:            {args.device}")
    print(f"dtype:             {args.dtype}")
    print(f"triton_available:  {is_triton_balm_available()}")
    print(f"shape:             [{args.batch_size}, {args.seq_len}, {args.hc_mult}, {args.hc_mult}]")
    print(f"hc_iters:          {args.hc_iters}")

    torch.manual_seed(0)
    comb = torch.rand(
        args.batch_size,
        args.seq_len,
        args.hc_mult,
        args.hc_mult,
        device=args.device,
        dtype=dtype,
    )
    torch_module = make_module(args, "torch")
    torch_out = torch_module._project_comb_balm(comb)
    torch_ms = time_projector(torch_module, comb, args.warmup, args.repeat, args.device)
    print(f"torch_forward_ms:  {torch_ms:.4f}")

    if args.device.startswith("cuda") and is_triton_balm_available():
        triton_module = make_module(args, "triton")
        triton_out = triton_module._project_comb_balm(comb)
        max_abs = (torch_out - triton_out).abs().max().item()
        max_rel = ((torch_out - triton_out).abs() / torch_out.abs().clamp_min(1e-8)).max().item()
        triton_ms = time_projector(triton_module, comb, args.warmup, args.repeat, args.device)
        print(f"triton_forward_ms: {triton_ms:.4f}")
        print(f"speedup:           {torch_ms / triton_ms:.3f}x")
        print(f"forward_max_abs:   {max_abs:.6e}")
        print(f"forward_max_rel:   {max_rel:.6e}")

        if not args.skip_grad:
            grad = torch.randn_like(torch_out)
            comb_ref = comb.detach().clone().requires_grad_(True)
            comb_tri = comb.detach().clone().requires_grad_(True)
            ref_out = torch_module._project_comb_balm(comb_ref)
            tri_out = triton_module._project_comb_balm(comb_tri)
            ref_out.backward(grad)
            tri_out.backward(grad)
            grad_max_abs = (comb_ref.grad - comb_tri.grad).abs().max().item()
            grad_max_rel = (
                (comb_ref.grad - comb_tri.grad).abs() / comb_ref.grad.abs().clamp_min(1e-8)
            ).max().item()
            print(f"grad_max_abs:      {grad_max_abs:.6e}")
            print(f"grad_max_rel:      {grad_max_rel:.6e}")
    else:
        print("triton_forward_ms: skipped (requires CUDA and Triton)")


if __name__ == "__main__":
    main()
