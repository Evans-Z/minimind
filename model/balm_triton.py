import math

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def is_triton_balm_available() -> bool:
    return _HAS_TRITON


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def balm_project_reference(
    comb: torch.Tensor,
    linear_cost: torch.Tensor,
    hc_balm_r: float,
    hc_balm_delta: float,
    hc_iters: int,
) -> torch.Tensor:
    hc = comb.shape[-1]
    linear_cost = linear_cost.to(device=comb.device, dtype=comb.dtype)
    hc_balm_r_t = torch.tensor(hc_balm_r, device=comb.device, dtype=comb.dtype)
    balm_step = hc_balm_r_t / (hc + hc_balm_delta)
    inv_r = 1.0 / hc_balm_r_t
    inv_z_denom = 1.0 / (2.0 * hc + hc_balm_delta)
    y = torch.zeros(*comb.shape[:-2], 2 * hc, device=comb.device, dtype=comb.dtype)
    comb_row_sum = comb.sum(dim=-1)
    comb_col_sum = comb.sum(dim=-2)
    for _ in range(hc_iters):
        u = y[..., :hc]
        v = y[..., hc:]
        at_y = u.unsqueeze(-1) + v.unsqueeze(-2)
        q = comb + (at_y - linear_cost) * inv_r
        comb_next = torch.clamp(q, min=0.0)
        comb_next_row_sum = comb_next.sum(dim=-1)
        comb_next_col_sum = comb_next.sum(dim=-2)
        row_sum = 2.0 * comb_next_row_sum - comb_row_sum
        col_sum = 2.0 * comb_next_col_sum - comb_col_sum
        z = (row_sum.sum(dim=-1) - hc) * inv_z_denom
        z_expand = z.unsqueeze(-1)
        y[..., :hc].sub_(balm_step * (row_sum - 1.0 - z_expand))
        y[..., hc:].sub_(balm_step * (col_sum - 1.0 - z_expand))
        comb = comb_next
        comb_row_sum = comb_next_row_sum
        comb_col_sum = comb_next_col_sum
    return comb


if _HAS_TRITON:

    @triton.jit
    def _balm_forward_kernel(
        comb_ptr,
        cost_ptr,
        out_ptr,
        n_mats: tl.constexpr,
        hc: tl.constexpr,
        block_h: tl.constexpr,
        hc_iters: tl.constexpr,
        inv_r: tl.constexpr,
        balm_step: tl.constexpr,
        inv_z_denom: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = tl.arange(0, block_h)
        cols = tl.arange(0, block_h)
        mask_vec = rows < hc
        mask_mat = (rows[:, None] < hc) & (cols[None, :] < hc)
        matrix_offset = pid * hc * hc
        offsets = matrix_offset + rows[:, None] * hc + cols[None, :]
        cost_offsets = rows[:, None] * hc + cols[None, :]

        h = tl.load(comb_ptr + offsets, mask=mask_mat, other=0.0).to(tl.float32)
        cost = tl.load(cost_ptr + cost_offsets, mask=mask_mat, other=0.0).to(tl.float32)
        row_dual = tl.zeros((block_h,), dtype=tl.float32)
        col_dual = tl.zeros((block_h,), dtype=tl.float32)
        row_sum_prev = tl.sum(h, axis=1)
        col_sum_prev = tl.sum(h, axis=0)

        for _ in tl.static_range(0, hc_iters):
            at_y = row_dual[:, None] + col_dual[None, :]
            q = h + (at_y - cost) * inv_r
            h_next = tl.maximum(q, 0.0)
            h_next = tl.where(mask_mat, h_next, 0.0)
            h_next_row_sum = tl.sum(h_next, axis=1)
            h_next_col_sum = tl.sum(h_next, axis=0)
            row_sum = 2.0 * h_next_row_sum - row_sum_prev
            col_sum = 2.0 * h_next_col_sum - col_sum_prev
            z = (tl.sum(tl.where(mask_vec, row_sum, 0.0), axis=0) - hc) * inv_z_denom
            row_dual -= balm_step * (row_sum - 1.0 - z)
            col_dual -= balm_step * (col_sum - 1.0 - z)
            row_dual = tl.where(mask_vec, row_dual, 0.0)
            col_dual = tl.where(mask_vec, col_dual, 0.0)
            h = h_next
            row_sum_prev = h_next_row_sum
            col_sum_prev = h_next_col_sum

        tl.store(out_ptr + offsets, h, mask=mask_mat)


class _BalmTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comb, linear_cost, hc_balm_r: float, hc_balm_delta: float, hc_iters: int):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is not available.")
        if not comb.is_cuda:
            raise RuntimeError("Triton BALM requires CUDA tensors.")
        if comb.shape[-1] != comb.shape[-2]:
            raise ValueError(f"BALM expects square matrices, got shape={tuple(comb.shape)}")

        hc = comb.shape[-1]
        block_h = _next_power_of_2(hc)
        if block_h > 64:
            raise ValueError(f"Triton BALM currently supports hc_mult up to 64, got {hc}")
        comb_contig = comb.contiguous()
        linear_cost_contig = linear_cost.to(device=comb.device, dtype=comb.dtype).contiguous()
        out = torch.empty_like(comb_contig)
        n_mats = math.prod(comb_contig.shape[:-2])
        inv_r = 1.0 / float(hc_balm_r)
        balm_step = float(hc_balm_r) / (float(hc) + float(hc_balm_delta))
        inv_z_denom = 1.0 / (2.0 * float(hc) + float(hc_balm_delta))
        _balm_forward_kernel[(n_mats,)](
            comb_contig,
            linear_cost_contig,
            out,
            n_mats,
            hc,
            block_h,
            int(hc_iters),
            inv_r,
            balm_step,
            inv_z_denom,
            num_warps=1 if block_h <= 16 else 2,
        )
        ctx.save_for_backward(comb_contig, linear_cost_contig)
        ctx.hc_balm_r = float(hc_balm_r)
        ctx.hc_balm_delta = float(hc_balm_delta)
        ctx.hc_iters = int(hc_iters)
        return out.view_as(comb)

    @staticmethod
    def backward(ctx, grad_output):
        comb, linear_cost = ctx.saved_tensors
        with torch.enable_grad():
            comb_ref = comb.detach().requires_grad_(True)
            out_ref = balm_project_reference(
                comb_ref,
                linear_cost,
                hc_balm_r=ctx.hc_balm_r,
                hc_balm_delta=ctx.hc_balm_delta,
                hc_iters=ctx.hc_iters,
            )
            grad_comb = torch.autograd.grad(
                out_ref,
                comb_ref,
                grad_output.contiguous(),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
        return grad_comb, None, None, None, None


def balm_project_triton(
    comb: torch.Tensor,
    linear_cost: torch.Tensor,
    hc_balm_r: float,
    hc_balm_delta: float,
    hc_iters: int,
) -> torch.Tensor:
    return _BalmTritonFunction.apply(comb, linear_cost, hc_balm_r, hc_balm_delta, hc_iters)
