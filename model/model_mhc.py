import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections
    (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
    Transformer blocks

    Owns the learned (`fn`, `base`, `scale`)
    parameters that turn the incoming `hc_mult` residual streams into collapse / expand
    weights. The decoder layer instantiates two of these (one for the attention site,
    one for the mlp site).

    ASCII shape guide — `B` = batch, `S` = seq, `H` = hc_mult, `D` = hidden_size::

              hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                             mix-logits
                                                             [B, S, (2+H)*H]
                                                                    │
                            ┌───────────────────────────────────────┴──────────────────────────────┐
                            ▼                          ▼                                           ▼
                        pre logits                post logits                               comb logits
                        [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                        * scale[0]                * scale[1]                                * scale[2]
                        + base[:H]                + base[H:2H]                              + base[2H:]
                        sigma() + eps             sigma() + eps                             sigma() + eps
                        │                         │                                         │
                        pre                       post                                     Sinkhorn(iters)
                        (stream collapse weights) (block-output placement)                 row/col normalise
                                                                                            │
                                                                                            comb
                                                                                            (stream mixer)
    """
    def __init__(
        self,
        hc_mult: int,
        hidden_size: int,
        hc_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
        initializer_range: float = 0.02,
        hc_projector: str = "sinkhorn",
        hc_balm_r: float = 1.0,
        hc_balm_delta: float = 1e-6,
        hc_balm_diag_cost: float = 0.0,
        hc_balm_offdiag_cost: float = 0.0,
        hc_balm_l2_cost: float = 0.0,
        hc_balm_cost_mode: str = "fixed",
        hc_balm_cost_scale: float = 1.0,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.hc_iters = hc_iters
        self.hc_eps = hc_eps
        self.initializer_range = initializer_range
        self.hc_projector = hc_projector.lower()
        self.hc_balm_r = hc_balm_r
        self.hc_balm_delta = hc_balm_delta
        self.hc_balm_diag_cost = hc_balm_diag_cost
        self.hc_balm_offdiag_cost = hc_balm_offdiag_cost
        self.hc_balm_l2_cost = hc_balm_l2_cost
        self.hc_balm_cost_mode = hc_balm_cost_mode.lower()
        self.hc_balm_cost_scale = hc_balm_cost_scale
        if self.hc_projector not in {"sinkhorn", "balm"}:
            raise ValueError(f"Unsupported hc_projector={hc_projector!r}. Expected 'sinkhorn' or 'balm'.")
        if self.hc_balm_cost_mode not in {"fixed", "learned", "learned_static"}:
            raise ValueError(
                f"Unsupported hc_balm_cost_mode={hc_balm_cost_mode!r}. Expected 'fixed', 'learned', or 'learned_static'."
            )
        if self.hc_balm_r <= 0:
            raise ValueError(f"hc_balm_r must be positive, got {self.hc_balm_r}")
        if self.hc_balm_delta <= 0:
            raise ValueError(f"hc_balm_delta must be positive, got {self.hc_balm_delta}")
        if self.hc_balm_diag_cost < 0:
            raise ValueError(f"hc_balm_diag_cost must be non-negative, got {self.hc_balm_diag_cost}")
        if self.hc_balm_offdiag_cost < 0:
            raise ValueError(f"hc_balm_offdiag_cost must be non-negative, got {self.hc_balm_offdiag_cost}")
        if self.hc_balm_l2_cost < 0:
            raise ValueError(f"hc_balm_l2_cost must be non-negative, got {self.hc_balm_l2_cost}")
        if self.hc_balm_cost_scale < 0:
            raise ValueError(f"hc_balm_cost_scale must be non-negative, got {self.hc_balm_cost_scale}")
        self.input_norm = UnweightedRMSNorm(rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * self.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))
        if self.hc_balm_cost_mode == "learned":
            self.cost_fn = nn.Parameter(torch.empty(self.hc_mult * self.hc_mult, self.hc_mult * self.hidden_size))
            self.cost_base = nn.Parameter(torch.empty(self.hc_mult * self.hc_mult))
        elif self.hc_balm_cost_mode == "learned_static":
            self.cost_static = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        init.normal_(self.fn, mean=0.0, std=self.initializer_range)
        init.zeros_(self.base)
        init.ones_(self.scale)
        if hasattr(self, "cost_fn"):
            init.normal_(self.cost_fn, mean=0.0, std=self.initializer_range)
            init.zeros_(self.cost_base)
        if hasattr(self, "cost_static"):
            init.zeros_(self.cost_static)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via the selected
        projector (`sinkhorn` or `balm`) for `hc_iters` steps. `pre` then
        collapses the `hc_mult` parallel streams into a single sequence (input
        projection into the sublayer); `post` and `comb` are returned for the
        caller to apply on the sublayer output.
        """
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(flat, self.fn.float())  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc:2*hc] * post_scale + self.base[hc:2*hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2*hc:].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2*hc:].view(hc, hc)
            )
            + self.hc_eps
        )
        if self.hc_projector == "sinkhorn":
            for _ in range(self.hc_iters):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        elif self.hc_projector == "balm":
            # Balanced ALM projection with assignment-structured operators:
            # Ah = [row_sum(H); col_sum(H)], (A^T y)_ij = u_i + v_j.
            # Optional objective terms in primal step:
            #   <C, H> + (lambda/2)||H||_F^2.
            hc = self.hc_mult
            u = torch.zeros(*comb.shape[:-2], hc, device=comb.device, dtype=comb.dtype)
            v = torch.zeros(*comb.shape[:-2], hc, device=comb.device, dtype=comb.dtype)
            if self.hc_balm_cost_mode == "learned":
                cost_logits = F.linear(flat, self.cost_fn.float(), self.cost_base.float())
                linear_cost = (
                    F.softplus(cost_logits).view(*mix.shape[:-1], hc, hc) + self.hc_eps
                ) * self.hc_balm_cost_scale
            elif self.hc_balm_cost_mode == "learned_static":
                linear_cost = (
                    F.softplus(self.cost_static.float()).view(1, 1, hc, hc) + self.hc_eps
                ) * self.hc_balm_cost_scale
            else:
                ones = torch.ones(hc, hc, device=comb.device, dtype=comb.dtype)
                eye = torch.eye(hc, device=comb.device, dtype=comb.dtype)
                # C = offdiag * 1 + (diag - offdiag) * I
                linear_cost = (
                    self.hc_balm_offdiag_cost * ones
                    + (self.hc_balm_diag_cost - self.hc_balm_offdiag_cost) * eye
                ) * self.hc_balm_cost_scale
            balm_step = self.hc_balm_r / (float(hc) + float(self.hc_balm_delta))
            z_denom = 2.0 * float(hc) + float(self.hc_balm_delta)
            for _ in range(self.hc_iters):
                at_y = u.unsqueeze(-1) + v.unsqueeze(-2)
                if self.hc_balm_l2_cost > 0:
                    q = (
                        self.hc_balm_r * comb + at_y - linear_cost
                    ) / (self.hc_balm_r + self.hc_balm_l2_cost)
                else:
                    q = comb + (at_y - linear_cost) / self.hc_balm_r
                comb_next = torch.clamp(q, min=0.0)
                residual = 2.0 * comb_next - comb
                row_sum = residual.sum(dim=-1)
                col_sum = residual.sum(dim=-2)
                z = (row_sum.sum(dim=-1) - float(hc)) / z_denom
                u = u - balm_step * (row_sum - 1.0 - z.unsqueeze(-1))
                v = v - balm_step * (col_sum - 1.0 - z.unsqueeze(-1))
                comb = comb_next
        else:
            raise NotImplementedError(f"Unsupported hc_projector={self.hc_projector!r}")
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=-2).to(hidden_streams.dtype)
        return post, comb, collapsed


class HyperHead(nn.Module):
    """Final HC-stram collapse; used by MiniMindMHCModel before the shared RMSNorm"""

    def __init__(
        self, 
        hc_mult: int, 
        hidden_size: int, 
        hc_eps: float,
        rms_norm_eps: float,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)
        self.eps = hc_eps
        self.initializer_range = initializer_range
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        init.normal_(self.hc_fn, mean=0.0, std=self.initializer_range)
        init.zeros_(self.hc_base)
        init.ones_(self.hc_scale)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_states.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps

        return (pre.unsqueeze(-1) * hidden_states).sum(dim=-2).to(hidden_states.dtype)