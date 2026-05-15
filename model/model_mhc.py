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
    def _validate_config(self):
        if self.hc_projector not in {"sinkhorn", "balm"}:
            raise ValueError(f"Unsupported hc_projector={self.hc_projector!r}. Expected 'sinkhorn' or 'balm'.")
        if self.hc_balm_r <= 0:
            raise ValueError(f"hc_balm_r must be positive, got {self.hc_balm_r}")
        if self.hc_balm_delta <= 0:
            raise ValueError(f"hc_balm_delta must be positive, got {self.hc_balm_delta}")
        if self.hc_balm_diag_cost < 0:
            raise ValueError(f"hc_balm_diag_cost must be non-negative, got {self.hc_balm_diag_cost}")
        if self.hc_balm_offdiag_cost < 0:
            raise ValueError(f"hc_balm_offdiag_cost must be non-negative, got {self.hc_balm_offdiag_cost}")
        if self.hc_balm_cost_scale < 0:
            raise ValueError(f"hc_balm_cost_scale must be non-negative, got {self.hc_balm_cost_scale}")

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
        hc_balm_cost_scale: float = 1.0,
        hc_balm_trainable_r: bool = False,
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
        self.hc_balm_cost_scale = hc_balm_cost_scale
        self.hc_balm_trainable_r = hc_balm_trainable_r
        self._validate_config()

        # projector setup
        if self.hc_projector == "sinkhorn":
            self._setup_sinkhorn()
        elif self.hc_projector == "balm":
            self._setup_balm()
        else:
            raise NotImplementedError(f"Unsupported hc_projector={self.hc_projector!r}")
        
        # weights of mHC
        self.input_norm = UnweightedRMSNorm(rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * self.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))
        
        self.init_weights()
    
    def _setup_sinkhorn(self):
        self._project_comb = self._project_comb_sinkhorn

    def _setup_balm(self):
        self._project_comb = self._project_comb_balm
        if self.hc_balm_trainable_r:
            # raw_r is unconstrained; softplus(raw_r) + hc_eps keeps r strictly positive.
            target_r = max(self.hc_balm_r - self.hc_eps, 1e-12)
            raw_init = torch.log(torch.expm1(torch.tensor(target_r, dtype=torch.float32)))
            self.hc_balm_raw_r = nn.Parameter(raw_init)
        else:
            self.hc_balm_raw_r = None
        # Balanced ALM constants: 1/(2n+delta), linear cost matrix
        self.inv_z_denom = 1.0 / (2.0 * self.hc_mult + self.hc_balm_delta)
        self.linear_cost = (
            self.hc_balm_offdiag_cost * torch.ones(self.hc_mult, self.hc_mult)
            + (self.hc_balm_diag_cost - self.hc_balm_offdiag_cost) * torch.eye(self.hc_mult)
        ) * self.hc_balm_cost_scale

    def _get_hc_balm_r(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.hc_balm_trainable_r:
            return F.softplus(self.hc_balm_raw_r.to(device=device, dtype=dtype)) + self.hc_eps
        return torch.tensor(self.hc_balm_r, device=device, dtype=dtype)

    @torch.no_grad()
    def init_weights(self):
        init.normal_(self.fn, mean=0.0, std=self.initializer_range)
        init.zeros_(self.base)
        init.ones_(self.scale)

    def _project_comb_sinkhorn(self, comb: torch.Tensor) -> torch.Tensor:
        for _ in range(self.hc_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return comb

    def _project_comb_balm(self, comb: torch.Tensor) -> torch.Tensor:
        hc = self.hc_mult
        linear_cost = self.linear_cost.to(device=comb.device, dtype=comb.dtype)
        hc_balm_r = self._get_hc_balm_r(dtype=comb.dtype, device=comb.device)
        balm_step = hc_balm_r / (self.hc_mult + self.hc_balm_delta)
        inv_r = 1.0 / hc_balm_r
        y = torch.zeros(*comb.shape[:-2], 2 * hc, device=comb.device, dtype=comb.dtype)
        comb_row_sum = comb.sum(dim=-1)
        comb_col_sum = comb.sum(dim=-2)
        for _ in range(self.hc_iters):
            u = y[..., :hc]
            v = y[..., hc:]
            at_y = u.unsqueeze(-1) + v.unsqueeze(-2)
            q = comb + (at_y - linear_cost) * inv_r
            comb_next = torch.clamp(q, min=0.0)
            comb_next_row_sum = comb_next.sum(dim=-1)
            comb_next_col_sum = comb_next.sum(dim=-2)
            row_sum = 2.0 * comb_next_row_sum - comb_row_sum
            col_sum = 2.0 * comb_next_col_sum - comb_col_sum
            z = (row_sum.sum(dim=-1) - hc) * self.inv_z_denom
            z_expand = z.unsqueeze(-1)
            y[..., :hc].sub_(balm_step * (row_sum - 1.0 - z_expand))
            y[..., hc:].sub_(balm_step * (col_sum - 1.0 - z_expand))
            comb = comb_next
            comb_row_sum = comb_next_row_sum
            comb_col_sum = comb_next_col_sum
        
        return comb

    def compute_mix(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        return F.linear(flat, self.fn.float())  # [B, S, (2+H)*H]

    def collapse_from_mix(self, mix: torch.Tensor, hidden_streams: torch.Tensor) -> torch.Tensor:
        pre_scale = self.scale[0]
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=-2).to(hidden_streams.dtype)

    def post_comb_from_mix(self, mix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        post = torch.sigmoid(mix[..., hc:2*hc] * post_scale + self.base[hc:2*hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2*hc:].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2*hc:].view(hc, hc)
            )
            + self.hc_eps
        )
        return post, self._project_comb(comb)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via the selected
        projector (`sinkhorn` or `balm`) for `hc_iters` steps. `pre` then
        collapses the `hc_mult` parallel streams into a single sequence (input
        projection into the sublayer); `post` and `comb` are returned for the
        caller to apply on the sublayer output.
        """
        mix = self.compute_mix(hidden_streams)
        post, comb = self.post_comb_from_mix(mix)
        collapsed = self.collapse_from_mix(mix, hidden_streams)
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