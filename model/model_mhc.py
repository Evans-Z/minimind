import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def safe_logit(p: float, eps: float) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


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
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.hc_iters = hc_iters
        self.hc_eps = hc_eps
        self.input_norm = UnweightedRMSNorm(rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * self.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))

        # initialise the parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        hc = self.hc_mult

        with torch.no_grad():
            # set fn to zero
            self.fn.data.zero_()
            # set scale to small positive values
            self.scale.data.fill_(0.01)
            # set base strategitally, so that the initial output 
            # is close to the identity mapping
            # pre and post targets (after sigmoid + eps)
            p_pre = max(1.0 / hc - self.hc_eps, self.hc_eps)
            p_post = max(0.95 - self.hc_eps, self.hc_eps)

            self.base.data[:hc].fill_(safe_logit(p_pre, self.hc_eps))
            self.base.data[hc:2*hc].fill_(safe_logit(p_post, self.hc_eps))
            
            # comb target: identity-like matrix
            # diagonal ~ 1.0 / hc, off-diagonal ~ 0.0
            diag_p = max(1.0 / hc - self.hc_eps, self.hc_eps)
            off_p = self.hc_eps
            
            comb = torch.full((hc, hc), safe_logit(off_p, self.hc_eps), device=self.base.device, dtype=self.base.dtype)
            comb.fill_diagonal_(safe_logit(diag_p, self.hc_eps))
            self.base.data[2*hc:].copy_(comb.reshape(-1))

    
    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-
        Knopp: starting from the sigmoid-positive matrix, alternate row and
        column normalisation for `hc_sinkhorn_iters` steps. `pre` then collapses
        the `hc_mult` parallel streams into a single sequence (input projection
        into the sublayer); `post` and `comb` are returned for the caller to
        apply on the sublayer output.
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
        for _ in range(self.hc_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=-2).to(hidden_streams.dtype)
        return post, comb, collapsed