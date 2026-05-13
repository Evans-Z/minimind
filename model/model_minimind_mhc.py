import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from model.model_minimind import (
    MiniMindConfig,
    RMSNorm,
    precompute_freqs_cis,
    Attention,
    FeedForward,
    MOEFeedForward,
)
from model.model_mhc import HyperConnection, HyperHead


class MiniMindMHCConfig(MiniMindConfig):
    model_type = "minimind_mhc"

    def __init__(
        self,
        hc_mult=4,
        hc_iters=20,
        hc_eps=1e-6,
        initializer_range=0.02,
        hc_projector="sinkhorn",
        hc_comb_activation="sigmoid",
        hc_balm_r=1.0,
        hc_balm_trainable_r=False,
        hc_balm_delta=1e-6,
        hc_balm_diag_cost=0.0,
        hc_balm_offdiag_cost=0.0,
        hc_balm_cost_scale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hc_mult = hc_mult
        self.hc_iters = hc_iters
        self.hc_eps = hc_eps
        self.initializer_range = initializer_range
        self.hc_projector = hc_projector
        self.hc_comb_activation = hc_comb_activation
        self.hc_balm_r = hc_balm_r
        self.hc_balm_trainable_r = hc_balm_trainable_r
        self.hc_balm_delta = hc_balm_delta
        self.hc_balm_diag_cost = hc_balm_diag_cost
        self.hc_balm_offdiag_cost = hc_balm_offdiag_cost
        self.hc_balm_cost_scale = hc_balm_cost_scale


class MiniMindMHCBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindMHCConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

        self.attn_hc = HyperConnection(
            hc_mult=config.hc_mult,
            hidden_size=config.hidden_size,
            hc_iters=config.hc_iters,
            hc_eps=config.hc_eps,
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            hc_projector=config.hc_projector,
            hc_comb_activation=config.hc_comb_activation,
            hc_balm_r=config.hc_balm_r,
            hc_balm_delta=config.hc_balm_delta,
            hc_balm_diag_cost=config.hc_balm_diag_cost,
            hc_balm_offdiag_cost=config.hc_balm_offdiag_cost,
            hc_balm_cost_scale=config.hc_balm_cost_scale,
            hc_balm_trainable_r=config.hc_balm_trainable_r,
        )
        self.mlp_hc = HyperConnection(
            hc_mult=config.hc_mult,
            hidden_size=config.hidden_size,
            hc_iters=config.hc_iters,
            hc_eps=config.hc_eps,
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            hc_projector=config.hc_projector,
            hc_comb_activation=config.hc_comb_activation,
            hc_balm_r=config.hc_balm_r,
            hc_balm_delta=config.hc_balm_delta,
            hc_balm_diag_cost=config.hc_balm_diag_cost,
            hc_balm_offdiag_cost=config.hc_balm_offdiag_cost,
            hc_balm_cost_scale=config.hc_balm_cost_scale,
            hc_balm_trainable_r=config.hc_balm_trainable_r,
        )

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # hidde_states throughout: [B, S, hc_mult, hidden]
        # `post` / `comb` come out of the HC modules in fp32
        # the .to(dtype) puts everything back to the input dype before mixing
        # so both sites stay consistent with `hidden_states`'s entry dtype.
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(collapsed),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        post, comb, collapsed = self.mlp_hc(hidden_states)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed))
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )
        return hidden_states, present_key_value


class MiniMindMHCModel(nn.Module):
    def __init__(self, config: MiniMindMHCConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindMHCBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.hc_head = HyperHead(
            config.hc_mult,
            config.hidden_size,
            config.hc_eps,
            config.rms_norm_eps,
            config.initializer_range,
        )

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        del kwargs
        _, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # [B, S, hc_mult, hidden]
        hidden_streams = hidden_states.unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_streams, present = layer(
                hidden_streams,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(self.hc_head(hidden_streams))
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss


class MiniMindMHCForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindMHCConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindMHCConfig = None):
        self.config = config or MiniMindMHCConfig()
        super().__init__(self.config)
        self.model = MiniMindMHCModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = (
                torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
                if attention_mask is not None
                else None
            )
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample
                else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
