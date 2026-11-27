"""Streaming-capable DiT for Seed-VC v3.

``StreamingDiT`` subclasses the v1 ``DiT`` (identical parameters, so v1
checkpoints warm-start it) and adds:

  * Explicit attention modes for training/offline inference:
      - "full": v1 behaviour, every token attends to every valid token.
      - "prompt_isolated": prefix tokens (time/style tokens + prompt frames)
        attend only to the prefix; source frames attend to everything valid.
  * A cached-prefix execution path. Under "prompt_isolated" the prefix K/V of
    each layer depend only on the reference and the diffusion time t, so they
    are recorded once per reference and step (``forward_prefix``) and reused by
    every streaming window (``forward_source``), which then only runs the
    window's tokens. This hooks into v1 ``Attention.kv_cache.update`` (called
    after RoPE), so the v1 attention code is unchanged.

With the MLP output head the cached path is exact (tests/v3). With the WaveNet
head (time convolutions) the first few frames of a window see zero padding
instead of prompt frames; keep the DiT left context >= the WaveNet receptive
field in that case.
"""
import contextlib

import torch

from modules.diffusion_transformer import DiT

ATTN_FULL = "full"
ATTN_PROMPT_ISOLATED = "prompt_isolated"
ATTN_MODES = (ATTN_FULL, ATTN_PROMPT_ISOLATED)


def build_attention_mask(lens, total_len, prefix_lens=None):
    """Boolean SDPA mask (B, 1, L, L); True = may attend.

    lens: (B,) valid tokens; prefix_lens: (B,) prefix tokens for prompt isolation.
    Every query row keeps at least one True entry, so SDPA never sees an empty row.
    """
    pos = torch.arange(total_len, device=lens.device)
    key_valid = pos[None, :] < lens[:, None]
    mask = key_valid[:, None, :].expand(-1, total_len, -1)
    if prefix_lens is not None:
        in_prefix = pos[None, :] < prefix_lens[:, None]
        mask = mask & (~in_prefix[:, :, None] | in_prefix[:, None, :])
    return mask.unsqueeze(1)


class PrefixKVStore:
    """Per-diffusion-step K/V of the prefix tokens for every transformer layer."""

    def __init__(self):
        self.entries = {}  # step -> list of (k, v), each (B, H, P, Dh)
        self.prefix_len = None
        self.batch_size = None

    def clear(self):
        self.entries.clear()
        self.prefix_len = None
        self.batch_size = None

    def memory_bytes(self):
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for layers in self.entries.values() for k, v in layers)


class _AttentionHook:
    """Placed on v1 ``Attention.kv_cache``; v1 calls ``update(input_pos, k, v)`` after RoPE."""

    def __init__(self, store, layer_idx, n_layers, mode, step):
        self.store, self.layer_idx, self.n_layers, self.mode, self.step = store, layer_idx, n_layers, mode, step

    def update(self, input_pos, k, v):
        if self.mode == "record":
            layers = self.store.entries.setdefault(self.step, [None] * self.n_layers)
            layers[self.layer_idx] = (k, v)
            return k, v
        k_prefix, v_prefix = self.store.entries[self.step][self.layer_idx]
        return torch.cat([k_prefix, k], dim=2), torch.cat([v_prefix, v], dim=2)


class StreamingDiT(DiT):
    def __init__(self, args):
        super().__init__(args)
        if self.is_causal:
            raise NotImplementedError("is_causal DiT is not supported by the v3 streaming path")
        self.n_special_tokens = int(self.time_as_token) + int(self.style_as_token)

    # ------------------------------------------------------------------ helpers
    @contextlib.contextmanager
    def _prefix_cache(self, store, mode, step):
        layers = self.transformer.layers
        for i, layer in enumerate(layers):
            layer.attention.kv_cache = _AttentionHook(store, i, len(layers), mode, step)
        try:
            yield
        finally:
            for layer in layers:
                layer.attention.kv_cache = None

    def _frame_tokens(self, x, prompt_x, style, cond, drop_condition):
        """Per-frame transformer input, exactly as v1 DiT.forward builds it. -> (B, T, H)"""
        T = x.size(2)
        cond = self.cond_projection(cond)
        x_in = torch.cat([x.transpose(1, 2), prompt_x.transpose(1, 2), cond], dim=-1)
        if self.transformer_style_condition and not self.style_as_token:
            x_in = torch.cat([x_in, style[:, None, :].expand(-1, T, -1)], dim=-1)
        if drop_condition:
            x_in = torch.cat([x_in[..., : self.in_channels], torch.zeros_like(x_in[..., self.in_channels:])], dim=-1)
        return self.cond_x_merge_linear(x_in)

    def _special_tokens(self, t1, style, drop_condition):
        """v1 token order after prepending: [time, style, frames...]."""
        tokens = []
        if self.time_as_token:
            tokens.append(t1.unsqueeze(1))
        if self.style_as_token:
            s = self.style_in(style)
            tokens.append((torch.zeros_like(s) if drop_condition else s).unsqueeze(1))
        return tokens

    def _head(self, x_res, x, t, t1, frame_mask):
        """v1 output head. x_res: (B, T, H) frame outputs; x: (B, C, T) noisy input."""
        if self.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x.transpose(1, 2)], dim=-1))
        if self.final_layer_type == "wavenet":
            h = self.conv1(x_res).transpose(1, 2)
            t2 = self.t_embedder2(t)
            h = self.wavenet(h, frame_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(x_res)
            h = self.final_layer(h, t1).transpose(1, 2)
            return self.conv2(h)
        return self.final_mlp(x_res).transpose(1, 2)

    # ---------------------------------------------------------- full sequence
    def forward(self, x, prompt_x, x_lens, t, style, cond, prompt_lens=None, attn_mode=ATTN_FULL,
                mask_content=False):
        """
        x, prompt_x: (B, C, T); x_lens: (B,); t: (B,); style: (B, S); cond: (B, T, D)
        prompt_lens: (B,) prompt frames at the start of each item (needed for prompt isolation).
        Returns the predicted velocity (B, C, T).
        """
        if attn_mode not in ATTN_MODES:
            raise ValueError(f"attn_mode must be one of {ATTN_MODES}, got {attn_mode}")
        drop = bool(self.training and torch.rand(1) < self.class_dropout_prob)
        if not self.training and isinstance(mask_content, bool) and mask_content:
            drop = True
        B, _, T = x.shape
        t1 = self.t_embedder(t)
        x_in = self._frame_tokens(x, prompt_x, style, cond, drop)
        special = self._special_tokens(t1, style, drop)
        if special:
            x_in = torch.cat(special + [x_in], dim=1)
        n_sp = self.n_special_tokens
        L = x_in.size(1)
        lens = x_lens.to(x.device).long() + n_sp
        prefix = None
        if attn_mode == ATTN_PROMPT_ISOLATED:
            if prompt_lens is None:
                raise ValueError("prompt_isolated attention needs prompt_lens")
            prefix = prompt_lens.to(x.device).long() + n_sp
        mask = build_attention_mask(lens, L, prefix)
        x_res = self.transformer(x_in, t1.unsqueeze(1), self.input_pos[:L], mask)[:, n_sp:]
        frame_mask = (torch.arange(T, device=x.device)[None] < x_lens.to(x.device)[:, None]).unsqueeze(1).to(x.dtype)
        return self._head(x_res, x, t, t1, frame_mask)

    # --------------------------------------------------- cached prefix (stream)
    @torch.no_grad()
    def forward_prefix(self, prompt_mel, prompt_cond, style, t, store, step):
        """Record per-layer K/V of [special tokens | prompt frames] for diffusion step ``step``.

        prompt_mel: (B, C, P); prompt_cond: (B, P, D); style: (B, S); t: (B,)
        The prompt region of the noisy input is zero, as in v1 ``solve_euler``.
        """
        B, _, P = prompt_mel.shape
        t1 = self.t_embedder(t)
        x_in = self._frame_tokens(torch.zeros_like(prompt_mel), prompt_mel, style, prompt_cond, False)
        special = self._special_tokens(t1, style, False)
        if special:
            x_in = torch.cat(special + [x_in], dim=1)
        L = x_in.size(1)
        mask = torch.ones(B, 1, L, L, dtype=torch.bool, device=x_in.device)
        with self._prefix_cache(store, "record", step):
            self.transformer(x_in, t1.unsqueeze(1), self.input_pos[:L], mask)
        store.prefix_len = L
        store.batch_size = B

    @torch.no_grad()
    def forward_source(self, x, cond, style, t, store, step):
        """Velocity for a source window that follows the cached prefix.

        x: (B, C, W) noisy window; cond: (B, W, D); style: (B, S); t: (B,)
        Positions continue after the prefix, exactly as in the single-pass layout
        [special | prompt | window].
        """
        B, _, W = x.shape
        if store.batch_size != B:
            raise ValueError(f"prefix cache was recorded for batch {store.batch_size}, got {B}")
        t1 = self.t_embedder(t)
        x_in = self._frame_tokens(x, torch.zeros_like(x), style, cond, False)
        P = store.prefix_len
        mask = torch.ones(B, 1, W, P + W, dtype=torch.bool, device=x.device)
        with self._prefix_cache(store, "use", step):
            x_res = self.transformer(x_in, t1.unsqueeze(1), self.input_pos[P:P + W], mask)
        frame_mask = torch.ones(B, 1, W, dtype=x.dtype, device=x.device)
        return self._head(x_res, x, t, t1, frame_mask)
