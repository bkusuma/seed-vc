"""Conditional flow matching for Seed-VC v3.

* ``CFMv3.forward``: training loss with mixed attention masks. Each batch
  samples "full" or "prompt_isolated" attention, and with probability
  ``crop_prob`` truncates the target to a short span so the model also sees
  window-like right edges. One model then serves offline inference (full) and
  the cached streaming path (prompt_isolated). The mixed-mask recipe follows
  CosyVoice 2's chunk-aware flow matching (arXiv 2412.10117).
* ``CFMv3.inference``: offline Euler sampler (v1 semantics + optional sway
  sampling and prompt isolation).
* ``PrefixCachedSampler``: streaming sampler. The reference prompt's per-layer
  K/V are computed once per diffusion step; each window then costs only its own
  tokens.
"""
import random

import torch

from modules.v3.dit import ATTN_FULL, ATTN_MODES, ATTN_PROMPT_ISOLATED, PrefixKVStore, StreamingDiT


def make_t_span(n_steps, device, sway_coef=None):
    """Euler time grid on [0, 1]. ``sway_coef`` (e.g. -1) applies F5-TTS-style sway sampling,
    the same transform v2's CFM uses: t + s * (cos(pi/2 t) - 1 + t)."""
    t_span = torch.linspace(0, 1, n_steps + 1, device=device)
    if sway_coef is not None:
        t_span = t_span + sway_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    return t_span


class CFMv3(torch.nn.Module):
    def __init__(self, args, attn_mode_probs=None, crop_prob=0.3, crop_min_frames=32, crop_max_frames=256,
                 sigma_min=1e-6):
        super().__init__()
        self.sigma_min = sigma_min
        self.estimator = StreamingDiT(args)
        self.in_channels = args.DiT.in_channels
        self.criterion = torch.nn.MSELoss() if args.get("reg_loss_type", "l1") == "l2" else torch.nn.L1Loss()
        probs = dict(attn_mode_probs or {ATTN_FULL: 0.5, ATTN_PROMPT_ISOLATED: 0.5})
        unknown = set(probs) - set(ATTN_MODES)
        if unknown:
            raise ValueError(f"unknown attention modes {unknown}")
        self.attn_modes = list(probs.keys())
        self.attn_weights = [float(probs[m]) for m in self.attn_modes]
        self.crop_prob = crop_prob
        self.crop_min_frames = crop_min_frames
        self.crop_max_frames = crop_max_frames

    def sample_attn_mode(self):
        return random.choices(self.attn_modes, weights=self.attn_weights, k=1)[0]

    def forward(self, x1, x_lens, prompt_lens, mu, style, attn_mode=None):
        """Flow-matching loss.

        x1: (B, C, T) target mel; x_lens: (B,); prompt_lens: (B,) (< x_lens);
        mu: (B, T, D) regulated content; style: (B, S).
        Returns (loss, attn_mode, x_lens_used).
        """
        b, _, _ = x1.shape
        attn_mode = attn_mode or self.sample_attn_mode()
        x_lens = x_lens.clone()
        if self.crop_prob > 0 and random.random() < self.crop_prob:
            span = torch.randint(self.crop_min_frames, self.crop_max_frames + 1, (b,), device=x_lens.device)
            x_lens = torch.minimum(x_lens, prompt_lens + span)

        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        frame = torch.arange(x1.size(2), device=x1.device)[None, None, :]
        in_prompt = frame < prompt_lens[:, None, None]
        prompt = torch.where(in_prompt, x1, torch.zeros_like(x1))
        y = torch.where(in_prompt, torch.zeros_like(y), y)

        out = self.estimator(y, prompt, x_lens, t.view(b), style, mu, prompt_lens=prompt_lens, attn_mode=attn_mode)
        loss = 0
        for i in range(b):
            p, n = int(prompt_lens[i]), int(x_lens[i])
            loss = loss + self.criterion(out[i, :, p:n], u[i, :, p:n])
        return loss / b, attn_mode, x_lens

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, n_timesteps, temperature=1.0, inference_cfg_rate=0.7,
                  attn_mode=ATTN_FULL, sway_coef=None, noise=None):
        """Offline Euler sampling over [prompt | source].

        mu: (B, T, D) condition for the whole sequence (prompt first); prompt: (B, C, P).
        Returns (B, C, T) with the prompt region zeroed (slice it off as in v1).
        """
        B, T = mu.size(0), mu.size(1)
        x = noise.clone() if noise is not None else torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = make_t_span(n_timesteps, mu.device, sway_coef)
        P = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :P] = prompt[..., :P]
        x[..., :P] = 0
        prompt_lens = torch.full((B,), P, dtype=torch.long, device=mu.device)
        for step in range(1, len(t_span)):
            t = t_span[step - 1]
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                v = self.estimator(
                    torch.cat([x, x]), torch.cat([prompt_x, torch.zeros_like(prompt_x)]), torch.cat([x_lens, x_lens]),
                    t.expand(2 * B), torch.cat([style, torch.zeros_like(style)]), torch.cat([mu, torch.zeros_like(mu)]),
                    prompt_lens=torch.cat([prompt_lens, prompt_lens]), attn_mode=attn_mode,
                )
                v_cond, v_uncond = v.chunk(2, dim=0)
                v = (1.0 + inference_cfg_rate) * v_cond - inference_cfg_rate * v_uncond
            else:
                v = self.estimator(x, prompt_x, x_lens, t.expand(B), style, mu, prompt_lens=prompt_lens,
                                   attn_mode=attn_mode)
            x = x + dt * v
            x[:, :, :P] = 0
        return x


class PrefixCachedSampler:
    """Streaming sampler with a cached reference prefix (prompt-isolated attention).

    Equivalent to ``CFMv3.inference(attn_mode="prompt_isolated")`` on
    [prompt | window] with the same noise, but each window only computes its
    own W tokens per step instead of P + W.
    """

    def __init__(self, cfm, prompt_mel, prompt_cond, style, n_steps, cfg_rate=0.0, sway_coef=None):
        if prompt_mel.size(0) != 1:
            raise ValueError("PrefixCachedSampler handles a single reference (batch 1)")
        self.estimator = cfm.estimator
        self.n_steps = int(n_steps)
        self.cfg_rate = float(cfg_rate)
        self.style = style
        self.t_span = make_t_span(self.n_steps, prompt_mel.device, sway_coef)
        self.store = PrefixKVStore()
        if self.cfg_rate > 0:
            mel_b = torch.cat([prompt_mel, torch.zeros_like(prompt_mel)])
            cond_b = torch.cat([prompt_cond, torch.zeros_like(prompt_cond)])
            style_b = torch.cat([style, torch.zeros_like(style)])
        else:
            mel_b, cond_b, style_b = prompt_mel, prompt_cond, style
        self._style_b = style_b
        for k in range(self.n_steps):
            t = self.t_span[k].expand(mel_b.size(0))
            self.estimator.forward_prefix(mel_b, cond_b, style_b, t, self.store, k)

    @property
    def prefix_len(self):
        return self.store.prefix_len

    def cache_bytes(self):
        return self.store.memory_bytes()

    @torch.no_grad()
    def sample(self, cond, noise):
        """cond: (1, W, D) window condition; noise: (1, C, W) initial state -> (1, C, W) mel."""
        x = noise.clone()
        for k in range(self.n_steps):
            t = self.t_span[k]
            dt = self.t_span[k + 1] - t
            if self.cfg_rate > 0:
                v = self.estimator.forward_source(
                    torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]), self._style_b, t.expand(2),
                    self.store, k,
                )
                v_cond, v_uncond = v.chunk(2, dim=0)
                v = (1.0 + self.cfg_rate) * v_cond - self.cfg_rate * v_uncond
            else:
                v = self.estimator.forward_source(x, cond, self.style, t.expand(1), self.store, k)
            x = x + dt * v
        return x
