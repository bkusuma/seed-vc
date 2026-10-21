"""Chunk-based, non-autoregressive streaming engine for Seed-VC v3.

Same timeline as real-time-gui.py (sliding window + SOLA), without the GUI and
with the v3 cost reductions:

  * the reference prompt's per-layer K/V are cached once per diffusion step
    (``PrefixCachedSampler``), so each block only runs its window's tokens;
  * mel frames sit on an absolute stream grid, so consecutive windows generate
    the *same* frames (same times, same initial noise from a frame-indexed noise
    bank) in their overlap, which keeps the SOLA crossfade coherent;
  * only the output region (+ ``vocoder_pad_frames`` per side) is vocoded.

Per block (all lengths are multiples of 20 ms, Whisper's frame):

    16 kHz buffer = ce_left | crossfade | search | block | right
    DiT window    =            dit_left | crossfade | search | block | right
    emitted audio =                       [crossfade | search | block]  -> SOLA -> block

Algorithmic latency = block + right + crossfade + search (the device adds its
own input/output buffering on top). Compute per block must stay below
``block_time`` for real-time operation.
"""
import math
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from modules.v3.content_encoder import CHUNK_SECONDS
from modules.v3.content_encoder import SAMPLE_RATE as ENCODER_SR
from modules.v3.device import AUTOCAST_DTYPES, as_float32, autocast, resolve_autocast_dtype, synchronize
from modules.v3.flow_matching import PrefixCachedSampler

QUANTUM = 0.02  # seconds; Whisper frame = SOLA unit
STAGES = ("encoder", "bottleneck_regulator", "dit", "vocoder", "sola")


@dataclass
class StreamingConfig:
    block_time: float = 0.18
    crossfade_time: float = 0.04
    ce_left_context: float = 3.0
    dit_left_context: float = 0.5
    right_context: float = 0.04
    diffusion_steps: int = 10
    cfg_rate: float = 0.7
    sway_coef: Optional[float] = -1.0
    prompt_max_time: float = 3.0
    vocoder_pad_frames: int = 16  # BigVGAN-v2 44k: 8 frames -> 4.2% crop error, 16 -> 0.12%
    seed: int = 1234
    noise_bank_frames: int = 8192
    precision: str = "fp32"  # fp32 | fp16 | bf16: autocast for the DiT (and its prefix cache); vocoder stays fp32

    @classmethod
    def from_config(cls, node=None, **overrides):
        values = {}
        names = {f.name for f in fields(cls)}
        for source in (node or {}, overrides):
            for k, v in dict(source).items():
                if k not in names or (v is None and k != "sway_coef"):  # sway_coef: None disables sway
                    continue
                values[k] = v
        return cls(**values)


class StreamingVoiceConverter:
    def __init__(self, model, config=None):
        self.model = model
        self.cfg = cfg = config or StreamingConfig()
        sr, hop = model.sr, model.hop
        if sr % 50:
            raise ValueError("model sample rate must be a multiple of 50 Hz (20 ms quantum)")
        q = lambda seconds: int(round(seconds / QUANTUM))
        self.n_block = max(1, q(cfg.block_time))
        self.n_cf = min(max(1, q(cfg.crossfade_time)), 4)  # <= 80 ms, as in the GUI
        self.n_search = 1
        self.n_right = max(0, q(cfg.right_context))
        self.n_dit_left = max(0, q(cfg.dit_left_context))
        self.n_ce_left = max(q(cfg.ce_left_context), self.n_dit_left)

        self.zc_in = ENCODER_SR // 50  # 320
        self.zc_out = sr // 50  # 882 at 44.1 kHz
        self.block_in = self.n_block * self.zc_in
        self.block_out = self.n_block * self.zc_out
        self.cf_out = self.n_cf * self.zc_out
        self.search_out = self.n_search * self.zc_out
        self.right_out = self.n_right * self.zc_out
        self.emit_out = self.block_out + self.cf_out + self.search_out

        self.ce_frames = self.n_ce_left + self.n_cf + self.n_search + self.n_block + self.n_right
        encoder = model.content_encoder
        limit = encoder.pad_seconds if encoder.pad_seconds is not None else CHUNK_SECONDS
        if self.ce_frames * QUANTUM > limit:
            raise ValueError(f"content-encoder window {self.ce_frames * QUANTUM:.2f}s exceeds the encoder's "
                             f"{limit:.0f}s padded length; lower ce_left_context")
        self.dit_frames = self.n_dit_left + self.n_cf + self.n_search + self.n_block + self.n_right
        self.window_out = self.dit_frames * self.zc_out
        self.n_mel = math.ceil(self.window_out / hop) + 1

        device = model.device
        self.device = device
        g = torch.Generator().manual_seed(cfg.seed)
        self.noise_bank = torch.randn(model.mel_fn.n_mels, cfg.noise_bank_frames, generator=g).to(device)
        fade = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, self.cf_out, device=device)) ** 2
        self.fade_in, self.fade_out = fade, 1 - fade
        if cfg.precision not in AUTOCAST_DTYPES:
            raise ValueError(f"precision must be one of {sorted(AUTOCAST_DTYPES)}")
        self.autocast_dtype = resolve_autocast_dtype(device, cfg.precision)  # device-supported dtype or None
        self.sampler = None
        self.timings = deque(maxlen=200)
        self.reset()

    def _autocast(self):
        return autocast(self.device, self.autocast_dtype)

    # ----------------------------------------------------------------- state
    def reset(self):
        self.in_buffer = torch.zeros(self.ce_frames * self.zc_in, device=self.device)
        self.sola_buffer = torch.zeros(self.cf_out, device=self.device)
        self.blocks_done = 0

    @property
    def output_sr(self):
        return self.model.sr

    @torch.inference_mode()
    def set_reference(self, ref_wave):
        """ref_wave: 1-D float waveform at the model sample rate (first prompt_max_time seconds are used)."""
        m = self.model
        m._ensure_caches()
        ref = as_float32(ref_wave, self.device).reshape(1, -1)
        ref = ref[:, : int(m.sr * self.cfg.prompt_max_time)]
        mel = m.mel_fn(ref)
        ref16 = m.resample(ref, m.sr, ENCODER_SR)
        feats, lens = m.extract_content(ref16)
        tokens = m.content_tokens(feats, lens)
        cond = m.regulate(tokens, lens, torch.tensor([mel.size(-1)], device=self.device))
        style = m.compute_style(ref16)
        with self._autocast():
            self.sampler = PrefixCachedSampler(m.cfm, mel, cond, style, self.cfg.diffusion_steps, self.cfg.cfg_rate,
                                               self.cfg.sway_coef)
        self.prompt_frames = mel.size(-1)
        return self

    # ------------------------------------------------------------- process
    def _sync(self):
        synchronize(self.device)  # CUDA and MPS run asynchronously; stage timings need a barrier

    @torch.inference_mode()
    def process_block(self, block_16k):
        """Consume ``block_samples_16k`` input samples (16 kHz), return ``block_samples_out`` at ``output_sr``."""
        if self.sampler is None:
            raise RuntimeError("call set_reference() before process_block()")
        m = self.model
        x = as_float32(block_16k, self.device).reshape(-1)
        if x.numel() != self.block_in:
            raise ValueError(f"expected {self.block_in} samples at 16 kHz, got {x.numel()}")
        stamps = [time.perf_counter()]
        self.in_buffer = torch.cat([self.in_buffer[self.block_in:], x])
        self.blocks_done += 1
        stream_end = self.blocks_done * self.block_out  # absolute output-rate sample index
        buffer_start = (self.blocks_done * self.block_in - self.in_buffer.numel()) / ENCODER_SR

        feats, lens = m.extract_content(self.in_buffer[None])
        self._sync(); stamps.append(time.perf_counter())

        tokens = m.content_tokens(feats, lens)
        g1 = math.ceil(stream_end / m.hop)
        g0 = g1 - self.n_mel
        cond = m.regulate(tokens, lens, torch.tensor([self.n_mel], device=self.device),
                          mel_start=g0 * m.hop / m.sr, content_start=buffer_start)
        self._sync(); stamps.append(time.perf_counter())

        frame_ids = torch.remainder(torch.arange(g0, g1, device=self.device), self.noise_bank.size(1))
        with self._autocast():
            mel = self.sampler.sample(cond, self.noise_bank[:, frame_ids][None]).float()
        self._sync(); stamps.append(time.perf_counter())

        s1 = stream_end - self.right_out
        s0 = s1 - self.emit_out
        pad = self.cfg.vocoder_pad_frames
        m0 = max(0, s0 // m.hop - g0 - pad)
        m1 = min(self.n_mel, -(-s1 // m.hop) - g0 + pad)
        wave = m.vocoder(mel[:, :, m0:m1].float())[0, 0]
        offset = (g0 + m0) * m.hop
        segment = wave[s0 - offset: s1 - offset].float()
        self._sync(); stamps.append(time.perf_counter())

        out = self._sola(segment)
        self._sync(); stamps.append(time.perf_counter())
        self.timings.append(np.diff(stamps))
        return out.cpu().numpy()

    def _sola(self, wave):
        """SOLA crossfade (DDSP-SVC algorithm, as in real-time-gui.py). wave: (crossfade + search + block,)"""
        cf, search, block = self.cf_out, self.search_out, self.block_out
        conv_input = wave[None, None, : cf + search]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, torch.ones(1, 1, cf, device=wave.device)) + 1e-8)
        offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        wave = wave[offset:].clone()
        wave[:cf] = wave[:cf] * self.fade_in + self.sola_buffer * self.fade_out
        self.sola_buffer = wave[block: block + cf].clone()
        return wave[:block]

    @torch.inference_mode()
    def warmup(self, n_blocks=2):
        for _ in range(n_blocks):
            self.process_block(np.zeros(self.block_in, dtype=np.float32))
        self.timings.clear()
        self.reset()

    # --------------------------------------------------------------- report
    @property
    def block_samples_16k(self):
        return self.block_in

    @property
    def block_samples_out(self):
        return self.block_out

    def latency_report(self, device_latency_ms=None):
        block_ms = self.n_block * QUANTUM * 1000
        parts = {
            "block_ms": block_ms,
            "right_context_ms": self.n_right * QUANTUM * 1000,
            "crossfade_ms": self.n_cf * QUANTUM * 1000,
            "sola_search_ms": self.n_search * QUANTUM * 1000,
        }
        report = dict(parts)
        report["algorithmic_ms"] = sum(parts.values())
        report["dit_window_frames"] = self.n_mel
        report["prompt_frames"] = getattr(self, "prompt_frames", None)
        if self.sampler is not None:
            report["prefix_cache_mb"] = self.sampler.cache_bytes() / 2 ** 20
        if self.timings:
            t = np.stack(self.timings) * 1000  # (n, stages) ms
            total = t.sum(1)
            report["compute_mean_ms"] = float(total.mean())
            report["compute_p95_ms"] = float(np.percentile(total, 95))
            report["stage_mean_ms"] = {s: float(v) for s, v in zip(STAGES, t.mean(0))}
            report["realtime_ok"] = bool(report["compute_p95_ms"] < block_ms)
        if device_latency_ms is not None:
            report["device_latency_ms"] = device_latency_ms
            report["estimated_end_to_end_ms"] = report["algorithmic_ms"] + device_latency_ms
        return report
