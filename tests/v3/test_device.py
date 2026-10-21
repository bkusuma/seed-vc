"""Device portability checks, with Apple Silicon (MPS) in mind.

Runs on the CPU by default. On a Mac, ``V3_TEST_DEVICE=mps python -m pytest tests/v3``
runs the device-aware tests (this file and the tiny-model tests) on the Metal GPU.
"""
import os
import re
import traceback
import warnings

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import modules.v3.device as dev
from conftest import ATOL, DEVICE, REPO_ROOT, build_tiny_model
from modules.v3.dit import PrefixKVStore
from modules.v3.length_regulator import time_alignment_index

V3_DIR = os.path.join(REPO_ROOT, "modules", "v3")
# Functions allowed to produce float64 tensors: they only ever run on the CPU (see modules/v3/device.py).
CPU_ONLY_FLOAT64 = ("time_alignment_index", "fbank", "get_mel_banks", "_get_waveform_and_window_properties")


def test_float64_only_in_cpu_helpers_static():
    """MPS cannot hold float64. Source-level guard: torch float64 only in the CPU-side alignment helper."""
    offenders = []
    for name in sorted(os.listdir(V3_DIR)):
        if not name.endswith(".py"):
            continue
        for i, line in enumerate(open(os.path.join(V3_DIR, name)), 1):
            code = line.split("#", 1)[0]
            if re.search(r"torch\.float64|torch\.double|\.double\(\)", code) and name != "length_regulator.py":
                offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, offenders


class _Float64Recorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if any(isinstance(t, torch.Tensor) and t.dtype == torch.float64 for t in tree_flatten(out)[0]):
            self.events.append((str(func), [f.name for f in traceback.extract_stack()]))
        return out


@pytest.mark.skipif(DEVICE.type != "cpu", reason="dispatch tracing runs on the CPU reference path")
def test_float64_only_in_cpu_helpers_dynamic():
    """Trace a full offline conversion + streaming blocks: every float64 tensor must come from a
    CPU-only helper, so nothing float64 can reach an MPS device."""
    model = build_tiny_model()
    sr = model.sr
    rng = np.random.default_rng(0)
    src, ref = rng.standard_normal(int(sr * 1.5)) * 0.1, rng.standard_normal(sr) * 0.1
    rec = _Float64Recorder()
    with rec:
        model.convert(src, ref, diffusion_steps=1, seed=0)
        eng = model.streaming_engine(diffusion_steps=1)
        eng.set_reference(ref)
        for _ in range(2):
            eng.process_block(rng.standard_normal(eng.block_samples_16k).astype(np.float32) * 0.1)
    assert any("time_alignment_index" in stack for _, stack in rec.events)  # the tracer is live
    bad = [(op, stack[-6:]) for op, stack in rec.events if not any(f in stack for f in CPU_ONLY_FLOAT64)]
    assert not bad, bad[:3]


def test_time_alignment_index_device_and_precision():
    idx = time_alignment_index(200, 44100 / 256, 50.0, n_in=10_000_000, out_start=36000.0, in_start=35990.0,
                               device=DEVICE)
    assert idx.device.type == DEVICE.type and idx.dtype == torch.long
    m = np.arange(200)
    expected = np.floor(((36000.0 + (m + 0.5) * 256 / 44100) - 35990.0) * 50 + 0.5)  # 10 h into a stream
    assert idx.cpu().numpy().tolist() == expected.astype(int).tolist()


def test_frontend_and_precision_defaults():
    assert dev.frontend_device("mps").type == "cpu"
    assert dev.frontend_device("cpu").type == "cpu"
    assert dev.half_precision_default("mps") and dev.half_precision_default("cuda")
    assert not dev.half_precision_default("cpu")
    assert dev.pick_device("cpu").type == "cpu"


def test_autocast_resolution_rules(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dev.resolve_autocast_dtype("cpu", "fp32") is None
        assert dev.resolve_autocast_dtype("cpu", "fp16") is None  # not useful on CPU
        assert dev.resolve_autocast_dtype("cpu", "bf16") is torch.bfloat16
        monkeypatch.setattr(dev, "_mps_autocast_available", lambda: True)
        monkeypatch.setattr(dev, "_macos_at_least", lambda major: False)
        assert dev.resolve_autocast_dtype("mps", "fp16") is torch.float16
        assert dev.resolve_autocast_dtype("mps", "bf16") is torch.float16  # bf16 needs macOS 14
        monkeypatch.setattr(dev, "_macos_at_least", lambda major: True)
        assert dev.resolve_autocast_dtype("mps", "bf16") is torch.bfloat16
        monkeypatch.setattr(dev, "_mps_autocast_available", lambda: False)
        assert dev.resolve_autocast_dtype("mps", "fp16") is None  # old PyTorch: fp32
    with pytest.raises(ValueError):
        dev.resolve_autocast_dtype("cpu", "int8")


# ------------------------------------------------- runs on V3_TEST_DEVICE (cpu by default)
def test_device_prefix_cache_exact():
    model = build_tiny_model()
    est = model.cfm.estimator
    P, W = 24, 30
    C, D = model.mel_fn.n_mels, model.config.model_params.DiT.content_dim
    g = torch.Generator().manual_seed(0)
    prompt, x = torch.randn(1, C, P, generator=g).to(DEVICE), torch.randn(1, C, W, generator=g).to(DEVICE)
    cond, style = torch.randn(1, P + W, D, generator=g).to(DEVICE), torch.randn(1, 192, generator=g).to(DEVICE)
    t = torch.tensor([0.3], device=DEVICE)
    with torch.inference_mode():
        full = est(torch.cat([torch.zeros_like(prompt), x], -1),
                   torch.cat([prompt, torch.zeros_like(x)], -1), torch.tensor([P + W], device=DEVICE), t, style,
                   cond, prompt_lens=torch.tensor([P], device=DEVICE), attn_mode="prompt_isolated")
        store = PrefixKVStore()
        est.forward_prefix(prompt, cond[:, :P], style, t, store, 0)
        window = est.forward_source(x, cond[:, P:], style, t, store, 0)
    assert window.device.type == DEVICE.type
    assert torch.allclose(full[..., P:], window, atol=ATOL, rtol=1e-3)


def test_device_streaming_reduced_precision():
    model = build_tiny_model()
    precision = "bf16" if DEVICE.type == "cpu" else "fp16"
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(model.sr).astype(np.float32) * 0.1
    blocks = [rng.standard_normal(1600).astype(np.float32) * 0.1 for _ in range(3)]
    outs = {}
    for p in ("fp32", precision):
        eng = model.streaming_engine(precision=p, cfg_rate=0.5)
        eng.set_reference(ref)
        outs[p] = np.concatenate([eng.process_block(b) for b in blocks])
        report = eng.latency_report()
        assert report["compute_mean_ms"] > 0 and set(report["stage_mean_ms"]) >= {"encoder", "dit", "vocoder"}
    assert np.isfinite(outs[precision]).all() and len(outs[precision]) == 3 * eng.block_samples_out
    rel = np.linalg.norm(outs[precision] - outs["fp32"]) / np.linalg.norm(outs["fp32"])
    assert rel < 0.2, rel


def test_device_ctc_backward():
    model = build_tiny_model(with_training_heads=True).train()
    bn = model.bottleneck
    feats = torch.randn(2, 60, model.content_dim, device=DEVICE)
    lens = torch.tensor([60, 45], device=DEVICE)
    out = bn(feats, lens)
    targets = torch.tensor([5, 6, 7, 8, 9, 10, 11], device=DEVICE)
    loss = bn.ctc_loss(out, lens, targets, torch.tensor([4, 3], device=DEVICE))
    loss.backward()
    grad = bn.encoder.input_projection.weight.grad
    assert torch.isfinite(loss) and grad is not None and grad.device.type == DEVICE.type
