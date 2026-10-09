"""Device helpers for Seed-VC v3, including Apple Silicon (MPS).

MPS notes that shape the v3 code:
  * float64 tensors cannot live on MPS (there is no CPU fallback for dtypes), so
    float64 maths runs on the CPU and only the result moves to the device.
  * Ops missing from older PyTorch MPS builds can run on the CPU when
    PYTORCH_ENABLE_MPS_FALLBACK=1 is set *before torch is imported*; the v3 entry
    points and ``modules/v3/__init__.py`` set it on macOS (real-time-gui.py does the same).
  * Front-ends with FFT / complex maths (STFT log-mels, Kaldi fbank, sinc
    resampling) run on the CPU when the model is on MPS. They are cheap, and this
    keeps results identical across macOS / PyTorch versions.
  * PyTorch's MPS autocast supports fp16, and bf16 only on macOS >= 14, so fp16
    is the MPS choice and bf16 falls back to fp16 on older macOS.
  * MPS runs asynchronously; timings need ``torch.mps.synchronize()``.
"""
import contextlib
import platform
import sys
import warnings

import numpy as np
import torch

AUTOCAST_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def pick_device(name=None):
    """Explicit name, else CUDA > MPS > CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def frontend_device(device):
    """Where STFT / fbank / resampling front-ends run for a model on ``device``."""
    device = torch.device(device)
    return torch.device("cpu") if device.type == "mps" else device


def half_precision_default(device):
    """Whether the frozen Whisper encoder should run in fp16 on ``device``."""
    return torch.device(device).type in ("cuda", "mps")


def _macos_at_least(major):
    version = platform.mac_ver()[0]
    try:
        return sys.platform == "darwin" and int(version.split(".")[0]) >= major
    except ValueError:
        return False


def _mps_autocast_available():
    check = getattr(torch.amp, "is_autocast_available", None)  # added in recent PyTorch releases
    return bool(check and check("mps"))


def resolve_autocast_dtype(device, precision):
    """Map a precision name to an autocast dtype that ``device`` supports (None = fp32)."""
    if precision not in AUTOCAST_DTYPES:
        raise ValueError(f"precision must be one of {sorted(AUTOCAST_DTYPES)}, got {precision!r}")
    dtype = AUTOCAST_DTYPES[precision]
    if dtype is None:
        return None
    kind = torch.device(device).type
    if kind == "cpu" and dtype is torch.float16:
        warnings.warn("fp16 autocast is not useful on CPU; running the DiT in fp32 (use bf16 on CPU)")
        return None
    if kind == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        warnings.warn("this GPU has no bf16 support; using fp16 autocast")
        return torch.float16
    if kind == "mps":
        if not _mps_autocast_available():
            warnings.warn("this PyTorch build has no MPS autocast; running the DiT in fp32 (upgrade PyTorch)")
            return None
        if dtype is torch.bfloat16 and not _macos_at_least(14):
            warnings.warn("bf16 on MPS needs macOS 14+; using fp16 autocast")
            return torch.float16
    return dtype


def as_float32(x, device):
    """numpy / tensor audio -> float32 tensor on ``device``, casting on the CPU first.

    ``torch.as_tensor(float64_array, dtype=float32, device="mps")`` first materialises a
    float64 tensor, which MPS cannot hold; casting before the transfer avoids that.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
    elif x.dtype != torch.float32:
        x = x.to(dtype=torch.float32) if x.device.type == "cpu" else x.cpu().to(torch.float32)
    return x.to(device)


def autocast(device, dtype):
    """Autocast context for ``dtype``; None is a true no-op (a disabled torch.autocast with an
    unsupported dtype still warns on MPS/CPU, which would fire every streaming block)."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)
