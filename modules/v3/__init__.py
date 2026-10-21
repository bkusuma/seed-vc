"""Seed-VC v3: 44.1 kHz BigVGAN-v2 synthesis, Whisper-large-v3-turbo content encoder,
BSQ content bottleneck with ASR (CTC) loss, and a cached-prefix streaming DiT.

See v3_upgrade_plan.md for the design and its trade-offs.
"""
import os as _os
import sys as _sys

if _sys.platform == "darwin" and "torch" not in _sys.modules:
    # only effective before torch is imported; entry points set it themselves
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from modules.v3.model import SeedVCv3, load_v3_config
from modules.v3.streaming import StreamingConfig, StreamingVoiceConverter

__all__ = ["SeedVCv3", "load_v3_config", "StreamingConfig", "StreamingVoiceConverter"]
