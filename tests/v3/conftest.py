import copy
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(REPO_ROOT, "checkpoints", "hf_cache"))

from modules.commons import recursive_munch  # noqa: E402

# Run the device-aware tests on another backend with e.g. `V3_TEST_DEVICE=mps python -m pytest tests/v3`.
DEVICE = torch.device(os.environ.get("V3_TEST_DEVICE", "cpu"))
# Accelerators may reorder / fuse fp32 reductions (TF32 on CUDA, MPS kernels): looser equality there.
ATOL = 1e-5 if DEVICE.type == "cpu" else 2e-3

TINY_WHISPER = dict(d_model=64, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=128, num_mel_bins=80,
                    max_source_positions=1500, decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=128,
                    vocab_size=128, max_target_positions=64)

_TINY_CONFIG = {
    "version": "3.0",
    "log_dir": "./runs",
    "preprocess_params": {
        "sr": 22050,
        "spect_params": {"n_fft": 1024, "win_length": 1024, "hop_length": 256, "n_mels": 32, "fmin": 0, "fmax": None},
    },
    "model_params": {
        "dit_type": "DiT",
        "reg_loss_type": "l1",
        "content_encoder": {"type": "whisper", "name": "tiny-random", "output_layer": None, "pad_seconds": 30},
        "bottleneck": {"enabled": True, "dim": 32, "intermediate_dim": 64, "num_blocks": 2, "kernel_size": 7,
                       "lookahead": 0, "codebook_size": 64, "asr_tokenizer": "byte", "asr_blocks": 1,
                       "recon_blocks": 1},
        "length_regulator": {"channels": 48, "sampling_ratios": [1, 1], "norm": "frame", "f0_condition": False,
                             "n_f0_bins": 256},
        "style_encoder": {"dim": 192, "campplus_path": os.path.join(REPO_ROOT, "campplus_cn_common.bin")},
        "vocoder": {"type": "bigvgan", "name": "unused-in-tests"},
        "DiT": {"hidden_dim": 48, "num_heads": 4, "depth": 3, "class_dropout_prob": 0.1, "block_size": 8192,
                "in_channels": 32, "style_condition": True, "final_layer_type": "mlp", "target": "mel",
                "content_dim": 48, "content_codebook_size": 1024, "content_type": "discrete", "f0_condition": False,
                "n_f0_bins": 256, "content_codebooks": 1, "is_causal": False, "long_skip_connection": False,
                "zero_prompt_speech_token": False, "time_as_token": False, "style_as_token": False,
                "uvit_skip_connection": True, "add_resblock_in_transformer": False},
        "cfm": {"attn_mode_probs": {"full": 0.5, "prompt_isolated": 0.5}, "crop_prob": 0.5, "crop_min_frames": 8,
                "crop_max_frames": 32},
    },
    "loss_params": {"base_lr": 1e-3, "lambda_ctc": 1.0, "lambda_recon": 1.0, "lambda_entropy": 0.1},
    "training": {"batch_size": 2, "min_duration": 0.5, "max_duration": 30.0, "perturb": "none"},
    "inference": {"diffusion_steps": 3, "inference_cfg_rate": 0.5, "sway_coef": -1.0, "attn_mode": "full",
                  "max_prompt_seconds": 5.0},
    "streaming": {"block_time": 0.1, "crossfade_time": 0.02, "ce_left_context": 1.0, "dit_left_context": 0.2,
                  "right_context": 0.02, "diffusion_steps": 3, "cfg_rate": 0.5, "sway_coef": -1.0,
                  "prompt_max_time": 1.0, "vocoder_pad_frames": 4},
}


def tiny_config(**model_overrides):
    cfg = copy.deepcopy(_TINY_CONFIG)
    for path, value in model_overrides.items():
        node = cfg["model_params"]
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return cfg


class FakeVocoder(torch.nn.Module):
    """Frame-local mel -> wave (B, 1, T * hop); stands in for BigVGAN in unit tests."""

    def __init__(self, n_mels, hop):
        super().__init__()
        self.up = torch.nn.ConvTranspose1d(n_mels, 1, kernel_size=hop, stride=hop)

    def forward(self, mel):
        return torch.tanh(self.up(mel))


def build_tiny_model(config=None, with_training_heads=False, device=None):
    from modules.v3.model import SeedVCv3

    cfg = recursive_munch(config or tiny_config())
    torch.manual_seed(0)
    model = SeedVCv3(cfg, encoder_pretrained=False, encoder_config=TINY_WHISPER, load_vocoder=False,
                     with_training_heads=with_training_heads)
    model.vocoder = FakeVocoder(cfg.preprocess_params.spect_params.n_mels, cfg.preprocess_params.spect_params.hop_length)
    return model.prepare(device or DEVICE).eval()


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny_model()


def weights_cached(repo_id):
    cache = os.environ["HF_HUB_CACHE"]
    return os.path.isdir(os.path.join(cache, "models--" + repo_id.replace("/", "--")))


slow = pytest.mark.skipif(os.environ.get("SEEDVC_V3_SLOW") != "1",
                          reason="set SEEDVC_V3_SLOW=1 to run tests that load real (cached) weights")
