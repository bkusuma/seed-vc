import os

import librosa
import numpy as np
import pytest
import torch

from conftest import REPO_ROOT, TINY_WHISPER, build_tiny_model, slow, tiny_config, weights_cached
from modules.commons import recursive_munch
from modules.flow_matching import CFM
from modules.v3.content_encoder import WhisperContentEncoder
from modules.v3.model import SeedVCv3
from modules.v3.realtime_io import AutomaticGainControl, EchoCanceller, RealtimeVCStream, simulate_stream


# ----------------------------------------------------------------- encoder
def test_encoder_frames_and_padding():
    enc = WhisperContentEncoder("tiny", pretrained=False, config=TINY_WHISPER)
    for n in (16000, 16000 * 3 + 123, 16000 * 35):  # last one exercises 30 s chunking
        feats, lens = enc(torch.randn(1, n) * 0.1)
        assert feats.shape == (1, WhisperContentEncoder.num_frames(n), 64) and int(lens[0]) == feats.size(1)
    wave = torch.randn(1, 32000) * 0.1
    enc.pad_seconds = None
    unpadded, _ = enc(wave)
    enc.pad_seconds = 30.0
    padded, _ = enc(wave)
    assert unpadded.shape == padded.shape and not torch.allclose(unpadded, padded)


def test_encoder_batch_items_independent():
    enc = WhisperContentEncoder("tiny", pretrained=False, config=TINY_WHISPER)
    a, b = torch.randn(16000) * 0.1, torch.randn(24000) * 0.1
    batch = torch.zeros(2, 24000)
    batch[0, :16000], batch[1] = a, b
    feats, lens = enc(batch, torch.tensor([16000, 24000]))
    single, _ = enc(a[None])
    assert lens.tolist() == [50, 75] and torch.allclose(feats[0, :50], single[0], atol=1e-5)


@slow
@pytest.mark.skipif(not weights_cached("openai/whisper-large-v3-turbo"), reason="weights not cached")
def test_encoder_matches_hf_exactly():
    from transformers import WhisperFeatureExtractor

    enc = WhisperContentEncoder("openai/whisper-large-v3-turbo")
    wave = librosa.load(os.path.join(REPO_ROOT, "examples/source/source_s1.wav"), sr=16000)[0][:16000 * 4]
    hf = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")([wave], sampling_rate=16000,
                                                                                 return_tensors="pt").input_features
    padded = torch.nn.functional.pad(torch.from_numpy(wave), (0, 480000 - len(wave)))
    assert torch.equal(enc.log_mel(padded[None]), hf)
    with torch.no_grad():
        ref = enc.encoder(hf).last_hidden_state[:, :200]
    ours, _ = enc(torch.from_numpy(wave)[None])
    assert torch.allclose(ours, ref, atol=1e-5)


# ------------------------------------------------------------------- model
def test_config_validation():
    with pytest.raises(ValueError, match="content_dim"):
        SeedVCv3(recursive_munch(tiny_config(DiT__content_dim=40)), load_content_encoder=False,
                 load_style_encoder=False, load_vocoder=False)
    with pytest.raises(ValueError, match="in_channels"):
        SeedVCv3(recursive_munch(tiny_config(DiT__in_channels=80)), load_content_encoder=False,
                 load_style_encoder=False, load_vocoder=False)


def test_offline_convert_shapes(tiny_model):
    sr = tiny_model.sr
    src, ref = np.random.randn(int(sr * 2.0)) * 0.1, np.random.randn(int(sr * 1.5)) * 0.1
    wave, info = tiny_model.convert(src, ref, diffusion_steps=2, seed=0)
    assert abs(len(wave) - len(src)) < tiny_model.hop
    assert info["content_tokens"] == (1, 100) and info["prompt_mel"][1] == 32
    wave2, _ = tiny_model.convert(src, ref, diffusion_steps=2, length_adjust=1.25, seed=0)
    assert abs(len(wave2) - 1.25 * len(src)) < 2 * tiny_model.hop


def test_long_source_is_chunked(tiny_model):
    sr = tiny_model.sr
    src, ref = np.random.randn(int(sr * 7.0)) * 0.1, np.random.randn(int(sr * 1.0)) * 0.1
    wave, _ = tiny_model.convert(src, ref, diffusion_steps=1, max_context_seconds=3.0, seed=0)
    assert abs(len(wave) - len(src)) < tiny_model.hop


def test_streaming_engine_lengths_and_latency(tiny_model):
    eng = tiny_model.streaming_engine()
    eng.set_reference(np.random.randn(tiny_model.sr) * 0.1)
    blocks = [eng.process_block(np.random.randn(eng.block_samples_16k).astype(np.float32) * 0.1) for _ in range(6)]
    assert all(len(b) == eng.block_samples_out for b in blocks)
    report = eng.latency_report(device_latency_ms=50.0)
    assert report["algorithmic_ms"] == pytest.approx(100 + 20 + 20 + 20)
    assert report["estimated_end_to_end_ms"] == pytest.approx(report["algorithmic_ms"] + 50.0)
    assert set(report["stage_mean_ms"]) == {"encoder", "bottleneck_regulator", "dit", "vocoder", "sola"}


def test_streaming_engine_rejects_window_beyond_padding(tiny_model):
    with pytest.raises(ValueError, match="padded length"):
        tiny_model.streaming_engine(ce_left_context=40.0)


def test_checkpoint_roundtrip(tmp_path, tiny_model):
    path = tmp_path / "v3.pth"
    tiny_model.save_checkpoint(path, stage="test")
    other = build_tiny_model()
    for p in other.cfm.parameters():
        torch.nn.init.zeros_(p)
    report = other.load_checkpoint(path)
    assert all(r["loaded_tensors"] == r["total_tensors"] for r in report.values())
    for key in ("cfm", "length_regulator", "bottleneck"):
        a, b = getattr(tiny_model, key).state_dict(), getattr(other, key).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_warm_start_from_v1_checkpoint(tmp_path, tiny_model):
    v1 = CFM(recursive_munch(tiny_config()["model_params"]))
    path = tmp_path / "v1.pth"
    torch.save({"net": {"cfm": {"module." + k: v for k, v in v1.state_dict().items()}}}, path)
    model = build_tiny_model()
    report = model.warm_start_from_v1(path)["cfm"]
    assert report["loaded_params"] == report["total_params"] and not report["shape_mismatch"]
    assert all(torch.equal(v1.state_dict()[k], model.cfm.state_dict()[k]) for k in v1.state_dict())


# ------------------------------------------------------------- real-time io
def _speech(path, sr, n):
    x = librosa.load(os.path.join(REPO_ROOT, path), sr=sr)[0][:n]
    return x / np.abs(x).max() * 0.5


def test_echo_canceller_erle_and_double_talk():
    sr, n = 44100, 44100 * 16
    far, near = _speech("examples/reference/s3p2.wav", sr, n), _speech("examples/source/source_s3.wav", sr, n)
    n = min(len(far), len(near)) // 441 * 441
    far, near = far[:n], near[:n]
    rng = np.random.default_rng(0)
    tail = int(0.12 * sr)
    h = rng.standard_normal(tail) * np.exp(-6.9 * np.arange(tail) / sr / 0.15)
    h = np.concatenate([np.zeros(int(0.03 * sr)), h / np.sqrt(np.sum(h ** 2)) * 10 ** (-6 / 20)])
    echo = np.convolve(far, h)[:n]
    dt = sr * 10
    near_sig = np.zeros(n)
    near_sig[dt:] = near[dt:]
    mic = echo + near_sig + rng.standard_normal(n) * 10 ** (-65 / 20)
    aec = EchoCanceller(sr)
    out = np.concatenate([aec.process(mic[i:i + 4410], far[i:i + 4410]) for i in range(0, n, 4410)])
    erle = 10 * np.log10(np.mean(mic[4 * sr:dt] ** 2) / np.mean(out[4 * sr:dt] ** 2))
    near_snr = 10 * np.log10(np.mean(near_sig[dt:] ** 2) / np.mean((out[dt:] - near_sig[dt:]) ** 2))
    assert erle > 18.0, erle
    assert near_snr > 18.0, near_snr


def test_agc_levels_gate_and_limiter():
    sr = 44100
    speech = _speech("examples/source/source_s1.wav", sr, sr * 8)
    speech = speech[: len(speech) // 441 * 441]
    quiet = speech / np.sqrt(np.mean(speech ** 2)) * 10 ** (-40 / 20)  # -40 dBFS RMS
    agc = AutomaticGainControl(sr, target_dbfs=-20.0)
    out = agc.process(quiet)
    active = np.abs(quiet[4 * sr:]) > 0
    level = 10 * np.log10(np.mean(out[4 * sr:][active] ** 2) + 1e-12)
    assert -26.0 < level < -14.0, level
    assert np.abs(out).max() <= 1.0
    noise = np.random.default_rng(0).standard_normal(sr * 2) * 10 ** (-70 / 20)
    agc_n = AutomaticGainControl(sr)
    assert np.sqrt(np.mean(agc_n.process(noise) ** 2)) < 10 ** (-70 / 20)  # gate attenuates, never boosts noise
    loud = np.clip(speech * 4.0, -1.5, 1.5)
    assert np.abs(AutomaticGainControl(sr, max_gain_db=12).process(loud)).max() <= 1.0


@pytest.mark.parametrize("device_sr", [22050, 48000])
def test_realtime_stream_chain(tiny_model, device_sr):
    eng = tiny_model.streaming_engine()
    eng.set_reference(np.random.randn(tiny_model.sr) * 0.1)
    stream = RealtimeVCStream(eng, samplerate=device_sr)
    x = np.random.randn(stream.block * 5).astype(np.float32) * 0.05
    out, times = simulate_stream(stream, x, echo_ir=np.r_[np.zeros(100), 0.3])
    assert len(out) == len(x) and len(times) == 5 and np.isfinite(out).all()


def test_streaming_bf16_autocast_close_to_fp32(tiny_model):
    ref = np.random.default_rng(0).standard_normal(tiny_model.sr).astype(np.float32) * 0.1
    block = np.random.default_rng(1).standard_normal(1600).astype(np.float32) * 0.1
    outs = {}
    for precision in ("fp32", "bf16"):
        eng = tiny_model.streaming_engine(precision=precision, cfg_rate=0.0)
        eng.set_reference(ref)
        outs[precision] = np.concatenate([eng.process_block(block) for _ in range(3)])
    assert np.isfinite(outs["bf16"]).all()
    rel = np.linalg.norm(outs["bf16"] - outs["fp32"]) / np.linalg.norm(outs["fp32"])
    assert rel < 0.2, rel
    with pytest.raises(ValueError, match="precision"):
        tiny_model.streaming_engine(precision="int8")
