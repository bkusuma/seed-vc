import numpy as np
import pytest
import torch
import torchaudio

from modules.audio import mel_spectrogram
from modules.v3.audio import MelSpectrogram, StreamingResampler, resolve_fmax


@pytest.mark.parametrize("sr,n_fft,hop,n_mels,fmax", [(44100, 1024, 256, 128, None), (22050, 1024, 256, 80, 8000)])
def test_mel_matches_v1_function(sr, n_fft, hop, n_mels, fmax):
    torch.manual_seed(0)
    y = torch.rand(2, sr) * 1.6 - 0.8
    ref = mel_spectrogram(y, n_fft, n_mels, sr, hop, n_fft, 0, fmax, center=False)
    ours = MelSpectrogram(sr, n_fft, n_fft, hop, n_mels, 0, fmax)(y)
    assert ours.shape == ref.shape == (2, n_mels, sr // hop)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_mel_num_frames():
    mel = MelSpectrogram(44100, 1024, 1024, 256, 128)
    for n in (4410, 44100, 132300):
        assert mel(torch.zeros(1, n)).shape[-1] == mel.num_frames(n) == n // 256


def test_resolve_fmax():
    assert resolve_fmax(None) is None
    assert resolve_fmax("None") is None
    assert resolve_fmax("null") is None
    assert resolve_fmax(8000) == 8000.0
    assert resolve_fmax("8000") == 8000.0


@pytest.mark.parametrize("orig,new,block", [(44100, 16000, 4410), (44100, 48000, 4410), (48000, 16000, 4800)])
def test_streaming_resampler_matches_offline(orig, new, block):
    torch.manual_seed(0)
    x = torch.randn(block * 12)
    rs = StreamingResampler(orig, new)
    out = torch.cat([rs(x[i:i + block]) for i in range(0, len(x), block)])
    assert out.numel() == len(x) * new // orig
    offline = torchaudio.functional.resample(x, orig, new)
    d = rs.lookahead * new // orig
    # streaming output is the offline output delayed by `d` samples (interior, away from the start-up edge)
    a, b = out[d + 200:], offline[200:out.numel() - d]
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
    assert np.isclose(rs.delay_seconds, rs.lookahead / orig)
