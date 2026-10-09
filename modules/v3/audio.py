"""Audio front-end utilities for Seed-VC v3.

The mel front-end matches ``modules.audio.mel_spectrogram`` (and BigVGAN's
``meldataset.mel_spectrogram``) numerically, but is a module with its own
buffers instead of a process-global cache. The v1 global cache is keyed only by
(sampling_rate, fmax, device), so two configs that share those but differ in
n_fft / n_mels (e.g. the v1 44k preset and v3) would silently reuse the wrong
mel basis inside one process.
"""
import math
from fractions import Fraction

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn


def resolve_fmax(value):
    """Map YAML ``null`` / the strings "None" / "null" to None, anything else to float."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("none", "null", ""):
            return None
        return float(value)
    return float(value)


class MelSpectrogram(torch.nn.Module):
    """Log-mel spectrogram, BigVGAN convention (reflect pad (n_fft-hop)/2, center=False, log(clamp(1e-5)))."""

    def __init__(self, sampling_rate, n_fft, win_length, hop_length, n_mels, fmin=0, fmax=None):
        super().__init__()
        self.sampling_rate = int(sampling_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = resolve_fmax(fmax)
        mel = librosa_mel_fn(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    @property
    def frame_rate(self):
        return self.sampling_rate / self.hop_length

    def num_frames(self, num_samples):
        """Number of mel frames produced for ``num_samples`` input samples (== num_samples // hop)."""
        return int(num_samples) // self.hop_length

    def forward(self, y):
        """y: (B, N) or (N,) float waveform in [-1, 1] -> (B, n_mels, N // hop)."""
        if y.dim() == 1:
            y = y.unsqueeze(0)
        y = y.float()
        pad = (self.n_fft - self.hop_length) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.float(),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        spec = torch.matmul(self.mel_basis.float(), spec)
        return torch.log(torch.clamp(spec, min=1e-5))


class StreamingResampler:
    """Exact block-wise resampling with a fixed lookahead delay.

    Each call consumes ``n`` input samples and returns ``n * new_sr / orig_sr``
    output samples. Output is delayed by ``lookahead`` input samples so every
    returned sample sees the full sinc kernel, which makes the concatenated
    output equal to resampling the whole signal at once (up to float error).
    Block lengths must be multiples of ``orig_sr / gcd(orig_sr, new_sr)``.
    """

    def __init__(self, orig_sr, new_sr, lookahead=None, device="cpu"):
        self.orig_sr = int(orig_sr)
        self.new_sr = int(new_sr)
        ratio = Fraction(self.new_sr, self.orig_sr)
        self.in_unit = ratio.denominator  # input samples per output group
        self.out_unit = ratio.numerator
        if lookahead is None:
            # torchaudio's default kernel half-width is ~6 zero crossings of the lower rate.
            half_width = math.ceil(6 * self.orig_sr / min(self.orig_sr, self.new_sr) / 0.99) + 1
            lookahead = half_width
        self.lookahead = int(math.ceil(lookahead / self.in_unit) * self.in_unit)
        self.history = self.lookahead  # left context, same size as lookahead
        self.device = device
        self.buffer = torch.zeros(self.history + self.lookahead, device=device)

    @property
    def delay_seconds(self):
        return self.lookahead / self.orig_sr

    def reset(self):
        self.buffer.zero_()

    def __call__(self, block):
        if self.orig_sr == self.new_sr:
            return block
        block = torch.as_tensor(block, dtype=torch.float32, device=self.device)
        n = block.numel()
        if n % self.in_unit != 0:
            raise ValueError(f"block length {n} must be a multiple of {self.in_unit} for {self.orig_sr}->{self.new_sr} Hz")
        self.buffer = torch.cat([self.buffer, block])
        # buffer = [history | n samples to emit | lookahead]
        out = torchaudio.functional.resample(self.buffer, self.orig_sr, self.new_sr)
        start = self.history * self.out_unit // self.in_unit
        n_out = n * self.out_unit // self.in_unit
        result = out[start:start + n_out]
        self.buffer = self.buffer[-(self.history + self.lookahead):]
        return result
