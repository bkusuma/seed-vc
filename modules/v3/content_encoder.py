"""Whisper content encoder for Seed-VC v3 (default: openai/whisper-large-v3-turbo).

Only the encoder is used. "Turbo" prunes the *decoder* (32 -> 4 layers); the
encoder is the full large-v3 encoder (32 layers, d_model 1280, 128 mel bins,
~637M params), so it costs ~7x whisper-small's encoder per frame.

Padding: Whisper's encoder was trained on 30 s windows (audio + trailing zero
padding). Feeding a short window *without* padding is out of distribution.
Measured on examples/ (1.5-3 s clips, cosine vs. the 30 s-padded features):
no padding 0.29-0.57, 5 s 0.70-0.79, 10 s 0.87-0.91, 15 s 0.93-0.96,
20 s 0.95-0.97. ``pad_seconds`` therefore defaults to 30 (native regime, same as
v1/HF). Shorter padding is a latency knob; the downstream models must then be
trained with the same value.

Differences from the v1 integration:
  * Own forward that slices the positional table, so any padded length
    <= 30 s works (HF's WhisperEncoder.forward only accepts exactly 3000 frames).
  * Log-mel computed in torch on the model's device (exact parity with
    WhisperFeatureExtractor, covered by tests/v3).
  * Optional layer truncation (``output_layer``) to trade quality for speed.
"""
import inspect

import torch
import torch.nn.functional as F

from modules.v3.device import frontend_device

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
FRAME_RATE = 50.0  # encoder output frames per second (frame j is centred at j / 50 s)
MAX_FRAMES = 1500  # 30 s
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 5


class WhisperContentEncoder(torch.nn.Module):
    def __init__(
        self,
        name="openai/whisper-large-v3-turbo",
        output_layer=None,
        pad_seconds=30.0,
        pretrained=True,
        config=None,
    ):
        """
        Args:
            name: HF repo id (or local path) of a Whisper checkpoint.
            output_layer: number of encoder layers to run (1-based). None = all
                layers + final layer norm (== HF ``last_hidden_state``).
            pad_seconds: zero-pad each window to this many seconds before encoding
                (30 = Whisper's native regime). None = no padding (fastest, out of
                distribution; only use if the downstream models were trained so).
            pretrained: load weights; False builds a randomly initialised
                encoder from ``config`` (dict) or the hub config (tests/dry runs).
        """
        super().__init__()
        from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        if pretrained:
            model = WhisperModel.from_pretrained(name, torch_dtype=torch.float32)
            encoder = model.encoder
            del model
        else:
            whisper_config = WhisperConfig(**config) if config is not None else WhisperConfig.from_pretrained(name)
            encoder = WhisperEncoder(whisper_config)
        n_layers = len(encoder.layers)
        self.output_layer = n_layers if output_layer is None else int(output_layer)
        if not 1 <= self.output_layer <= n_layers:
            raise ValueError(f"output_layer must be in [1, {n_layers}], got {self.output_layer}")
        if self.output_layer < n_layers:
            encoder.layers = encoder.layers[: self.output_layer]
        self.apply_final_norm = self.output_layer == n_layers
        self.encoder = encoder
        self.output_dim = encoder.config.d_model
        self.num_mel_bins = encoder.config.num_mel_bins
        self.pad_seconds = None if pad_seconds is None else float(pad_seconds)
        if self.pad_seconds is not None and not 0 < self.pad_seconds <= CHUNK_SECONDS:
            raise ValueError(f"pad_seconds must be in (0, {CHUNK_SECONDS}] or None, got {pad_seconds}")

        try:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(name) if pretrained else None
        except OSError:
            feature_extractor = None
        if feature_extractor is None or feature_extractor.feature_size != self.num_mel_bins:
            feature_extractor = WhisperFeatureExtractor(feature_size=self.num_mel_bins)
        # (n_freq, n_mels) Slaney filters exactly as used by the HF extractor
        self.register_buffer("mel_filters", torch.from_numpy(feature_extractor.mel_filters).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

        layer_params = inspect.signature(encoder.layers[0].forward).parameters
        self._layer_kwargs = {k: None for k in ("attention_mask", "layer_head_mask") if k in layer_params}

        self.requires_grad_(False)
        self.eval()

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def train(self, mode=True):
        # Frozen feature extractor: always eval (no dropout / layerdrop).
        return super().train(False)

    @staticmethod
    def num_frames(num_samples_16k):
        n_mel = int(num_samples_16k) // HOP_LENGTH
        return (n_mel + 1) // 2

    def log_mel(self, waves_16k):
        """(B, N) -> (B, n_mels, N // 160), identical to WhisperFeatureExtractor's torch path.

        Runs in fp32 (buffers may have been cast by module.half()) and on the CPU when
        the input lives on MPS; the result is returned on the input's device.
        """
        device = waves_16k.device
        fdev = frontend_device(device)
        waves_16k = waves_16k.float().to(fdev)
        if waves_16k.size(-1) < N_FFT:
            waves_16k = F.pad(waves_16k, (0, N_FFT - waves_16k.size(-1)))
        window = self.window.float().to(fdev)
        stft = torch.stft(waves_16k, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.float().to(fdev).T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.amax(dim=(1, 2), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        return ((log_spec + 4.0) / 4.0).to(device)

    def encode_log_mel(self, mel):
        """(B, n_mels, F) -> (B, ceil(F / 2), d_model) without HF's 3000-frame restriction."""
        enc = self.encoder
        x = F.gelu(enc.conv1(mel.to(self.dtype)))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)
        n = x.size(1)
        if n > enc.embed_positions.weight.size(0):
            raise ValueError(f"{n} encoder frames exceed Whisper's 30 s limit; use forward() for long audio")
        x = x + enc.embed_positions.weight[:n].to(x.dtype)
        for layer in enc.layers:
            out = layer(x, **self._layer_kwargs)
            x = out[0] if isinstance(out, (tuple, list)) else out
        if self.apply_final_norm:
            x = enc.layer_norm(x)
        return x

    def _encode_window(self, wave_16k):
        """wave_16k: (N,) with N <= 30 s -> (num_frames(N), d_model) float32."""
        n_frames = self.num_frames(wave_16k.numel())
        if self.pad_seconds is not None:
            target = int(round(SAMPLE_RATE * self.pad_seconds))
            if wave_16k.numel() < target:
                wave_16k = F.pad(wave_16k, (0, target - wave_16k.numel()))
        feats = self.encode_log_mel(self.log_mel(wave_16k[None]))[0]
        return feats[:n_frames].float()

    def _encode_long(self, wave_16k):
        """Chunk >30 s audio with 5 s overlap (same scheme as v1 inference.py)."""
        chunk = SAMPLE_RATE * CHUNK_SECONDS
        overlap = SAMPLE_RATE * OVERLAP_SECONDS
        if wave_16k.numel() <= chunk:
            return self._encode_window(wave_16k)
        pieces = []
        start = 0
        first = True
        while start < wave_16k.numel():
            if first:
                seg = wave_16k[:chunk]
                feats = self._encode_window(seg)
                pieces.append(feats)
                start = chunk
                first = False
            else:
                seg = wave_16k[start - overlap:start - overlap + chunk]
                feats = self._encode_window(seg)
                pieces.append(feats[int(FRAME_RATE * OVERLAP_SECONDS):])
                start += chunk - overlap
        feats = torch.cat(pieces, dim=0)
        return feats[: self.num_frames(wave_16k.numel())]

    @torch.no_grad()
    def forward(self, waves_16k, wave_lens=None):
        """
        Args:
            waves_16k: (B, N) float 16 kHz audio.
            wave_lens: optional (B,) valid lengths; items are encoded one by one
                so padding never influences the features of shorter items.
        Returns:
            feats: (B, T_max, d_model) float32, zero-padded.
            feat_lens: (B,) long.
        """
        if waves_16k.dim() == 1:
            waves_16k = waves_16k[None]
        if wave_lens is None:
            wave_lens = torch.full((waves_16k.size(0),), waves_16k.size(-1), dtype=torch.long)
        outs = [self._encode_long(waves_16k[b, : int(wave_lens[b])]) for b in range(waves_16k.size(0))]
        feat_lens = torch.tensor([o.size(0) for o in outs], dtype=torch.long, device=waves_16k.device)
        feats = torch.nn.utils.rnn.pad_sequence(outs, batch_first=True)
        return feats, feat_lens
