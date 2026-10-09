"""Content bottleneck for Seed-VC v3: compact BSQ codebook + ASR (CTC) loss.

Whisper features carry speaker information. The DiT should only see what is
needed to reproduce *what* was said and *how* (phonetics, prosody), not *who*
said it. Three pressures act on the bottleneck during training (stage A):

  1. Capacity: binary spherical quantization to ``log2(codebook_size)`` bits per
     50 Hz frame (default 11 bits = 550 bps). The DiT consumes only the indices.
  2. CTC ASR loss on the quantized latents keeps linguistic content decodable.
  3. Speaker-conditioned reconstruction: a decoder that is *given* the speaker
     embedding reconstructs the (original-speaker) Whisper features from the
     bottleneck of a timbre-perturbed input, so the bottleneck gains nothing by
     carrying speaker identity but is rewarded for prosody/phonetics.
  (+ a BSQ entropy loss for codebook utilisation.)

Streaming: the encoder uses causal (left-padded) depthwise convs and no GRN
(GRN normalises over the whole time axis, which makes outputs window-dependent),
so index t depends only on features <= t + lookahead.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.astral_quantization.bsq import BinarySphericalQuantize
from modules.astral_quantization.convnext import ConvNextV2LayerNorm


class CausalConvNeXtBlock(nn.Module):
    """ConvNeXt block with configurable right context and frame-local ops only."""

    def __init__(self, dim, intermediate_dim, kernel_size=7, lookahead=0, layer_scale_init=1e-6):
        super().__init__()
        if not 0 <= lookahead <= kernel_size - 1:
            raise ValueError("lookahead must be in [0, kernel_size - 1]")
        self.pad_left = kernel_size - 1 - lookahead
        self.pad_right = lookahead
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim)
        self.norm = ConvNextV2LayerNorm(dim, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(F.pad(x, (self.pad_left, self.pad_right)))
        x = self.norm(x).transpose(1, 2)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = (self.gamma * x).transpose(1, 2)
        return residual + x


class ConvNeXtStack(nn.Module):
    """(B, C_in, T) -> (B, C_out, T); optional global conditioning ``g`` (B, gin, 1)."""

    def __init__(self, dim, intermediate_dim, num_blocks, input_dim=None, output_dim=None,
                 gin_channels=0, kernel_size=7, lookahead=0):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, dim, 1) if input_dim and input_dim != dim else nn.Identity()
        self.gin = nn.Conv1d(gin_channels, dim, 1) if gin_channels > 0 else None
        self.blocks = nn.ModuleList(
            [CausalConvNeXtBlock(dim, intermediate_dim, kernel_size, lookahead) for _ in range(num_blocks)]
        )
        self.final_norm = ConvNextV2LayerNorm(dim, data_format="channels_first")
        self.output_projection = nn.Conv1d(dim, output_dim, 1) if output_dim and output_dim != dim else nn.Identity()
        self.receptive_right = lookahead * num_blocks
        self.receptive_left = (kernel_size - 1 - lookahead) * num_blocks

    def forward(self, x, g=None):
        x = self.input_projection(x)
        if self.gin is not None:
            if g is None:
                raise ValueError("this stack needs global conditioning g")
            x = x + self.gin(g)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(self.final_norm(x))


def bsq_entropy_loss(u, mask=None, inv_temperature=10.0, gamma=1.0):
    """Factorised BSQ entropy loss.

    For codes c in {-1/sqrt(L), +1/sqrt(L)}^L and q(c|u) ~ exp(tau * <c, u>), the
    distribution factorises per bit: p_d(+) = sigmoid(2 * tau * u_d / sqrt(L)).
    Loss = mean per-frame entropy (-> confident codes) - gamma * entropy of the
    batch-average bit distribution (-> balanced, fully used codebook).

    Args:
        u: (B, T, L) L2-normalised pre-quantization latents.
        mask: optional (B, T) bool of valid frames.
    """
    if mask is not None:
        u = u[mask]
    else:
        u = u.reshape(-1, u.size(-1))
    if u.numel() == 0:
        return u.new_zeros(())
    L = u.size(-1)
    p = torch.sigmoid(2.0 * inv_temperature * u.float() / math.sqrt(L))
    p = p.clamp(1e-6, 1 - 1e-6)
    per_frame = -(p * p.log() + (1 - p) * (1 - p).log()).sum(-1).mean()
    p_avg = p.mean(0)
    codebook = -(p_avg * p_avg.log() + (1 - p_avg) * (1 - p_avg).log()).sum()
    return per_frame - gamma * codebook


class ByteTokenizer:
    """UTF-8 byte targets for CTC (language agnostic, 256 symbols + blank)."""

    def __init__(self):
        self.vocab_size = 257  # 256 bytes + blank
        self.blank_id = 256

    @staticmethod
    def normalize(text):
        return " ".join(text.strip().lower().split())

    def encode(self, text):
        return list(self.normalize(text).encode("utf-8"))

    def decode(self, ids):
        return bytes([i for i in ids if 0 <= i < 256]).decode("utf-8", errors="ignore")


class WhisperTextTokenizer:
    """Whisper BPE targets for CTC. Much larger softmax (~51.9k classes at 50 Hz)."""

    def __init__(self, name="openai/whisper-large-v3-turbo"):
        from transformers import WhisperTokenizer

        self.tokenizer = WhisperTokenizer.from_pretrained(name)
        self.blank_id = len(self.tokenizer)
        self.vocab_size = self.blank_id + 1

    def encode(self, text):
        return self.tokenizer.encode(" " + text.strip(), add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode([i for i in ids if i != self.blank_id])


def build_text_tokenizer(kind, encoder_name=None):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "whisper":
        return WhisperTextTokenizer(encoder_name)
    raise ValueError(f"unknown ASR tokenizer '{kind}' (expected 'byte' or 'whisper')")


class ContentBottleneck(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        dim=512,
        intermediate_dim=1536,
        num_blocks=6,
        kernel_size=7,
        lookahead=0,
        codebook_size=2048,
        entropy_inv_temperature=10.0,
        entropy_gamma=1.0,
        asr_vocab_size=257,
        asr_blank_id=256,
        asr_blocks=2,
        recon_blocks=4,
        speaker_dim=192,
        with_training_heads=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.codebook_size = codebook_size
        self.entropy_inv_temperature = entropy_inv_temperature
        self.entropy_gamma = entropy_gamma
        self.encoder = ConvNeXtStack(dim, intermediate_dim, num_blocks, input_dim=input_dim,
                                     kernel_size=kernel_size, lookahead=lookahead)
        # Entropy loss is computed here (bsq_entropy_loss); the module's built-in
        # soft entropy loss is applied to a 4-D tensor it treats as 2-D, so it is disabled.
        self.quantizer = BinarySphericalQuantize(
            dim=dim, codebook_size=codebook_size, entropy_loss_weight=0.0, spherical=True,
            enable_entropy_loss=False, soft_entropy_loss=False,
        )
        self.asr_blank_id = asr_blank_id
        self.asr_vocab_size = asr_vocab_size
        if with_training_heads:
            # Training-only heads may look ahead (symmetric context).
            sym = (kernel_size - 1) // 2
            self.asr_head = ConvNeXtStack(dim, intermediate_dim, asr_blocks, output_dim=asr_vocab_size,
                                          kernel_size=kernel_size, lookahead=sym)
            self.recon_decoder = ConvNeXtStack(dim, intermediate_dim, recon_blocks, output_dim=input_dim,
                                               gin_channels=speaker_dim, kernel_size=kernel_size, lookahead=sym)
        else:
            self.asr_head = None
            self.recon_decoder = None

    @property
    def bits(self):
        return int(math.log2(self.codebook_size))

    @property
    def receptive_right(self):
        """Future encoder frames that influence an index (0 for the causal default)."""
        return self.encoder.receptive_right

    def strip_training_heads(self):
        self.asr_head = None
        self.recon_decoder = None
        return self

    def forward(self, feats, feat_lens=None):
        """
        Args:
            feats: (B, T, input_dim) encoder features.
            feat_lens: optional (B,) valid lengths.
        Returns dict with
            latent:    (B, T, dim) pre-quantization latents
            quantized: (B, T, dim) straight-through quantized latents
            indices:   (B, T) long in [0, codebook_size)
            u:         (B, T, bits) L2-normalised projection (for the entropy loss)
            mask:      (B, T) bool valid-frame mask
        """
        B, T, _ = feats.shape
        if feat_lens is None:
            feat_lens = torch.full((B,), T, dtype=torch.long, device=feats.device)
        mask = torch.arange(T, device=feats.device)[None] < feat_lens[:, None]
        latent = self.encoder(feats.transpose(1, 2)).transpose(1, 2)
        latent = latent * mask[..., None]
        quantized, indices, _ = self.quantizer(latent)
        u = F.normalize(self.quantizer.project_in(latent.float()), dim=-1)
        return {"latent": latent, "quantized": quantized, "indices": indices.long(), "u": u, "mask": mask}

    @torch.no_grad()
    def encode(self, feats, feat_lens=None):
        out = self.forward(feats, feat_lens)
        return out["indices"], out["quantized"]

    def entropy_loss(self, out):
        return bsq_entropy_loss(out["u"], out["mask"], self.entropy_inv_temperature, self.entropy_gamma)

    def ctc_loss(self, out, feat_lens, targets, target_lens):
        """CTC on quantized latents. targets: (B, S) padded with anything; target_lens: (B,)."""
        if self.asr_head is None:
            raise RuntimeError("ASR head was stripped; ctc_loss is training-only")
        logits = self.asr_head(out["quantized"].transpose(1, 2)).transpose(1, 2)  # (B, T, V)
        log_probs = logits.float().log_softmax(-1).transpose(0, 1)  # (T, B, V)
        return F.ctc_loss(log_probs, targets, feat_lens, target_lens, blank=self.asr_blank_id,
                          reduction="mean", zero_infinity=True)

    def recon_loss(self, out, speaker_emb, target_feats, mask=None):
        """L1 between decoded features (given the speaker) and layer-normed target features."""
        if self.recon_decoder is None:
            raise RuntimeError("reconstruction decoder was stripped; recon_loss is training-only")
        pred = self.recon_decoder(out["quantized"].transpose(1, 2), g=speaker_emb[:, :, None]).transpose(1, 2)
        target = F.layer_norm(target_feats.float(), (target_feats.size(-1),))
        mask = out["mask"] if mask is None else mask
        err = (pred.float() - target).abs().mean(-1)
        return (err * mask).sum() / mask.sum().clamp(min=1)

    def ctc_greedy_decode(self, out, feat_lens):
        """Greedy CTC decoding for monitoring (returns lists of token ids)."""
        logits = self.asr_head(out["quantized"].transpose(1, 2)).transpose(1, 2)
        best = logits.argmax(-1)
        results = []
        for b in range(best.size(0)):
            seq = best[b, : int(feat_lens[b])].tolist()
            ids, prev = [], None
            for s in seq:
                if s != prev and s != self.asr_blank_id:
                    ids.append(s)
                prev = s
            results.append(ids)
        return results
