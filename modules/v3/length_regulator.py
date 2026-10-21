"""Time-aligned length regulator for Seed-VC v3.

Maps 50 Hz content (bottleneck indices or continuous features) to the mel frame
rate (44100 / 256 = 172.27 fps). Two changes vs v1 ``InterpolateRegulator``,
both needed for streaming:

  * Alignment by absolute time instead of ``F.interpolate(size=T_mel)``: mel
    frame m (centre (m + 0.5) * hop / sr) takes the content frame whose centre
    (j / content_rate) is nearest. A window cut anywhere in the stream gets the
    same alignment as the offline pass.
  * Frame-local normalisation. v1 uses GroupNorm(1, C), which normalises over the
    *whole sequence*, so a frame's output depended on everything else in the
    window. ``FrameLayerNorm`` has the same parameter shapes, so v1 regulator
    weights still load for warm starts.

Submodule names mirror the v1 regulator (model.*, embedding, content_in_proj,
f0_embedding, f0_mask) for shape-filtered warm starts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.v2.length_regulator import f0_to_coarse  # v1 module imports dac at top level


class FrameLayerNorm(nn.Module):
    """LayerNorm over channels for each frame of a (B, C, T) tensor."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = F.layer_norm(x.transpose(1, 2), (x.size(1),), self.weight, self.bias, self.eps)
        return x.transpose(1, 2)


def time_alignment_index(n_out, out_rate, in_rate, n_in, out_start=0.0, in_start=0.0, time_scale=1.0, device=None):
    """Nearest-in-time source index for each output frame.

    Output frame m is centred at ``out_start + (m + 0.5) / out_rate`` seconds; input
    frame j is centred at ``in_start + j / in_rate``. ``time_scale`` stretches the
    output timeline (length_adjust > 1 => slower speech => time_scale = 1 / adjust).
    """
    # float64 on the CPU: absolute stream times grow without bound, and MPS has no float64.
    m = torch.arange(n_out, dtype=torch.float64)
    t = out_start + (m + 0.5) / out_rate
    j = torch.floor((t * time_scale - in_start) * in_rate + 0.5)
    return j.clamp(0, max(n_in - 1, 0)).long().to(device)


class TimeAlignedRegulator(nn.Module):
    def __init__(
        self,
        channels,
        sampling_ratios=(1, 1, 1, 1),
        is_discrete=True,
        in_channels=None,
        codebook_size=2048,
        out_channels=None,
        f0_condition=False,
        n_f0_bins=256,
        content_rate=50.0,
        mel_rate=44100 / 256,
        f0_rate=100.0,
        norm="frame",
    ):
        super().__init__()
        out_channels = out_channels or channels
        layers = []
        for _ in sampling_ratios:
            layers += [
                nn.Conv1d(channels, channels, 3, 1, 1),
                FrameLayerNorm(channels) if norm == "frame" else nn.GroupNorm(1, channels),
                nn.Mish(),
            ]
        layers.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*layers)
        self.receptive_half_width = len(sampling_ratios)  # frames of context per side (k=3 convs)

        self.is_discrete = is_discrete
        self.embedding = nn.Embedding(codebook_size, channels) if is_discrete else None
        self.content_in_proj = None if is_discrete else nn.Linear(in_channels, channels)
        self.content_rate = float(content_rate)
        self.mel_rate = float(mel_rate)
        self.f0_rate = float(f0_rate)
        self.f0_condition = f0_condition
        if f0_condition:
            self.n_f0_bins = n_f0_bins
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))

    def embed(self, content):
        if self.is_discrete:
            if content.dim() == 3:
                content = content[:, 0]
            return self.embedding(content.long())
        return self.content_in_proj(content)

    def forward(self, content, mel_lens, content_lens=None, f0=None, mel_start=0.0, content_start=0.0,
                f0_start=0.0, time_scale=1.0):
        """
        Args:
            content: (B, Tc) indices or (B, Tc, D) features at ``content_rate``.
            mel_lens: (B,) number of output mel frames per item.
            content_lens: (B,) valid content frames (defaults to Tc).
            f0: optional (B, Tf) Hz at ``f0_rate``.
            mel_start / content_start / f0_start: absolute start times (s) of the
                respective first frames, so windows can be aligned to a stream.
            time_scale: content time per output time (1 / length_adjust).
        Returns:
            cond: (B, T_mel_max, out_channels), zero beyond mel_lens.
            mel_lens
        """
        B = content.size(0)
        device = content.device
        x = self.embed(content)  # (B, Tc, C)
        Tc = x.size(1)
        if content_lens is None:
            content_lens = torch.full((B,), Tc, dtype=torch.long, device=device)
        mel_lens = mel_lens.to(device).long()
        T = int(mel_lens.max())
        idx = torch.stack([
            time_alignment_index(T, self.mel_rate, self.content_rate, int(content_lens[b]),
                                 mel_start, content_start, time_scale, device)
            for b in range(B)
        ])  # (B, T)
        x = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.size(-1)))
        mask = (torch.arange(T, device=device)[None] < mel_lens[:, None]).unsqueeze(1)  # (B, 1, T)
        x = x.transpose(1, 2)  # (B, C, T)
        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.unsqueeze(-1)
            else:
                f0_idx = torch.stack([
                    time_alignment_index(T, self.mel_rate, self.f0_rate, f0.size(1), mel_start, f0_start,
                                         time_scale, device)
                    for _ in range(B)
                ])
                f0_coarse = f0_to_coarse(torch.gather(f0, 1, f0_idx), self.n_f0_bins)
                x = x + self.f0_embedding(f0_coarse.clamp(0, self.n_f0_bins - 1)).transpose(1, 2)
        x = x * mask
        out = self.model(x) * mask
        return out.transpose(1, 2).contiguous(), mel_lens
