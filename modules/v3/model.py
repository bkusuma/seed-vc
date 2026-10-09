"""Seed-VC v3 model container: builder, checkpoint I/O, v1 warm start, offline conversion."""
import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from munch import Munch

from modules.commons import recursive_munch
from modules.v3.audio import MelSpectrogram, resolve_fmax
from modules.v3.bottleneck import ContentBottleneck, build_text_tokenizer
from modules.v3.content_encoder import FRAME_RATE as CONTENT_RATE
from modules.v3.content_encoder import SAMPLE_RATE as ENCODER_SR
from modules.v3.content_encoder import WhisperContentEncoder
from modules.v3.dit import ATTN_FULL
from modules.v3.flow_matching import CFMv3
from modules.v3.length_regulator import TimeAlignedRegulator

VERSION = "3.0"
TRAINABLE_KEYS = ("bottleneck", "length_regulator", "cfm")
# Regulator parameters that v1 never trained in continuous-input mode (or whose input differs).
_V1_LR_SKIP_PREFIXES = ("embedding.", "mask_token", "content_in_proj.")


def load_v3_config(config):
    """Path / dict / Munch -> Munch."""
    if isinstance(config, Munch):
        return config
    if isinstance(config, (str, os.PathLike)):
        with open(config) as f:
            config = yaml.safe_load(f)
    return recursive_munch(config)


def _get(node, key, default=None):
    return node.get(key, default) if node is not None else default


def crossfade(chunk1, chunk2, overlap):
    """Equal-power crossfade, same as v1 inference.py."""
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


class SeedVCv3(nn.Module):
    def __init__(self, config, load_content_encoder=True, encoder_pretrained=True, encoder_config=None,
                 load_style_encoder=True, load_vocoder=True, with_training_heads=False):
        """
        Args:
            config: path / dict / Munch following configs/v3/*.yml.
            load_content_encoder / encoder_pretrained / encoder_config: build the
                Whisper encoder; ``encoder_config`` (dict) builds a random encoder
                from a custom WhisperConfig (tests).
            load_style_encoder / load_vocoder: skip for training or unit tests.
            with_training_heads: build the bottleneck's CTC / reconstruction heads.
        """
        super().__init__()
        self.config = config = load_v3_config(config)
        pp, mp = config.preprocess_params, config.model_params
        sp = pp.spect_params
        self.sr = int(pp.sr)
        self.hop = int(sp.hop_length)
        self.mel_fn = MelSpectrogram(self.sr, sp.n_fft, sp.win_length, sp.hop_length, sp.n_mels,
                                     _get(sp, "fmin", 0), _get(sp, "fmax"))

        ce = mp.content_encoder
        if _get(ce, "type", "whisper") != "whisper":
            raise ValueError("v3 supports content_encoder.type == 'whisper'")
        if load_content_encoder:
            self.content_encoder = WhisperContentEncoder(ce.name, _get(ce, "output_layer"), _get(ce, "pad_seconds", 30.0),
                                                         pretrained=encoder_pretrained, config=encoder_config)
            self.content_dim = self.content_encoder.output_dim
        else:
            self.content_encoder = None
            self.content_dim = int(_get(ce, "dim", 1280))

        bn = _get(mp, "bottleneck") or Munch()
        self.text_tokenizer = None
        if _get(bn, "enabled", True):
            if with_training_heads:
                self.text_tokenizer = build_text_tokenizer(_get(bn, "asr_tokenizer", "byte"), ce.name)
                vocab, blank = self.text_tokenizer.vocab_size, self.text_tokenizer.blank_id
            else:
                vocab, blank = 257, 256
            self.bottleneck = ContentBottleneck(
                input_dim=self.content_dim, dim=_get(bn, "dim", 512), intermediate_dim=_get(bn, "intermediate_dim", 1536),
                num_blocks=_get(bn, "num_blocks", 6), kernel_size=_get(bn, "kernel_size", 7),
                lookahead=_get(bn, "lookahead", 0), codebook_size=_get(bn, "codebook_size", 2048),
                entropy_inv_temperature=_get(bn, "entropy_inv_temperature", 10.0),
                entropy_gamma=_get(bn, "entropy_gamma", 1.0), asr_vocab_size=vocab, asr_blank_id=blank,
                asr_blocks=_get(bn, "asr_blocks", 2), recon_blocks=_get(bn, "recon_blocks", 4),
                speaker_dim=int(mp.style_encoder.dim), with_training_heads=with_training_heads,
            )
            discrete, codebook = True, int(_get(bn, "codebook_size", 2048))
        else:
            self.bottleneck = None
            discrete, codebook = False, 1

        lr = mp.length_regulator
        self.length_regulator = TimeAlignedRegulator(
            channels=int(lr.channels), sampling_ratios=tuple(_get(lr, "sampling_ratios", (1, 1, 1, 1))),
            is_discrete=discrete, in_channels=self.content_dim, codebook_size=codebook,
            f0_condition=bool(_get(lr, "f0_condition", False)), n_f0_bins=int(_get(lr, "n_f0_bins", 256)),
            content_rate=CONTENT_RATE, mel_rate=self.sr / self.hop, norm=_get(lr, "norm", "frame"),
        )

        self._validate_dims()
        cf = _get(mp, "cfm") or Munch()
        self.cfm = CFMv3(mp, attn_mode_probs=_get(cf, "attn_mode_probs"), crop_prob=_get(cf, "crop_prob", 0.3),
                         crop_min_frames=_get(cf, "crop_min_frames", 32), crop_max_frames=_get(cf, "crop_max_frames", 344))
        self.style_encoder = self._build_style_encoder() if load_style_encoder else None
        self.vocoder = self._build_vocoder() if load_vocoder else None

    # ------------------------------------------------------------------ build
    def _validate_dims(self):
        mp, sp = self.config.model_params, self.config.preprocess_params.spect_params
        problems = []
        if int(mp.DiT.in_channels) != int(sp.n_mels):
            problems.append(f"DiT.in_channels ({mp.DiT.in_channels}) != n_mels ({sp.n_mels})")
        if int(mp.DiT.content_dim) != int(mp.length_regulator.channels):
            problems.append(f"DiT.content_dim ({mp.DiT.content_dim}) != length_regulator.channels ({mp.length_regulator.channels})")
        if int(mp.style_encoder.dim) != 192:
            problems.append(f"style_encoder.dim must be 192 for CAMPPlus, got {mp.style_encoder.dim}")
        if mp.DiT.final_layer_type == "wavenet" and int(mp.wavenet.hidden_dim) != int(mp.DiT.hidden_dim):
            problems.append("wavenet.hidden_dim must equal DiT.hidden_dim (v1 FinalLayer is conditioned on the DiT time embedding)")
        if problems:
            raise ValueError("Inconsistent v3 config:\n  " + "\n  ".join(problems))

    def _build_style_encoder(self):
        from modules.campplus.DTDNN import CAMPPlus

        model = CAMPPlus(feat_dim=80, embedding_size=192)
        path = _get(self.config.model_params.style_encoder, "campplus_path", "campplus_cn_common.bin")
        if not os.path.isfile(path):
            from hf_utils import load_custom_model_from_hf

            path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval().requires_grad_(False)
        return model

    def _build_vocoder(self):
        voc = self.config.model_params.vocoder
        if _get(voc, "type", "bigvgan") != "bigvgan":
            raise ValueError("v3 supports vocoder.type == 'bigvgan'")
        from modules.bigvgan import bigvgan

        model = bigvgan.BigVGAN.from_pretrained(voc.name, use_cuda_kernel=bool(_get(voc, "use_cuda_kernel", False)))
        sp = self.config.preprocess_params.spect_params
        h = model.h
        expected = {"num_mels": int(sp.n_mels), "hop_size": self.hop, "sampling_rate": self.sr,
                    "n_fft": int(sp.n_fft), "win_size": int(sp.win_length)}
        mismatched = {k: (h[k], v) for k, v in expected.items() if int(h[k]) != v}
        if resolve_fmax(h.get("fmax")) != resolve_fmax(_get(sp, "fmax")):
            mismatched["fmax"] = (h.get("fmax"), _get(sp, "fmax"))
        if mismatched:
            raise ValueError(f"vocoder {voc.name} does not match the mel front-end: {mismatched} (vocoder, config)")
        model.remove_weight_norm()
        model.eval().requires_grad_(False)
        return model

    # ------------------------------------------------------------ utilities
    @property
    def device(self):
        return self.cfm.estimator.input_pos.device

    def prepare(self, device, half_encoder=None):
        """Move to ``device``; run the Whisper encoder in fp16 on CUDA by default."""
        device = torch.device(device)
        self.to(device)
        if self.content_encoder is not None:
            half = device.type == "cuda" if half_encoder is None else half_encoder
            self.content_encoder.to(torch.float16 if half else torch.float32)
        self._ensure_caches()
        return self

    def _ensure_caches(self):
        tr = self.cfm.estimator.transformer
        if tr.freqs_cis is None or tr.freqs_cis.device != self.device:
            tr.max_seq_length, tr.max_batch_size = -1, -1
            self.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8)
            tr.freqs_cis = tr.freqs_cis.float()

    def trainable_modules(self):
        return {k: getattr(self, k) for k in TRAINABLE_KEYS if getattr(self, k) is not None}

    # ---------------------------------------------------------- features
    def resample(self, wave, orig_sr, new_sr):
        return wave if orig_sr == new_sr else torchaudio.functional.resample(wave, orig_sr, new_sr)

    @torch.no_grad()
    def compute_style(self, waves_16k, wave_lens=None):
        """CAMPPlus speaker embedding (B, 192) from 16 kHz audio (v2 implementation)."""
        if wave_lens is None:
            wave_lens = torch.full((waves_16k.size(0),), waves_16k.size(-1), dtype=torch.long)
        feats = []
        for b in range(waves_16k.size(0)):
            feat = torchaudio.compliance.kaldi.fbank(waves_16k[b:b + 1, : int(wave_lens[b])].float(), num_mel_bins=80,
                                                     dither=0, sample_frequency=ENCODER_SR)
            feats.append(feat - feat.mean(dim=0, keepdim=True))
        max_len = max(f.size(0) for f in feats)
        feat_lens = torch.tensor([f.size(0) for f in feats], dtype=torch.int32, device=waves_16k.device) // 2
        feats = torch.stack([torch.nn.functional.pad(f, (0, 0, 0, max_len - f.size(0)), value=float(f.min()))
                             for f in feats])
        return self.style_encoder(feats, feat_lens)

    @torch.no_grad()
    def extract_content(self, waves_16k, wave_lens=None):
        return self.content_encoder(waves_16k, wave_lens)

    def content_tokens(self, feats, feat_lens):
        """Bottleneck indices (B, T) when enabled, else the continuous features."""
        if self.bottleneck is None:
            return feats
        with torch.set_grad_enabled(self.bottleneck.training and torch.is_grad_enabled()):
            return self.bottleneck(feats, feat_lens)["indices"]

    def regulate(self, content, content_lens, mel_lens, **kwargs):
        return self.length_regulator(content, mel_lens, content_lens, **kwargs)[0]

    # ------------------------------------------------------------ inference
    @torch.inference_mode()
    def convert(self, source, reference, diffusion_steps=None, inference_cfg_rate=None, length_adjust=1.0,
                attn_mode=None, sway_coef="config", max_prompt_seconds=None, max_context_seconds=30.0,
                overlap_frames=16, seed=None):
        """Offline conversion. ``source`` / ``reference``: 1-D float waveforms at ``self.sr``.

        Returns (np.ndarray float32 at self.sr, info dict with tensor shapes).
        """
        if self.length_regulator.f0_condition:
            raise NotImplementedError("f0-conditioned v3 inference is not wired yet (the regulator supports it)")
        inf = _get(self.config, "inference") or Munch()
        steps = int(diffusion_steps or _get(inf, "diffusion_steps", 25))
        cfg_rate = float(_get(inf, "inference_cfg_rate", 0.7) if inference_cfg_rate is None else inference_cfg_rate)
        attn_mode = attn_mode or _get(inf, "attn_mode", ATTN_FULL)
        sway = _get(inf, "sway_coef", -1.0) if sway_coef == "config" else sway_coef
        max_prompt = float(max_prompt_seconds or _get(inf, "max_prompt_seconds", 25.0))
        if seed is not None:
            torch.manual_seed(seed)
        self._ensure_caches()
        device = self.device
        src = torch.as_tensor(source, dtype=torch.float32, device=device).reshape(1, -1)
        ref = torch.as_tensor(reference, dtype=torch.float32, device=device).reshape(1, -1)[:, : int(self.sr * max_prompt)]
        src16, ref16 = self.resample(src, self.sr, ENCODER_SR), self.resample(ref, self.sr, ENCODER_SR)

        feats_s, lens_s = self.extract_content(src16)
        feats_r, lens_r = self.extract_content(ref16)
        tok_s, tok_r = self.content_tokens(feats_s, lens_s), self.content_tokens(feats_r, lens_r)
        mel_r = self.mel_fn(ref)
        n_src = int(self.mel_fn.num_frames(src.size(-1)) * length_adjust)
        cond_s = self.regulate(tok_s, lens_s, torch.tensor([n_src], device=device), time_scale=1.0 / length_adjust)
        cond_r = self.regulate(tok_r, lens_r, torch.tensor([mel_r.size(-1)], device=device))
        style = self.compute_style(ref16)

        P = mel_r.size(-1)
        max_window = int(self.sr / self.hop * max_context_seconds)
        max_src = max_window - P
        if max_src <= overlap_frames:
            raise ValueError("reference too long for the context window; lower max_prompt_seconds")
        overlap_wave = overlap_frames * self.hop
        chunks, prev_tail, processed = [], None, 0
        while processed < cond_s.size(1):
            chunk = cond_s[:, processed:processed + max_src]
            last = processed + max_src >= cond_s.size(1)
            cat = torch.cat([cond_r, chunk], dim=1)
            mel = self.cfm.inference(cat, torch.tensor([cat.size(1)], device=device), mel_r, style, steps,
                                     inference_cfg_rate=cfg_rate, attn_mode=attn_mode, sway_coef=sway)[:, :, P:]
            wave = self.vocoder(mel.float())[0, 0].float().cpu().numpy()
            if processed == 0 and last:
                chunks.append(wave)
                break
            if processed == 0:
                chunks.append(wave[:-overlap_wave])
            elif last:
                chunks.append(crossfade(prev_tail, wave, overlap_wave))
                break
            else:
                chunks.append(crossfade(prev_tail, wave[:-overlap_wave], overlap_wave))
            prev_tail = wave[-overlap_wave:]
            processed += mel.size(2) - overlap_frames
        out = np.concatenate(chunks).astype(np.float32)
        info = {
            "source_16k": tuple(src16.shape), "content_features": tuple(feats_s.shape), "content_tokens": tuple(tok_s.shape),
            "source_cond": tuple(cond_s.shape), "prompt_mel": tuple(mel_r.shape), "prompt_cond": tuple(cond_r.shape),
            "style": tuple(style.shape), "output_samples": out.shape[0], "diffusion_steps": steps, "cfg_rate": cfg_rate,
            "attn_mode": attn_mode,
        }
        return out, info

    def streaming_engine(self, **overrides):
        from modules.v3.streaming import StreamingConfig, StreamingVoiceConverter

        return StreamingVoiceConverter(self, StreamingConfig.from_config(_get(self.config, "streaming"), **overrides))

    # ---------------------------------------------------------- checkpoints
    def save_checkpoint(self, path, keys=TRAINABLE_KEYS, **extra):
        modules = {k: m for k, m in self.trainable_modules().items() if k in keys}
        state = {
            "version": VERSION,
            "config": self.config.toDict() if isinstance(self.config, Munch) else dict(self.config),
            "net": {k: m.state_dict() for k, m in modules.items()},
        }
        state.update(extra)
        torch.save(state, path)

    @staticmethod
    def _load_filtered(module, params, skip_prefixes=()):
        own = module.state_dict()
        params = {k[len("module."):] if k.startswith("module.") else k: v for k, v in params.items()}
        loaded = {k: v for k, v in params.items()
                  if k in own and v.shape == own[k].shape and not k.startswith(skip_prefixes)}
        module.load_state_dict(loaded, strict=False)
        n_loaded = sum(v.numel() for v in loaded.values())
        n_total = sum(v.numel() for v in own.values())
        return {
            "loaded_tensors": len(loaded), "total_tensors": len(own),
            "loaded_params": n_loaded, "total_params": n_total,
            "shape_mismatch": sorted(k for k, v in params.items() if k in own and v.shape != own[k].shape),
            "missing": sorted(k for k in own if k not in loaded),
            "unexpected": sorted(k for k in params if k not in own),
        }

    def load_checkpoint(self, path, keys=TRAINABLE_KEYS):
        """Shape-filtered load of the given module keys; returns a per-module report."""
        state = torch.load(path, map_location="cpu")
        report = {}
        for key in keys:
            module = getattr(self, key, None)
            if module is not None and key in state.get("net", {}):
                report[key] = self._load_filtered(module, state["net"][key])
        return report

    @staticmethod
    def summarize_report(report):
        return {k: f"{r['loaded_tensors']}/{r['total_tensors']} tensors"
                   + (f", {len(r['unexpected'])} unexpected" if r["unexpected"] else "")
                   + (f", {len(r['shape_mismatch'])} shape mismatches" if r["shape_mismatch"] else "")
                for k, r in report.items()}

    def warm_start_from_v1(self, path):
        """Shape-filtered load of a v1 checkpoint (``net.cfm`` / ``net.length_regulator``)."""
        state = torch.load(path, map_location="cpu")
        net = state["net"]
        report = {"cfm": self._load_filtered(self.cfm, net["cfm"])}
        if "length_regulator" in net:
            report["length_regulator"] = self._load_filtered(self.length_regulator, net["length_regulator"],
                                                             _V1_LR_SKIP_PREFIXES)
        return report
