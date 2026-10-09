# Seed-VC v3.0 — Technical Upgrade Plan

Status: implemented on branch `claude/v3-upgrade` (§9: validation; §11: Apple Silicon).
Scope: additive. v1 (`inference.py`, `real-time-gui.py`, `train.py`) and v2 (`*_v2.py`) are untouched.

---

## 0. Bottom line

1. **v3 is a new architecture, so it needs new weights.** No released checkpoint matches it. Swapping the content encoder (768-d → 1280-d features), the mel front-end (22.05 kHz/80 bands → 44.1 kHz/128 bands at 172 fps) and the conditioning path invalidates every existing DiT checkpoint. The code runs end-to-end and the tensor shapes are validated. Until you train it, the audio it produces is noise. *Confidence: high.*
   - Mitigation: the v3 "quality" preset deliberately mirrors the v1 44 kHz SVC DiT (768×17, 128 mel bins). Measured warm start: **186.3 M of 186.4 M DiT parameters** and 7.7 M of 9.3 M regulator parameters load from that checkpoint; only the new discrete content embedding starts from scratch (§9). That makes training v3 a **fine-tune**, not a from-scratch run.
2. **"Turbo" does not speed up voice conversion.** Whisper-large-v3-turbo prunes the *decoder* from 32 layers to 4. Seed-VC uses only the *encoder*, which is the full large-v3 encoder: 32 layers, d=1280, **637.0 M params vs 88.2 M** for whisper-small (measured, 7.2×). That works against the real-time goal. Mitigations built in:
   - a configurable encoder padding length (§5.2);
   - optional layer truncation;
   - using the turbo *decoder* where it actually helps: cheap ASR pseudo-labels for the new ASR loss.

   **Correction made during implementation:** I first planned a variable-length (unpadded) encoder pass for streaming. Measurement killed it. On 1.5–3 s windows, the features had cosine 0.29–0.57 vs the native 30 s-padded features, and the turbo decoder transcribed a 2 s clip as "A." instead of the sentence. Whisper-v3's encoder is only reliable with long zero-padding. Measured fidelity vs padding:

   | Padding | Cosine vs 30 s-padded |
   |---|---|
   | 5 s | 0.70–0.79 |
   | 10 s | 0.87–0.91 |
   | 15 s | 0.93–0.96 |
   | 20 s | 0.95–0.97 |

   Default is therefore 30 s padding (same as v1), so **every real-time block pays a full 30 s large-v3 encoder pass**, about 1.9 TFLOP. That is the largest single cost in the v3 real-time path.
3. **Hop 256 at 44.1 kHz doubles the DiT sequence length** compared with the existing 44 kHz model (hop 512): 172.3 fps vs 86.1 fps. Linear-layer cost is about 2× and attention cost about 4×. The upgrade implements what you asked for (BigVGAN-v2 44 kHz, 128 bands, 256×). `nvidia/bigvgan_v2_44khz_128band_512x` is a one-line config change if latency wins over time resolution.
4. **Sub-400 ms is a budget, not something I could measure here.** This container has no GPU or audio device. v3 adds the mechanisms that cut per-block compute:
   - a cached prompt prefix (KV cache), so each block computes only its window (81 frames in the `rt` preset) instead of prompt + window (597). Measured: 3.1–3.4× faster per DiT step on CPU for the quality preset, max error 4e-6;
   - a shorter encoder padding option;
   - a cropped vocoder;
   - fp16/bf16 autocast for the DiT.

   It also reports the exact algorithmic latency, and a benchmark gives per-stage timings on your GPU. The algorithmic part can be set to about 160 ms. Whether total delay stays under 400 ms depends on compute per block (it must be under the block time) plus device I/O. *Confidence: medium that a 4090-class GPU fits; low for mid-range GPUs at 8+ diffusion steps with CFG.*
5. **The five components you named date from 2024** (Whisper-v3-turbo: Oct 2024; BigVGAN-v2: 2024). The clearest late-2025/2026 advance for your latency goal is **1-step mean-flow sampling for streaming VC**:
   - MeanVC (arXiv 2510.08392, Oct 2025);
   - MeanVC 2 (arXiv 2606.09050, Jun 2026);
   - MeanVoiceFlow2 (arXiv 2609.40087, Sep 2026).

   It needs a different training objective, so it is out of scope here. §10 lists it as the top follow-up.
6. **Target hardware is an M1 MacBook Pro.** v3 now runs on Apple Silicon (MPS) for offline conversion and development (§11). **Real time does not fit any M1**:
   - the real-time preset needs **11.2 TFLOPS sustained** (measured FLOPs);
   - Apple quotes 2.6 / 5.2 / 10.4 TFLOPS FP32 peak for M1 / M1 Pro / M1 Max.

   The Whisper-large-v3 encoder alone is 85% of that budget. *Confidence: high on the FLOPs; the conclusion holds even at 100% utilisation.*

---

## 1. Phase 1 findings (current state)

| Area | What exists | Implication for v3 |
|---|---|---|
| DiT (`modules/diffusion_transformer.py`) | Non-autoregressive flow-matching DiT. RoPE, UViT skips, AdaLN, optional time/style tokens, MLP or WaveNet head | Already NAR. "Chunked execution" means a streaming **execution path**, not a new generator |
| CFM (`modules/flow_matching.py`) | Euler solver, batched CFG, prompt infilling (`x[..., :P]=0`) | Reused semantics; v3 adds a cached-prefix sampler |
| Real-time (`real-time-gui.py`) | Sliding window + SOLA. README: ~300 ms algorithmic + ~100 ms device; 430 ms measured for xlsr-tiny | Recomputes the 3 s prompt through every layer, every step, every block |
| 44.1 kHz | `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`: BigVGAN-v2 44k **512×**, 2048 FFT / 512 hop, 128 bands, DiT 768×17 | Best warm-start source for v3 |
| Content encoders | whisper-small (v1), XLSR (v1 rt), HuBERT+ASTRAL BSQ (v2) | — |
| Disentanglement | v2 has BSQ codebooks (32 / 2048) **inference-only**. `configs/astral_quantization/*.yml` reference `modules.astral_quantization.asr_decoder`, which is **not in the repo** | The ASR-loss training path must be written, not wired |

Latent issues found (left as-is in v1/v2; avoided in v3):
- **Window-dependent normalization.** `InterpolateRegulator` uses `GroupNorm(1, C)`, which normalizes over the **whole sequence**. A frame's conditioning therefore depends on the window it sits in, which is wrong for streaming. v3 uses a per-frame channel norm with the same parameter shapes, so warm-starting still works.
- **Forced 30 s pass.** HF `WhisperEncoder.forward` rejects inputs that are not exactly 3000 mel frames, so every real-time block runs a full 30 s encoder pass.
- **BSQ entropy-loss shape bug.** `BinarySphericalQuantize` computes `soft_entropy_loss` on a 4-D tensor that the function treats as 2-D. It is harmless at inference but would be a wrong training signal. v3 computes the BSQ entropy loss itself, on `(N, bits)`.
- **Shared mel cache.** `modules/audio.mel_spectrogram`'s global mel-basis cache is keyed only by `(sr, fmax, device)`. Two configs with the same sr and fmax but different n_fft/n_mels (v1 44k preset vs v3) would reuse the wrong basis in one process. v3 uses a per-module basis.
- **Hard-coded `fmax`.** `inference.py` / `real-time-gui.py` map any `fmax` other than the string `"None"` to 8000. A YAML `fmax: null` becomes 8000. v3 resolves `fmax` properly.

---

## 2. v3 architecture

```
mic/file ──► [AEC ► AGC/gate] ──► 16 kHz ──► Whisper-large-v3-turbo encoder (padded window, 50 Hz, 1280-d)
                                                   │
                                     ContentBottleneck: causal ConvNeXt ► BSQ (2^11 codes)  ──(train only)──► CTC ASR head + speaker-conditioned recon
                                                   │ indices (50 Hz)
                                     TimeAlignedRegulator: embed ► nearest-in-time ► 172.3 fps ► frame-local conv stack
                                                   │ cond (T_mel, 768)
 reference ──► 44.1k 128-band mel (prompt) ─► StreamingDiT (CFM, prompt-isolated attention, cached prefix KV)
 reference ──► CAMPPlus style (192) ───────┘       │ mel (128, T_mel)
                                     BigVGAN-v2 44 kHz 128-band 256× ──► SOLA ──► out
```

### Tensor shapes (batch B, N samples at 44.1 kHz)

| Stage | Shape | Rate |
|---|---|---|
| source wave | (B, N) | 44100 Hz |
| 16 kHz wave | (B, N16), N16 = round(N·16000/44100) | 16000 Hz |
| Whisper log-mel | (B, 128, N16//160) | 100 fps |
| Whisper encoder out | (B, Tc, 1280), Tc = ceil((N16//160)/2) | 50 fps |
| Bottleneck latent → BSQ | (B, Tc, 512) → indices (B, Tc) ∈ [0, 2048) | 50 fps |
| Regulated cond | (B, Tm, 768), Tm = N // 256 | 172.27 fps |
| Target / prompt mel | (B, 128, Tm) | 172.27 fps |
| DiT I/O | x, prompt_x: (B, 128, T); cond: (B, T, 768); style: (B, 192); t: (B,) → (B, 128, T) | — |
| BigVGAN out | (B, 1, Tm·256) | 44100 Hz |

Time alignment (train and inference identical): mel frame *m* uses content frame `j = floor(((m + 0.5)·256/44100)/0.02 + 0.5)`, clipped. Whisper frame *j* is centred at *j*·20 ms. Because the mapping depends only on absolute time, never on window length, streaming windows reproduce the offline alignment exactly.

---

## 3. Design decisions and trade-offs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Integration | New `modules/v3/` + new entry points | Replace v1 | No regressions; v1/v2 checkpoints keep working |
| DiT code | Subclass v1 `DiT` (same parameter names) | New DiT | Warm start from v1 checkpoints; proven blocks |
| Streaming generator | Sliding window, **NAR** per block, prompt-isolated attention + cached prefix KV | Chunk-wise AR with clean-mel cache (MeanVC) | You asked for NAR. Caveat: MeanVC's ablation reports CER 5.01% → 5.97% without its clean-chunk cache, so NAR windows give up some cross-chunk consistency |
| Training masks | Random mix: full / prompt-isolated / prompt-isolated + random right-crop | Fixed full attention | CosyVoice 2 (arXiv 2412.10117) mixed-mask training: one model serves offline and streaming |
| Bottleneck rate | BSQ 2^11 = 2048 codes @ 50 Hz (550 bps) | 2^9–2^10 | Same width as v2's proven "wide" quantizer. Smaller = less timbre leakage but worse prosody/intelligibility. Config knob |
| ASR loss | CTC over UTF-8 bytes (257 classes) | CTC over Whisper BPE (51 866) | BPE CTC logits at 50 Hz cost ~2.4 GB/batch-of-8×30 s. Bytes are language-agnostic and cheap; BPE is a config option |
| Bottleneck conv | Causal (left-padded) ConvNeXt | Symmetric (ASTRAL) | Removes the bottleneck's right-edge dependency in streaming windows |
| Encoder input | Zero-pad each window to `pad_seconds` (default 30 = native) via own forward | Unpadded variable length | Unpadded short windows are out of distribution (measured cosine 0.29–0.57). Padding is a latency knob (15 s ≈ 2× cheaper at cosine ~0.95) and must match between training and inference |
| AEC | Two-path partitioned-block frequency-domain NLMS + Geigel gate + light NLP (numpy) | speexdsp / WebRTC APM | No new native dependency; portable; testable offline. Single-filter variants measured worse (§9) |

---

## 4. Exact file modifications

### New files

| File | Purpose |
|---|---|
| `configs/v3/seed_vc_v3_44k.yml` | Quality preset: DiT 768×17 (warm-startable from v1 44k), Whisper-v3-turbo, BSQ-2048, BigVGAN 44k/256× |
| `configs/v3/seed_vc_v3_44k_rt.yml` | Real-time preset: DiT 512×13, 15 s encoder padding, fp16 DiT, 160 ms algorithmic latency |
| `modules/v3/__init__.py` | Package exports |
| `modules/v3/audio.py` | `MelSpectrogram` module (per-instance mel basis, resolved `fmax`), `resolve_fmax`, `StreamingResampler` (exact block resampling) |
| `modules/v3/content_encoder.py` | `WhisperContentEncoder`: encoder-only load, torch Whisper log-mel (exact parity vs HF), configurable padding, layer truncation, 30 s chunking for long audio |
| `modules/v3/bottleneck.py` | `ContentBottleneck` (causal ConvNeXt → BSQ), `bsq_entropy_loss`, CTC head, speaker-conditioned recon decoder, `ByteTokenizer`/`WhisperTextTokenizer` |
| `modules/v3/length_regulator.py` | `TimeAlignedRegulator` (discrete/continuous, frame-local norm, time-based alignment) |
| `modules/v3/dit.py` | `StreamingDiT(DiT)`: explicit attention modes, prefix KV capture/reuse |
| `modules/v3/flow_matching.py` | `CFMv3`: mixed-mask training loss, offline sampler, `PrefixCachedSampler` |
| `modules/v3/model.py` | `SeedVCv3` container, builder, checkpoint I/O, v1 warm start, offline `convert()` |
| `modules/v3/streaming.py` | `StreamingVoiceConverter`: block engine (padded-window encoder, cached DiT, cropped vocoder, SOLA), latency report |
| `modules/v3/device.py` | Device selection (CUDA > MPS > CPU), MPS-safe float32 transfer, CPU front-end placement, autocast dtype rules, synchronize |
| `modules/v3/realtime_io.py` | `EchoCanceller`, `AutomaticGainControl`, `RealtimeVCStream` (sounddevice, lazy import), `simulate_stream` |
| `data/v3_dataset.py` | 44.1 kHz/128-band dataset with optional transcript side-cars (`.txt`/`.lab`/`.normalized.txt`) |
| `train_v3.py` | Stage `bottleneck` (CTC + recon + entropy) and stage `dit` (mixed-mask CFM) |
| `inference_v3.py` | Offline CLI (`--dry-run` allows random weights) |
| `realtime_v3.py` | Headless real-time CLI (`--simulate in.wav` works without audio hardware) |
| `tools/pseudo_label_v3.py` | Whisper-v3-turbo transcription → side-car `.txt` for the CTC loss |
| `tools/v3_dry_run.py` | Phase 4 end-to-end dry run + stage timing + shape report |
| `tests/v3/*.py` | pytest suite (shapes, cache equivalence, alignment, Whisper log-mel parity, AEC ERLE, AGC, training step, device portability / MPS guards; `V3_TEST_DEVICE=mps` runs it on Apple Silicon) |

### Modified files
- `README.md` — short "V3 (experimental)" section pointing here.

No new pip dependencies. `sounddevice` is already required and imported lazily, so PortAudio-less machines can still run `--simulate`.

---

## 5. Upgrade specifications

### 5.1 44.1 kHz synthesis (Phase 3a)
- Mel: `sr=44100, n_fft=1024, win=1024, hop=256, n_mels=128, fmin=0, fmax=None`. This is identical to `nvidia/bigvgan_v2_44khz_128band_256x/config.json` (verified). `MelSpectrogram` is numerically identical to `modules/audio.mel_spectrogram` (tested), whose padding and log-compression match BigVGAN's `meldataset`. The model builder refuses a vocoder whose `num_mels`/`hop`/`sr`/`n_fft`/`win`/`fmax` differ from the front-end.
- Vocoder: `BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_256x')`, weight-norm removed, `use_cuda_kernel` configurable. 112.7 M params; (B,128,T) → (B,1,256·T).
- Streaming vocoder cropping: vocode only `[block + crossfade + search]` frames plus `vocoder_pad_frames` on each side. The pad was chosen by measurement (§9): 16 frames → 0.12% relative error vs full-length vocoding. The originally planned 8 frames gave 4.2%.

### 5.2 Encoder modernization (Phase 3b)
- `WhisperModel.from_pretrained(name).encoder` only; the decoder is dropped. fp16 on CUDA, fp32 on CPU.
- Own forward:
  ```
  gelu(conv1) → gelu(conv2) → + embed_positions[:T]
  → layers[:output_layer]
  → (final layer_norm if all layers)
  ```
  It accepts any padded length ≤ 30 s (`pad_seconds`, default 30). Longer inputs are chunked to 30 s with 5 s overlap, as in v1. With `pad_seconds: 30` the output equals HF `WhisperEncoder` exactly (measured max abs diff 0.0).
- Log-mel computed in torch on-device: n_fft 400, hop 160, 128 Slaney bins from the HF extractor, `log10`, `max−8` clamp, `(x+4)/4`. Parity with `WhisperFeatureExtractor` is tested.
- `pad_seconds: null` (unpadded) is available but **not recommended**: see §0 item 2 for the measurements.

### 5.3 Streaming and latency (Phase 3d)
- **Attention modes** in `StreamingDiT` (boolean SDPA masks; True = attend):
  - `full`: v1 behaviour.
  - `prompt_isolated`: prompt and special tokens attend only to themselves; source tokens attend to everything valid.
- **Prefix KV cache.** Under `prompt_isolated`, the prefix's per-layer K/V depend only on the reference and the step's *t*, so they are computed **once per reference** for each of the *n* steps × {cond, uncond}. Each block then runs only its window's tokens with keys `[prefix ‖ window]`.
  - It plugs into v1 `Attention.kv_cache.update()`; v1 attention code is unchanged.
  - Exactness: cached output equals the single-pass prompt-isolated forward (measured max abs err 4.2e-6 with real dimensions, fp32).
  - Memory: 2·P·hidden values per layer·step·branch. Measured 51.4 MB per step for the quality preset (P=516, fp32, no CFG). ×steps, ×2 with CFG, ÷2 with fp16/bf16 autocast (`streaming.precision`).
- **Mixed-mask training** (`CFMv3.forward`): per batch, sample `full`/`prompt_isolated`; with probability `crop_prob`, truncate the target to a random short span so training sees window-like right edges.
- **Block engine** (port of the GUI timeline, GUI-free). Per block:
  1. append input;
  2. encode the last `ce_left + crossfade + search + block + right` seconds (padded to `pad_seconds`);
  3. run the bottleneck on that whole window (causal, so earlier context only helps);
  4. time-align to the DiT window's mel frames (absolute stream grid);
  5. run the cached-prefix Euler sampler;
  6. vocode the cropped range;
  7. apply SOLA (DDSP-SVC algorithm, as in GUI).

  Frame-anchored noise: initial noise is indexed by absolute mel frame, so overlapping frames of consecutive windows start from the same noise, which improves crossfade coherence.
- **Latency.**
  ```
  algorithmic = block + right_ctx + crossfade + sola_search
  end-to-end  ≈ algorithmic + compute(≤ block) + device I/O
  ```
  Preset `rt`: 100 + 20 + 20 + 20 ms = 160 ms algorithmic. With compute ≤ 100 ms and device I/O 40–100 ms, that is ≈ 300–360 ms. `StreamingVoiceConverter.latency_report()` prints this live.

### 5.4 Disentanglement (Phase 3c)
`ContentBottleneck`:
- Whisper features (1280) → causal ConvNeXt-V2 (512-d, 6 blocks).
- Then BSQ: project to 11 bits, L2-normalise, sign, project out; straight-through.

Training losses (stage `bottleneck`, encoder frozen):
- **CTC ASR loss** on quantized latents → light ConvNeXt head → 257 byte classes (blank = 256). Transcripts come from side-cars, or from `tools/pseudo_label_v3.py`, which uses the turbo decoder.
- **Speaker-conditioned reconstruction.** A decoder with `gin` = CAMPPlus embedding of the *original* audio reconstructs layer-normed Whisper features of the *original* audio from the bottleneck of the *timbre-perturbed* audio (OpenVoice converter, as in v1). Speaker identity is supplied through `gin`, so the bottleneck has no incentive to carry it, while prosody and phonetics are rewarded.
- **BSQ entropy loss** (per-sample entropy ↓, codebook entropy ↑) for codebook utilisation.
- Inference uses encoder + quantizer only. The DiT consumes **discrete indices** through an embedding table: the hard bottleneck that limits timbre leakage.

Stage `dit`:
- Freeze encoder and bottleneck.
- Source condition from perturbed audio; prompt region from the original (v1 recipe).
- 44.1 kHz 128-band target; CAMPPlus style; mixed-mask CFM loss.

### 5.5 Real-time I/O (Phase 3e)
- **AEC** (`EchoCanceller`):
  - Partitioned-block frequency-domain NLMS: overlap-save, 2N FFT, per-bin step normalised by the far-end power over the whole filter span, AUMDF-style gradient constraint.
  - **Two echo-path models.** A background filter adapts; a foreground filter produces the output. The background is copied over when it is better and reset when it diverges.
  - Tail ≈ 200 ms at device rate; far-end = what we just played.
  - Configurable bulk delay (`--aec-delay-ms`) for long device round trips.
  - Geigel gate on background adaptation, plus a light non-linear processor (NLP).
  - Measured on a synthetic 44.1 kHz echo path:
    - ~22 dB steady-state ERLE;
    - ~24 dB near-end SNR during double talk;
    - ~2% of one CPU core.

    The single-filter variants were unstable (smoothed-PSD normalisation overshoots at speech onsets) or stalled (Geigel false alarms on 46% of far-only frames).
- **AGC** (`AutomaticGainControl`):
  - RMS tracker with separate attack/release.
  - Target −20 dBFS, gain cap +30 dB.
  - Noise gate with hold, so silence is not boosted.
  - Soft limiter (tanh above −1 dBFS).
- **Stream**: sounddevice duplex callback.
  ```
  mono → AEC → AGC → engine.process_block → playback
  ```
  The output becomes the AEC far-end reference. The gate state can mute output during silence, a lightweight replacement for the GUI's FSMN-VAD.

---

## 6. Training plan

| Stage | Trainable | Frozen | Data | Notes |
|---|---|---|---|---|
| A `bottleneck` | ConvNeXt + BSQ + CTC head + recon decoder | Whisper enc, CAMPPlus, OpenVoice | Speech with transcripts (or pseudo-labels) | Watch CTC CER and codebook perplexity |
| B `dit` | DiT (CFM) + regulator | Whisper enc, bottleneck, CAMPPlus | Speech (+ singing for SVC) | `--init-from-v1` warm start |

- Warm start (`SeedVCv3.warm_start_from_v1`): shape-filtered load from `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth`.
  - Expected transfer: the whole DiT, plus the regulator conv stack/norms.
  - Not transferred: the content embedding (new discrete input) and the f0 embedding (f0 off by default).
  - Measured transfer: 186.3 M / 186.4 M DiT params; 7.7 M / 9.3 M regulator params (§9).
- Compute: not estimated here. The original v1/v2 training budgets are not documented in the repo (*I do not have a verified figure*).
- Evaluate with `eval.py` metrics (speaker similarity, WER/CER, DNSMOS). Compare against the v1 44 kHz model at equal diffusion steps.

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Whisper-large features may carry more speaker info than whisper-small (hypothesis, not measured here) | Medium | Timbre leakage | Hard BSQ bottleneck + perturbation + recon-with-speaker; tune codebook bits |
| Whisper features at the window's right edge lack future context (bidirectional encoder) | High | Content errors in the last frames of each block | `right_context`, `ce_left` ≥ 1 s; same regime as v1's real-time path |
| 30 s-padded large-v3 pass per block dominates real-time compute | High | Latency | `pad_seconds` 15–20 (retrain to match), `output_layer` truncation, or a lighter encoder for the `rt` preset |
| NAR windows lose cross-block consistency | Medium | Prosody/timbre drift between blocks | Frame-anchored noise, SOLA, long `dit_left`; AR clean-cache is a follow-up |
| 44.1 k / hop 256 DiT too slow for real-time | Medium–high on mid GPUs | > 400 ms | 512× vocoder preset, `rt` DiT, fewer steps, `cfg_rate=0`, torch.compile / CUDA graphs |
| AEC misaligned delay | Medium | Residual echo | Delay pre-compensation from stream latency; adjustable `--aec-delay-ms`; headphones recommended |

Reversibility: everything is additive and behind new entry points. Deleting `modules/v3/` plus the new scripts restores the previous state.

---

## 8. Validation plan (Phase 4)
1. **Unit (pytest, CPU, no downloads)**:
   - DiT forward shapes for all attention modes;
   - prefix-cache vs single-pass equivalence;
   - regulator alignment vs window offsets;
   - BSQ indices range and entropy finite;
   - CTC loss finite and backprop;
   - AEC ERLE on synthetic echo;
   - AGC convergence, gate and limiter;
   - one training step per stage with tiny configs.
2. **Integration (real weights)**:
   - Whisper log-mel parity vs `WhisperFeatureExtractor`;
   - unpadded vs 30 s-padded encoder features;
   - BigVGAN 44k round-trip (mel → wave → mel);
   - vocoder crop-pad sweep;
   - v1 warm-start transfer ratio.
3. **End-to-end dry run** (`tools/v3_dry_run.py`):
   - offline conversion of `examples/` with real encoder/vocoder/CAMPPlus and warm-started DiT;
   - streaming simulation over the same file;
   - output-length invariants, latency budget, per-stage CPU timings, prefix-cache speed-up.

## 9. Validation results (Phase 4, CPU-only container)

All of these are reproducible with `python -m pytest tests/v3` (54 passed, 1 opt-in slow test passed) and `python tools/v3_dry_run.py` (7/7 checks pass).

| Check | Result |
|---|---|
| Whisper log-mel vs `WhisperFeatureExtractor` | identical (max abs diff 0.0) |
| v3 encoder (`pad_seconds: 30`) vs HF `WhisperEncoder` | identical (max abs diff 0.0) |
| Unpadded encoder windows (1.5–3 s) vs padded | cosine 0.29–0.57; transcripts broken. Design changed (§0.2) |
| v1 DiT parity (`attn_mode="full"`) | bit-identical outputs across 4 head/token configurations |
| Prefix KV cache vs single pass | max abs err 4.2e-6 (real dims); 3.1–3.4× faster per DiT step on CPU |
| Cached sampler vs offline sampler (Euler + sway, CFG on/off) | max abs err 2.4e-7 |
| Regulator window vs offline (absolute-time alignment) | exact in the interior. The v1 GroupNorm variant fails the same test |
| BigVGAN 44k/256× round trip (mel → wave → mel) | log-mel L1 0.086. Control with an `fmax=8000` front-end: 1.25 |
| Vocoder crop padding | 8 frames → 4.2% error (failed), 16 → 0.12% (new default) |
| v1 44k checkpoint warm start | DiT 186.3/186.4 M params; regulator 7.7/9.3 M |
| Offline conversion, quality preset, 4 s source | shapes (1,64000) → (1,200,1280) → (1,200) → (1,689,768) → 176,384 samples (source 176,400; diff < 1 hop) |
| Streaming simulation, rt preset, AEC+AGC chain | output length exact (12 × 4,410); algorithmic latency 160 ms |
| CPU compute per 100 ms block (4 threads, rt preset) | 3.8 s: encoder 3.0 s, DiT (4 steps) 0.29 s, vocoder 0.54 s. **Not real-time on CPU**; the encoder dominates. GPU timings are unmeasured |
| Training CLI, both stages, real dims | pseudo-label → stage A → stage B (v1 warm start + stage A bottleneck) → `inference_v3.py --checkpoint` all run |
| AEC (synthetic echo path) | 21.6 dB ERLE (4–12 s), 24.3 dB near-end SNR in double talk |

Defects found and fixed during validation:
- Unpadded Whisper windows: features out of distribution (measured, see above).
- AEC instability (normalisation lag) and stalled adaptation (double-talk false alarms) → two-path design.
- AEC/AGC frame size at 22.05 kHz: 220-sample frames don't tile 20 ms blocks → 20 ms frames there.
- Vocoder crop pad too small (8 → 16 frames).
- Stage A checkpoints included the untrained DiT (~1 GB); loading one could overwrite a warm-started DiT. Checkpoints are now scoped per stage, and `inference_v3.py` rejects stage-A checkpoints.
- Frozen stage-B bottleneck left in train mode (no numeric effect yet, but fragile) → eval.

Known limits of this validation:
- **No GPU here**, so the sub-400 ms claim is unmeasured. Run `realtime_v3.py --simulate ... --dry-run` on the target GPU and read `compute_p95_ms` against `block_ms`.
- **No trained v3 weights**, so audio quality, speaker similarity and timbre leakage are unmeasured.
- Stage B on CPU with two ~15 s clips was OOM-killed at ~14 GB: the masked attention falls back to the O(T²) math kernel on CPU. On CUDA, PyTorch's memory-efficient SDPA kernel supports boolean masks. Use GPU training or `training.max_duration` ≤ 10 s on CPU.

## 10. Out of scope / recommended next steps
1. **Mean-flow 1-step sampling** (MeanVC family). This is the largest remaining latency lever: 8–10 NFEs → 1–2. It requires a new loss (average-velocity target via JVP) and a retrain.
2. Chunk-wise AR clean-mel cache as an optional streaming mode (MeanVC ablation shows intelligibility gains).
3. torch.compile / CUDA-graph capture of the cached-prefix sampler (eager kernel-launch overhead dominates small windows).
4. Gradient-reversal speaker classifier on the bottleneck if leakage remains measurable.

## 11. Apple Silicon (M1) support

**Status.** v3 runs on Apple Silicon (MPS) at the code level. Every MPS-specific risk found in an audit is fixed and guarded by tests. However, I have **not executed it on Metal**, because this container has no Mac. The three commands below confirm it on your M1.

**Conclusion first:** v3 will run on an M1 for offline conversion, dry runs and small experiments. It **cannot run in real time on any M1**. That isn't a tuning problem; the FLOP budget below exceeds the chip.

### Changes for MPS
| Risk | Fix |
|---|---|
| float64 tensors can't exist on MPS, and there's no CPU fallback for dtypes. Two places hit this: time-alignment maths, and `torch.as_tensor(float64_numpy, dtype=float32, device=...)`, which materialises float64 first | float64 maths runs on the CPU; numpy input is cast to float32 on the CPU before transfer (`device.as_float32`). Two tests guard this: a source scan, and a dispatch trace of a full conversion plus streaming run, which asserts float64 only appears inside CPU-only helpers |
| Ops missing from older PyTorch MPS builds | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before `import torch` in every v3 entry point and in `modules/v3/__init__.py` |
| FFT/complex front-ends: Whisper log-mel, BigVGAN mel, Kaldi fbank, sinc resampling | Run on the CPU when the model is on MPS. They're cheap and give identical results across macOS/PyTorch versions |
| CTC loss support varies by PyTorch version on MPS | Computed on the CPU (exact; gradients flow back to MPS) |
| Whisper encoder ran in fp32 everywhere except CUDA | fp16 on MPS too: 2.5 GB → 1.3 GB |
| DiT precision | Autocast fp16 on MPS. bf16 needs macOS 14 (PyTorch's rule), otherwise fp16. With no MPS autocast (old PyTorch), fp32 with a warning. Disabled autocast is a true no-op, so there are no per-block warnings |
| MPS runs asynchronously, so stage timings were wrong | `torch.mps.synchronize()` before reading clocks |
| Device selection | CUDA > MPS > CPU in every CLI, with a `--device` flag |

### Validate on your Mac
```
V3_TEST_DEVICE=mps python -m pytest tests/v3 -q
python tools/v3_dry_run.py --device mps --quick
python realtime_v3.py --simulate examples/source/source_s1.wav --reference examples/reference/s1p1.wav --dry-run --max-seconds 2
```
The dry run's last line, "compute budget per block", prints:
- the TFLOPS your Mac achieves per stage;
- the projected compute per 100 ms block;
- whether the real-time preset fits.

### Why real time does not fit an M1
FLOPs per streaming block, measured with `torch.utils.flop_counter` on the real models:

| Per block | `rt` preset (100 ms block, 8 steps, no CFG, 15 s pad) | quality preset (180 ms block, 10 steps, CFG, 30 s pad) |
|---|---|---|
| Whisper-large-v3 encoder | 953 GFLOP | 1,905 GFLOP |
| DiT with cached prefix (uncached) | 63 (464) GFLOP | 720 (3,444) GFLOP |
| BigVGAN-v2 44k | 106 GFLOP | 138 GFLOP |
| **Sustained throughput needed for real time** | **11.2 TFLOPS** | **15.4 TFLOPS** |

Apple's quoted FP32 peaks are 2.6 TFLOPS for M1 (8-core GPU), 5.2 for M1 Pro (16-core) and 10.4 for M1 Max (32-core). Even at 100% utilisation, no M1 reaches the `rt` budget.
- *Uncertainty:* I haven't verified whether M1's GPU runs fp16 faster than the FP32 figure. Even at 2×, only an M1 Max could fit, and only at more than 54% sustained utilisation. That is unlikely for PyTorch MPS (*unverified*; the dry run measures it).
- Without the encoder, DiT plus vocoder still needs about 1.7 TFLOPS sustained in the `rt` preset. BigVGAN-44k alone needs about 1.1 TFLOPS.

Memory (measured parameter counts):

| Preset | Footprint | Breakdown |
|---|---|---|
| `rt` | ≈ 2.3 GB + activations | encoder 1.3 GB fp16, DiT 0.27 GB, vocoder 0.45 GB, prefix cache ~0.1 GB fp16 |
| quality | ≈ 3.5 GB + activations | as above, plus DiT 0.75 GB and prefix cache ~0.5 GB fp16 |

An 8 GB M1 works for `rt` but is tight next to macOS. On 8 GB Macs, lower `inference_v3.py --max-context-seconds`.

Offline conversion is viable. Example: 10 s of audio with the quality preset (25 steps + CFG) measures 36.4 TFLOP:

| Chip | Estimated time for 10 s of audio |
|---|---|
| M1 | 23–47 s |
| M1 Pro | 12–23 s |
| M1 Max | 6–12 s |

These estimates assume 30–60% of peak. About 10 steps halves them.

### Options if real time on the Mac is required (your decision)
1. **Keep v3 as specified.** Use the M1 for offline conversion and development, and a CUDA GPU (local or cloud) for real time. No code change.
2. **A Mac real-time variant.** This departs from the v3 spec. It replaces the two dominant costs:
   - the encoder → a windowed SSL encoder with no 30 s padding, like v1's 12-layer XLSR real-time model;
   - the vocoder → a lighter one.

   It would also use fewer DiT steps (1–2 after a mean-flow retrain). Rough budget: ~0.2 TFLOP per 100 ms block, which is plausible on M1 Pro/Max and unlikely on base M1 (*estimate*). It needs an SSL-encoder option in v3, a new preset and its own training run.
3. **Larger blocks (≥ 0.5 s).** These amortise the fixed encoder cost on an M1 Max but push latency well past 400 ms. Not recommended.

## 12. Open questions for you
- Which M1 exactly (M1 / M1 Pro / M1 Max, and RAM)? See §11: real time does not fit any M1 with this spec.
- Is real time on the Mac a must (§11 option 2), or is offline on the Mac plus a CUDA GPU for live use acceptable (option 1)?
- Speech-only, or singing too? Singing needs `f0_condition: true` (RMVPE adds ~10–20 ms per block, *unverified*).
- Do you have transcripts? If not, run `tools/pseudo_label_v3.py` first (turbo decoder).
