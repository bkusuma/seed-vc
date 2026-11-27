"""Phase 4 dry run for Seed-VC v3 with the real Whisper-v3-turbo encoder, BigVGAN-v2 44 kHz/256x vocoder and
CAMPPlus style encoder. Trainable modules are random or warm-started from v1 (no v3 checkpoint exists yet), so
the converted audio is noise; the point is tensor shapes, I/O invariants, compatibility and timing.

    python tools/v3_dry_run.py [--v1-checkpoint PATH] [--quick]

Writes runs/v3_dry_run/report.json and report.md; exits non-zero if a check fails.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
if sys.platform == "darwin":  # must precede `import torch`: run ops missing on MPS on the CPU
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import numpy as np
import torch

from modules.v3.audio import MelSpectrogram
from modules.v3.device import pick_device, synchronize
from modules.v3.dit import PrefixKVStore
from modules.v3.model import SeedVCv3
from modules.v3.realtime_io import RealtimeVCStream, simulate_stream

V1_44K = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
SOURCE = "examples/source/source_s1.wav"
REFERENCE = "examples/reference/s1p1.wav"
VOCODER_PROBE = "examples/reference/trump_0.wav"  # native 44.1 kHz speech


class Report:
    def __init__(self):
        self.items, self.failed = [], 0

    def check(self, name, ok, **data):
        self.items.append({"check": name, "ok": bool(ok), **data})
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: " + ", ".join(f"{k}={v}" for k, v in data.items()))

    def info(self, name, **data):
        self.items.append({"check": name, "ok": None, **data})
        print(f"[INFO] {name}: " + ", ".join(f"{k}={v}" for k, v in data.items()))


def params(module):
    return round(sum(p.numel() for p in module.parameters()) / 1e6, 1)


DEVICE = None  # set in main(); timings synchronise CUDA / MPS before reading the clock


def timed(fn, *a, **kw):
    if DEVICE is not None:
        synchronize(DEVICE)
    t = time.perf_counter()
    out = fn(*a, **kw)
    if DEVICE is not None:
        synchronize(DEVICE)
    return out, time.perf_counter() - t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1-checkpoint", default=None, help=f"default: {V1_44K} from Plachta/Seed-VC")
    p.add_argument("--quick", action="store_true", help="shorter audio, fewer steps")
    p.add_argument("--out-dir", default="runs/v3_dry_run")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    args = p.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    global DEVICE
    device = DEVICE = pick_device(args.device)
    R = Report()
    import platform

    R.info("environment", device=str(device), torch=torch.__version__, threads=torch.get_num_threads(),
           platform=platform.platform(), mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))

    # 1. build the quality preset with real frozen components ------------------------------------------
    model, t_build = timed(SeedVCv3, "configs/v3/seed_vc_v3_44k.yml")
    R.info("build quality preset", seconds=round(t_build, 1), **{
        f"{k}_M": params(getattr(model, k)) for k in
        ("content_encoder", "bottleneck", "length_regulator", "cfm", "style_encoder", "vocoder")})

    # 2. warm start from the v1 44 kHz SVC checkpoint -----------------------------------------------------
    v1_path = args.v1_checkpoint
    if v1_path is None:
        from hf_utils import load_custom_model_from_hf

        v1_path = load_custom_model_from_hf("Plachta/Seed-VC", V1_44K, None)
    warm = model.warm_start_from_v1(v1_path)
    cfm_ratio = warm["cfm"]["loaded_params"] / warm["cfm"]["total_params"]
    lr = warm["length_regulator"]
    R.check("v1 warm start: DiT", cfm_ratio > 0.99, loaded_M=round(warm["cfm"]["loaded_params"] / 1e6, 1),
            total_M=round(warm["cfm"]["total_params"] / 1e6, 1), ratio=round(cfm_ratio, 4),
            shape_mismatch=warm["cfm"]["shape_mismatch"], not_loaded=warm["cfm"]["missing"])
    R.info("v1 warm start: regulator", loaded_M=round(lr["loaded_params"] / 1e6, 2),
           total_M=round(lr["total_params"] / 1e6, 2), not_loaded=lr["missing"])
    model.prepare(device).eval()

    # 3. front-end / vocoder compatibility: mel -> BigVGAN -> mel ---------------------------------------
    probe = librosa.load(VOCODER_PROBE, sr=model.sr)[0][: model.sr * 3]
    probe_t = torch.from_numpy(probe).to(device)[None]
    with torch.inference_mode():
        mel = model.mel_fn(probe_t)
        wave = model.vocoder(mel)[0]
        n = min(wave.size(-1), probe_t.size(-1))
        err = (model.mel_fn(wave[:, :n]) - model.mel_fn(probe_t[:, :n])).abs().mean().item()
        wrong_fe = MelSpectrogram(model.sr, 1024, 1024, 256, 128, 0, 8000).to(device)  # fmax mismatch control
        wave_bad = model.vocoder(wrong_fe(probe_t))[0]
        err_bad = (model.mel_fn(wave_bad[:, :n]) - model.mel_fn(probe_t[:, :n])).abs().mean().item()
    R.check("BigVGAN 44k/256x round trip (log-mel L1)", err < 0.5 and err < 0.7 * err_bad,
            mel_shape=tuple(mel.shape), wave_samples=int(wave.size(-1)), expected_samples=mel.size(-1) * model.hop,
            l1=round(err, 3), l1_with_fmax8000_frontend=round(err_bad, 3))

    # 4. streaming vocoder crop: how much mel padding makes cropped == full ------------------------------
    with torch.inference_mode():
        full = model.vocoder(mel)[0, 0]
        a, b = 200, 240
        sweep = {}
        for pad in (0, 2, 4, 8, 16, 32):
            seg = model.vocoder(mel[:, :, a - pad:b + pad])[0, 0][pad * model.hop:(pad + b - a) * model.hop]
            ref = full[a * model.hop:b * model.hop]
            sweep[pad] = float((seg - ref).norm() / ref.norm())
    chosen = model.config.streaming.vocoder_pad_frames
    R.check("vocoder crop padding sweep (relative error vs full)", sweep[chosen] < 1e-2,
            **{f"pad{k}": f"{v:.1e}" for k, v in sweep.items()}, config_pad=chosen)

    # 5. offline conversion shape trace --------------------------------------------------------------------
    src_s = 2.0 if args.quick else 4.0
    source = librosa.load(SOURCE, sr=model.sr)[0][: int(model.sr * src_s)]
    reference = librosa.load(REFERENCE, sr=model.sr)[0][: model.sr * 3]
    steps = 2 if args.quick else 4
    (out, info), t_off = timed(model.convert, source, reference, diffusion_steps=steps, inference_cfg_rate=0.7, seed=0)
    R.check("offline conversion (quality preset, warm-started)", abs(len(out) - len(source)) < model.hop and
            np.isfinite(out).all(), seconds=round(t_off, 1), rtf=round(t_off / (len(out) / model.sr), 2),
            **{k: v for k, v in info.items() if k not in ("diffusion_steps", "cfg_rate", "attn_mode")},
            source_samples=len(source))
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    import soundfile as sf

    sf.write(os.path.join(out_dir, "offline_dry_run.wav"), out, model.sr)

    # 6. prefix-cache speed-up for one streaming window (quality preset DiT) ----------------------------
    eng = model.streaming_engine(diffusion_steps=1, cfg_rate=0.0)
    eng.set_reference(reference)
    est = model.cfm.estimator
    W, P = eng.n_mel, eng.prompt_frames
    C, D = model.mel_fn.n_mels, model.config.model_params.DiT.content_dim
    x, cond, style = torch.randn(1, C, W, device=device), torch.randn(1, W, D, device=device), torch.randn(1, 192, device=device)
    prompt_mel, prompt_cond = torch.randn(1, C, P, device=device), torch.randn(1, P, D, device=device)
    t = torch.tensor([0.5], device=device)
    with torch.inference_mode():
        x_full = torch.cat([torch.zeros(1, C, P, device=device), x], -1)
        px_full = torch.cat([prompt_mel, torch.zeros(1, C, W, device=device)], -1)
        c_full = torch.cat([prompt_cond, cond], 1)
        run_full = lambda: est(x_full, px_full, torch.tensor([P + W], device=device), t, style, c_full,
                               prompt_lens=torch.tensor([P], device=device), attn_mode="prompt_isolated")
        store = PrefixKVStore()
        est.forward_prefix(prompt_mel, prompt_cond, style, t, store, 0)
        run_cached = lambda: est.forward_source(x, cond, style, t, store, 0)
        run_full(), run_cached()  # warm up
        reps = 3
        t_full = min(timed(run_full)[1] for _ in range(reps))
        t_cached = min(timed(run_cached)[1] for _ in range(reps))
        exact = (run_full()[..., P:] - run_cached()).abs().max().item()
    R.check("prefix KV cache (one DiT step, quality preset)", exact < 1e-3, prompt_frames=P, window_frames=W,
            single_pass_ms=round(1000 * t_full, 1), cached_ms=round(1000 * t_cached, 1),
            speedup=round(t_full / t_cached, 2), max_abs_diff=f"{exact:.1e}",
            cache_mb_per_step=round(store.memory_bytes() / 2 ** 20, 1))
    del model, eng

    # 7. streaming simulation through the full I/O chain (rt preset, random trainable weights) -------------
    rt = SeedVCv3("configs/v3/seed_vc_v3_44k_rt.yml").prepare(device).eval()
    eng = rt.streaming_engine(diffusion_steps=2 if args.quick else 4)
    eng.set_reference(reference)
    stream = RealtimeVCStream(eng, aec=True, agc=True)
    sim_s = 0.6 if args.quick else 1.2
    mic = librosa.load(SOURCE, sr=stream.sr)[0][: int(stream.sr * sim_s)]
    echo = np.concatenate([np.zeros(int(0.03 * stream.sr)), [0.25]])
    sim, block_times = simulate_stream(stream, mic, echo_ir=echo)
    rep = eng.latency_report()
    n_blocks = len(mic) // stream.block
    R.check("streaming simulation (rt preset, AEC+AGC chain)", len(sim) == n_blocks * stream.block and
            np.isfinite(sim).all(), blocks=n_blocks, block_samples=stream.block, output_samples=len(sim),
            algorithmic_ms=rep["algorithmic_ms"], cpu_compute_mean_ms=round(rep["compute_mean_ms"], 1),
            stage_mean_ms={k: round(v, 1) for k, v in rep["stage_mean_ms"].items()},
            prefix_cache_mb=round(rep["prefix_cache_mb"], 1), dit_window_frames=rep["dit_window_frames"])
    R.info("latency budget", rt_preset_algorithmic_ms=rep["algorithmic_ms"],
           quality_preset_algorithmic_ms=180 + 40 + 40 + 20,
           note="end-to-end = algorithmic + device I/O; compute per block must stay < block_time")
    sf.write(os.path.join(out_dir, "streaming_dry_run.wav"), sim, stream.sr)

    # 8. compute budget: FLOPs per block at the rt preset defaults vs what this device achieved ------------
    from torch.utils.flop_counter import FlopCounterMode

    def count_flops(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops()

    preset = rt.streaming_engine()  # preset defaults: steps, cfg
    preset.set_reference(reference)
    C, D = rt.mel_fn.n_mels, rt.config.model_params.DiT.content_dim
    n_voc = -(-preset.emit_out // rt.hop) + 1 + 2 * preset.cfg.vocoder_pad_frames
    with torch.inference_mode():
        flops = {
            "encoder": count_flops(lambda: rt.extract_content(preset.in_buffer[None])),
            "dit": count_flops(lambda: preset.sampler.sample(torch.zeros(1, preset.n_mel, D, device=device),
                                                             torch.zeros(1, C, preset.n_mel, device=device))),
            "vocoder": count_flops(lambda: rt.vocoder(torch.zeros(1, C, n_voc, device=device))),
        }
    block_s = preset.n_block * 0.02
    stage_s = {k: rep["stage_mean_ms"][k] / 1000 for k in flops}
    sim_dit_flops = flops["dit"] * eng.cfg.diffusion_steps / preset.cfg.diffusion_steps  # simulation used fewer steps
    achieved = {k: (sim_dit_flops if k == "dit" else flops[k]) / stage_s[k] for k in flops}
    projected_ms = 1000 * sum(flops[k] / achieved[k] for k in flops)
    R.info("compute budget per block (rt preset defaults)", device=str(device),
           gflops={k: round(v / 1e9, 1) for k, v in flops.items()},
           required_tflops_for_realtime=round(sum(flops.values()) / block_s / 1e12, 2),
           achieved_tflops={k: round(v / 1e12, 3) for k, v in achieved.items()},
           projected_compute_ms=round(projected_ms, 1), block_ms=round(1000 * block_s),
           realtime_feasible=projected_ms < 1000 * block_s)

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(R.items, f, indent=2, default=str)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("| check | result | details |\n|---|---|---|\n")
        for it in R.items:
            status = {True: "PASS", False: "FAIL", None: "info"}[it["ok"]]
            details = "; ".join(f"{k}={v}" for k, v in it.items() if k not in ("check", "ok"))
            f.write(f"| {it['check']} | {status} | {details} |\n")
    print(f"\n{len(R.items)} entries, {R.failed} failed; report in {out_dir}")
    return R.failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
