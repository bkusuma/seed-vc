"""Seed-VC v3 headless real-time voice conversion (no GUI dependencies).

Live (needs PortAudio + a trained checkpoint):
    python realtime_v3.py --config configs/v3/seed_vc_v3_44k_rt.yml --checkpoint runs/dit/v3_dit_final.pth \\
        --reference ref.wav [--input-device N --output-device M] [--no-aec] [--no-agc]

Simulation over a file (works without audio hardware; ``--echo-gain-db`` adds a
synthetic speaker->mic echo path to exercise the echo canceller):
    python realtime_v3.py --simulate input.wav --output out.wav --reference ref.wav --dry-run

Prints the latency budget: algorithmic (block + right context + crossfade +
SOLA search) and, for live streams, the device latency reported by PortAudio.
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

import librosa
import numpy as np
import soundfile as sf

from inference_v3 import pick_device
from modules.v3.model import SeedVCv3
from modules.v3.realtime_io import RealtimeVCStream, simulate_stream


def synthetic_echo_path(sr, delay_ms=30.0, tail_ms=120.0, rt60=0.15, gain_db=-6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(tail_ms * sr / 1000)
    h = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / sr / rt60)
    h *= 10 ** (gain_db / 20) / np.sqrt(np.sum(h ** 2))
    return np.concatenate([np.zeros(int(delay_ms * sr / 1000)), h])


def device_arg(value):
    """sounddevice takes an int index or a name substring."""
    return int(value) if value.isdigit() else value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v3/seed_vc_v3_44k_rt.yml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dry-run", action="store_true", help="allow running without a trained checkpoint")
    p.add_argument("--reference", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--samplerate", type=int, default=None, help="device rate (default: model rate, 44100)")
    p.add_argument("--input-device", type=device_arg, default=None)
    p.add_argument("--output-device", type=device_arg, default=None)
    p.add_argument("--block-time", type=float, default=None)
    p.add_argument("--diffusion-steps", type=int, default=None)
    p.add_argument("--cfg-rate", type=float, default=None)
    p.add_argument("--no-aec", action="store_true")
    p.add_argument("--no-agc", action="store_true")
    p.add_argument("--aec-delay-ms", type=float, default=0.0, help="far-end bulk delay compensation")
    p.add_argument("--mute-when-silent", action="store_true", help="output silence while the input gate is closed")
    p.add_argument("--duration", type=float, default=None, help="stop a live stream after N seconds")
    p.add_argument("--simulate", default=None, help="input wav to stream through the pipeline instead of a device")
    p.add_argument("--output", default="reconstructed/realtime_v3_sim.wav")
    p.add_argument("--echo-gain-db", type=float, default=None, help="simulation: add a synthetic echo path")
    p.add_argument("--max-seconds", type=float, default=None, help="simulation: truncate the input")
    args = p.parse_args(argv)

    if not args.checkpoint and not args.dry_run:
        p.error("no v3 checkpoint given; pass --checkpoint or --dry-run")
    device = pick_device(args.device)
    model = SeedVCv3(args.config)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    else:
        print("WARNING: no trained v3 checkpoint; output is not meaningful audio (dry run)")
    model.prepare(device).eval()
    overrides = {k: v for k, v in {"block_time": args.block_time, "diffusion_steps": args.diffusion_steps,
                                   "cfg_rate": args.cfg_rate}.items() if v is not None}
    engine = model.streaming_engine(**overrides)
    engine.set_reference(librosa.load(args.reference, sr=model.sr)[0])
    stream = RealtimeVCStream(engine, samplerate=args.samplerate, input_device=args.input_device,
                              output_device=args.output_device, aec=not args.no_aec, agc=not args.no_agc,
                              mute_when_gated=args.mute_when_silent, aec_kwargs={"bulk_delay_ms": args.aec_delay_ms})

    if args.simulate:
        wave = librosa.load(args.simulate, sr=stream.sr)[0]
        if args.max_seconds:
            wave = wave[: int(args.max_seconds * stream.sr)]
        echo = synthetic_echo_path(stream.sr, gain_db=args.echo_gain_db) if args.echo_gain_db is not None else None
        engine.warmup()
        out, block_seconds = simulate_stream(stream, wave, echo_ir=echo)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        sf.write(args.output, out, stream.sr)
        report = engine.latency_report()
        report["simulated_blocks"] = len(block_seconds)
        report["chain_block_mean_ms"] = float(1000 * block_seconds.mean()) if len(block_seconds) else None
        report["resampler_delay_ms"] = stream.resampler_delay_ms
        print(json.dumps(report, indent=2))
        print("saved", args.output)
        return report

    stream.run(duration=args.duration)


if __name__ == "__main__":
    main()
