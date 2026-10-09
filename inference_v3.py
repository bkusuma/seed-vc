"""Seed-VC v3 offline voice conversion.

    python inference_v3.py --source src.wav --target ref.wav --checkpoint runs/dit/v3_dit_final.pth

No released checkpoint exists for v3 yet. ``--dry-run`` runs the full pipeline
with the real encoder/vocoder/style encoder and randomly initialised trainable
modules (optionally warm-started with ``--init-from-v1``); the output audio is
noise, but every tensor shape and the I/O path are exercised.
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

import librosa
import soundfile as sf
import torch

from modules.v3.model import SeedVCv3


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="./examples/source/source_s1.wav")
    p.add_argument("--target", default="./examples/reference/s1p1.wav")
    p.add_argument("--output", default="./reconstructed")
    p.add_argument("--config", default="configs/v3/seed_vc_v3_44k.yml")
    p.add_argument("--checkpoint", default=None, help="v3 checkpoint (train_v3.py output)")
    p.add_argument("--init-from-v1", default=None, help="v1 checkpoint path for a warm-started dry run")
    p.add_argument("--dry-run", action="store_true", help="allow running without a trained v3 checkpoint")
    p.add_argument("--diffusion-steps", type=int, default=None)
    p.add_argument("--inference-cfg-rate", type=float, default=None)
    p.add_argument("--length-adjust", type=float, default=1.0)
    p.add_argument("--attn-mode", choices=("full", "prompt_isolated"), default=None)
    p.add_argument("--max-source-seconds", type=float, default=None, help="truncate the source (quick tests)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if not args.checkpoint and not args.dry_run:
        p.error("no v3 checkpoint given; train one with train_v3.py or pass --dry-run")
    device = pick_device(args.device)
    model = SeedVCv3(args.config)
    if args.checkpoint:
        report = model.load_checkpoint(args.checkpoint)
        print("loaded:", model.summarize_report(report))
        if "cfm" not in report:
            p.error(f"{args.checkpoint} has no DiT weights (a stage-A bottleneck checkpoint?); use a stage-B checkpoint")
    elif args.init_from_v1:
        print("warm start:", {k: f"{r['loaded_params'] / 1e6:.1f}M/{r['total_params'] / 1e6:.1f}M params"
                              for k, r in model.warm_start_from_v1(args.init_from_v1).items()})
    if not args.checkpoint:
        print("WARNING: no trained v3 checkpoint; output is not meaningful audio (dry run)")
    model.prepare(device).eval()

    source = librosa.load(args.source, sr=model.sr)[0]
    if args.max_source_seconds:
        source = source[: int(args.max_source_seconds * model.sr)]
    reference = librosa.load(args.target, sr=model.sr)[0]
    start = time.time()
    wave, info = model.convert(source, reference, args.diffusion_steps, args.inference_cfg_rate, args.length_adjust,
                               args.attn_mode, seed=args.seed)
    elapsed = time.time() - start
    print("shapes:", info)
    print(f"RTF: {elapsed / (len(wave) / model.sr):.3f} ({elapsed:.1f}s for {len(wave) / model.sr:.1f}s)")
    os.makedirs(args.output, exist_ok=True)
    name = f"vc_v3_{os.path.splitext(os.path.basename(args.source))[0]}_{os.path.splitext(os.path.basename(args.target))[0]}.wav"
    out_path = os.path.join(args.output, name)
    sf.write(out_path, wave, model.sr)
    print("saved", out_path)
    return out_path


if __name__ == "__main__":
    main()
