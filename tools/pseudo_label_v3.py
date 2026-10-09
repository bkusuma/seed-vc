"""Write ASR transcripts next to audio files for the v3 bottleneck's CTC loss.

Uses Whisper-large-v3-turbo's (4-layer) decoder, the part of "turbo" that is
actually fast. Existing side-car transcripts are kept unless --overwrite.

    python tools/pseudo_label_v3.py --dataset-dir data/ [--language en] [--batch-size 8]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

import librosa
import torch

from data.v3_dataset import AUDIO_EXTENSIONS, find_transcript


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--language", default=None, help="force a language (e.g. en); default: auto-detect")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="label at most N files (smoke tests)")
    args = p.parse_args(argv)

    from transformers import pipeline

    cuda = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=args.model, device=0 if cuda else -1,
                   torch_dtype=torch.float16 if cuda else torch.float32)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.dataset_dir) for f in fs
                   if f.lower().endswith(AUDIO_EXTENSIONS))
    todo = [f for f in files if args.overwrite or not find_transcript(f)][: args.limit]
    print(f"{len(files)} audio files, {len(todo)} to transcribe")
    generate_kwargs = {"task": "transcribe"}
    if args.language:
        generate_kwargs["language"] = args.language
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        audio = [{"raw": librosa.load(f, sr=16000)[0], "sampling_rate": 16000} for f in batch]
        results = asr(audio, batch_size=len(batch), generate_kwargs=generate_kwargs, chunk_length_s=30)
        for path, res in zip(batch, results):
            text = res["text"].strip()
            with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"{path}: {text}")


if __name__ == "__main__":
    main()
