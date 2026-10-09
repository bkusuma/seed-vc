"""Dataset for Seed-VC v3 training: 44.1 kHz audio, 128-band mels, optional transcripts.

Transcripts (for the bottleneck's CTC ASR loss) are read from side-car files next
to each audio file, first match wins: ``<name>.normalized.txt`` (LibriTTS),
``<name>.original.txt``, ``<name>.txt``, ``<name>.lab``. Missing transcripts are
returned as "" and simply contribute no CTC loss. ``tools/pseudo_label_v3.py``
writes ``<name>.txt`` with Whisper-large-v3-turbo for unlabelled data.
"""
import os

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modules.v3.audio import MelSpectrogram

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")
TRANSCRIPT_SUFFIXES = (".normalized.txt", ".original.txt", ".txt", ".lab")
MEL_PAD_VALUE = -10.0  # same padding value as v1 (log(1e-5) = -11.5 is the floor)


def find_transcript(audio_path):
    base = os.path.splitext(audio_path)[0]
    for suffix in TRANSCRIPT_SUFFIXES:
        path = base + suffix
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
    return ""


class V3Dataset(Dataset):
    def __init__(self, data_dir, sr, spect_params, min_duration=1.0, max_duration=30.0, require_transcript=False):
        self.files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files
            if f.lower().endswith(AUDIO_EXTENSIONS)
        )
        if require_transcript:
            self.files = [f for f in self.files if find_transcript(f)]
        if not self.files:
            raise ValueError(f"no usable audio files under {data_dir}")
        self.sr = int(sr)
        self.min_samples = int(min_duration * self.sr)
        self.max_samples = int(max_duration * self.sr)
        self.mel_fn = MelSpectrogram(self.sr, spect_params["n_fft"], spect_params["win_length"],
                                     spect_params["hop_length"], spect_params["n_mels"],
                                     spect_params.get("fmin", 0), spect_params.get("fmax"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Fall back to the next file on unreadable / out-of-range audio (bounded attempts).
        for attempt in range(len(self.files)):
            path = self.files[(idx + attempt) % len(self.files)]
            try:
                wave, _ = librosa.load(path, sr=self.sr, mono=True)
            except Exception as e:  # corrupt file, unsupported codec, ...
                print(f"[v3_dataset] skipping {path}: {e}")
                continue
            if self.min_samples <= len(wave) <= self.max_samples:
                break
        else:
            raise RuntimeError("no file in the dataset satisfies the duration limits")
        wave = torch.from_numpy(np.clip(wave, -1.0, 1.0)).float()
        mel = self.mel_fn(wave[None])[0]
        return wave, mel, find_transcript(path)


def collate_v3(batch):
    batch = sorted(batch, key=lambda b: b[1].size(1), reverse=True)
    n_mels = batch[0][1].size(0)
    max_mel = batch[0][1].size(1)
    max_wave = max(b[0].numel() for b in batch)
    waves = torch.zeros(len(batch), max_wave)
    mels = torch.full((len(batch), n_mels, max_mel), MEL_PAD_VALUE)
    wave_lens = torch.zeros(len(batch), dtype=torch.long)
    mel_lens = torch.zeros(len(batch), dtype=torch.long)
    texts = []
    for i, (wave, mel, text) in enumerate(batch):
        waves[i, : wave.numel()] = wave
        mels[i, :, : mel.size(1)] = mel
        wave_lens[i] = wave.numel()
        mel_lens[i] = mel.size(1)
        texts.append(text)
    return {"waves": waves, "mels": mels, "wave_lens": wave_lens, "mel_lens": mel_lens, "texts": texts}


def build_v3_dataloader(data_dir, sr, spect_params, batch_size=4, num_workers=0, shuffle=True, **dataset_kwargs):
    dataset = V3Dataset(data_dir, sr, spect_params, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_v3, drop_last=len(dataset) >= batch_size)
