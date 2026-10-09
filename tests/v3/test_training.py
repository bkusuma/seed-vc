import numpy as np
import soundfile as sf
import torch

from conftest import TINY_WHISPER, tiny_config
from train_v3 import V3Trainer


def make_dataset(tmp_path, sr=22050, n=3):
    rng = np.random.default_rng(0)
    for i in range(n):
        t = np.arange(int(sr * (1.0 + 0.3 * i))) / sr
        wave = 0.3 * np.sin(2 * np.pi * (180 + 40 * i) * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        sf.write(tmp_path / f"utt{i}.wav", (wave + 0.01 * rng.standard_normal(len(t))).astype(np.float32), sr)
        if i < n - 1:  # one file without transcript: CTC must skip it
            (tmp_path / f"utt{i}.txt").write_text(f"hello number {i}\n")
    return tmp_path


def trainer(tmp_path, stage, **kw):
    cfg = tiny_config()
    cfg["log_dir"] = str(tmp_path / "runs")
    return V3Trainer(cfg, str(make_dataset(tmp_path)), "t", stage, batch_size=3, max_steps=2, save_every=1,
                     device="cpu", encoder_pretrained=False, encoder_config=TINY_WHISPER, perturb="none", **kw)


def test_bottleneck_stage_step_and_resume(tmp_path):
    torch.manual_seed(0)
    tr = trainer(tmp_path, "bottleneck")
    before = tr.model.bottleneck.encoder.input_projection.weight.detach().clone()
    batch = tr._move(next(iter(tr.loader)))
    metrics = tr.step_bottleneck(batch)
    assert all(np.isfinite(metrics[k]) for k in ("loss", "ctc", "recon", "entropy"))
    assert metrics["ctc_items"] == 2 and metrics["ctc"] > 0
    assert not torch.equal(before, tr.model.bottleneck.encoder.input_projection.weight)
    final = tr.train()
    resumed = trainer(tmp_path, "bottleneck")
    assert resumed.iters == 2
    report = resumed.model.load_checkpoint(final, keys=("bottleneck",))
    assert report["bottleneck"]["loaded_tensors"] == report["bottleneck"]["total_tensors"]


def test_dit_stage_step(tmp_path):
    torch.manual_seed(0)
    tr = trainer(tmp_path, "dit")
    frozen = tr.model.bottleneck.encoder.input_projection.weight.detach().clone()
    batch = tr._move(next(iter(tr.loader)))
    metrics = [tr.step_dit(batch) for _ in range(3)]
    assert all(np.isfinite(m["loss"]) for m in metrics)
    assert {m["attn_mode"] for m in metrics} <= {"full", "prompt_isolated"}
    assert torch.equal(frozen, tr.model.bottleneck.encoder.input_projection.weight)  # bottleneck frozen in stage B
