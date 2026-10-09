"""Seed-VC v3 training.

Stage A (``--stage bottleneck``): train the BSQ content bottleneck on frozen
Whisper features with CTC ASR loss + speaker-conditioned reconstruction +
entropy loss. Needs transcripts (side-car .txt/.lab, or tools/pseudo_label_v3.py)
for the CTC term.

Stage B (``--stage dit``): freeze encoder + bottleneck, train the regulator and
the DiT with mixed-mask flow matching on 44.1 kHz / 128-band mels. Pass the
stage A checkpoint with ``--bottleneck-ckpt``; ``--init-from-v1`` warm-starts the
DiT from a v1 checkpoint (local path or a file name in Plachta/Seed-VC).

Example:
    python train_v3.py --config configs/v3/seed_vc_v3_44k.yml --dataset-dir data/ --run-name bn --stage bottleneck
    python train_v3.py --config configs/v3/seed_vc_v3_44k.yml --dataset-dir data/ --run-name dit --stage dit \\
        --bottleneck-ckpt runs/bn/v3_bottleneck_final.pth \\
        --init-from-v1 DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
"""
import argparse
import glob
import os
import random
import shutil
import sys

os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

import torch
import torchaudio
from tqdm import tqdm

from data.v3_dataset import build_v3_dataloader
from modules.v3.content_encoder import SAMPLE_RATE as ENCODER_SR
from modules.v3.model import SeedVCv3, load_v3_config
from optimizers import build_optimizer

STAGES = ("bottleneck", "dit")


class OpenVoicePerturber:
    """Timbre perturbation used by v1 training: convert each utterance to a random
    reference speaker with the OpenVoice v2 tone-color converter (22.05 kHz)."""

    SR = 22050

    def __init__(self, device):
        from hf_utils import load_custom_model_from_hf
        from modules.openvoice.api import ToneColorConverter

        ckpt, cfg = load_custom_model_from_hf("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
        self.converter = ToneColorConverter(cfg, device=str(device))
        self.converter.load_ckpt(ckpt)
        self.converter.model.eval()
        self.se_db = torch.load(load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None), map_location="cpu")
        self.device = device

    @torch.no_grad()
    def __call__(self, waves, wave_lens, sr):
        w22 = torchaudio.functional.resample(waves, sr, self.SR)
        l22 = (wave_lens.float() * self.SR / sr).long()
        src_se = self.converter.extract_se(w22, l22)
        tgt_se = self.se_db[torch.randint(0, len(self.se_db), (waves.size(0),))].to(self.device)
        converted = self.converter.convert(w22, l22, src_se, tgt_se).squeeze(1)
        return torchaudio.functional.resample(converted, self.SR, sr)[:, : waves.size(1)]


class V3Trainer:
    def __init__(self, config_path, dataset_dir, run_name, stage, batch_size=0, num_workers=0, max_steps=10000,
                 save_every=1000, log_every=10, device="cuda", lr=None, pretrained_ckpt=None, bottleneck_ckpt=None,
                 init_from_v1=None, perturb=None, grad_clip=10.0, encoder_pretrained=True, encoder_config=None):
        if stage not in STAGES:
            raise ValueError(f"--stage must be one of {STAGES}")
        self.stage = stage
        self.device = torch.device(device)
        cfg = load_v3_config(config_path)
        self.cfg = cfg
        train_cfg = cfg.get("training") or {}
        self.log_dir = os.path.join(cfg.get("log_dir", "./runs"), run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        if isinstance(config_path, (str, os.PathLike)):
            shutil.copyfile(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))
        self.max_steps, self.save_every, self.log_every, self.grad_clip = max_steps, save_every, log_every, grad_clip
        losses = cfg.get("loss_params") or {}
        self.lambda_ctc = float(losses.get("lambda_ctc", 1.0))
        self.lambda_recon = float(losses.get("lambda_recon", 1.0))
        self.lambda_entropy = float(losses.get("lambda_entropy", 0.1))
        lr = float(lr or losses.get("base_lr", 1e-4))

        self.model = SeedVCv3(cfg, load_vocoder=False, with_training_heads=stage == "bottleneck",
                              encoder_pretrained=encoder_pretrained, encoder_config=encoder_config).prepare(self.device)
        if stage == "bottleneck" and self.model.bottleneck is None:
            raise ValueError("config has bottleneck.enabled: false; nothing to train in stage 'bottleneck'")
        self.model.requires_grad_(False)
        self.model.eval()  # frozen parts (encoder, style encoder, stage-B bottleneck) stay in eval mode
        self.trainable = {"bottleneck": self.model.bottleneck} if stage == "bottleneck" else \
            {"length_regulator": self.model.length_regulator, "cfm": self.model.cfm}
        for module in self.trainable.values():
            module.requires_grad_(True)
        self.optimizer = build_optimizer(self.trainable, lr=lr)

        pp = cfg.preprocess_params
        self.loader = build_v3_dataloader(
            dataset_dir, pp.sr, pp.spect_params, batch_size=batch_size or int(train_cfg.get("batch_size", 4)),
            num_workers=num_workers, min_duration=float(train_cfg.get("min_duration", 1.0)),
            max_duration=float(train_cfg.get("max_duration", 20.0)),
        )
        perturb = perturb if perturb is not None else train_cfg.get("perturb", "none")
        self.perturb = OpenVoicePerturber(self.device) if perturb == "openvoice" else None

        self.epoch, self.iters = 0, 0
        self._load_initial_weights(pretrained_ckpt, bottleneck_ckpt, init_from_v1)

    # -------------------------------------------------------------- weights
    def _ckpt_glob(self):
        return sorted(glob.glob(os.path.join(self.log_dir, f"v3_{self.stage}_step_*.pth")),
                      key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))

    def _load_initial_weights(self, pretrained_ckpt, bottleneck_ckpt, init_from_v1):
        resume = self._ckpt_glob()
        if bottleneck_ckpt:
            report = self.model.load_checkpoint(bottleneck_ckpt, keys=("bottleneck",))
            if "bottleneck" not in report:
                raise ValueError(f"{bottleneck_ckpt} has no 'bottleneck' weights")
            print("bottleneck:", self.model.summarize_report(report)["bottleneck"])
        if init_from_v1:
            path = init_from_v1
            if not os.path.isfile(path):
                from hf_utils import load_custom_model_from_hf

                path = load_custom_model_from_hf("Plachta/Seed-VC", init_from_v1, None)
            for key, rep in self.model.warm_start_from_v1(path).items():
                print(f"warm start {key}: {rep['loaded_params'] / 1e6:.1f}M / {rep['total_params'] / 1e6:.1f}M params "
                      f"({len(rep['shape_mismatch'])} shape mismatches)")
        if pretrained_ckpt:
            print("pretrained:", self.model.summarize_report(self.model.load_checkpoint(pretrained_ckpt)))
        if resume:
            state = torch.load(resume[-1], map_location="cpu")
            self.model.load_checkpoint(resume[-1], keys=self.save_keys)
            self.optimizer.load_state_dict(state["optimizer"])
            self.optimizer.load_scheduler_state_dict(state["scheduler"])
            self.epoch, self.iters = state.get("epoch", 0), state.get("iters", 0)
            print(f"resumed from {resume[-1]} at step {self.iters}")

    @property
    def save_keys(self):
        # Stage A saves only the bottleneck. Stage B also stores the (frozen) bottleneck, so its
        # checkpoint is self-contained for inference.
        return ("bottleneck",) if self.stage == "bottleneck" else ("bottleneck", "length_regulator", "cfm")

    def save(self, final=False):
        name = f"v3_{self.stage}_final.pth" if final else f"v3_{self.stage}_step_{self.iters:07d}.pth"
        path = os.path.join(self.log_dir, name)
        extra = {} if final else {"optimizer": self.optimizer.state_dict(),
                                  "scheduler": self.optimizer.scheduler_state_dict(),
                                  "iters": self.iters, "epoch": self.epoch}
        self.model.save_checkpoint(path, keys=self.save_keys, stage=self.stage, **extra)
        for old in self._ckpt_glob()[:-2]:
            os.remove(old)
        return path

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def _to_16k(self, waves, wave_lens):
        sr = self.model.sr
        w16 = torchaudio.functional.resample(waves, sr, ENCODER_SR)
        return w16, (wave_lens.float() * ENCODER_SR / sr).long().clamp(max=w16.size(1))

    @torch.no_grad()
    def _encoder_inputs(self, batch):
        waves, wave_lens = batch["waves"], batch["wave_lens"]
        w16, l16 = self._to_16k(waves, wave_lens)
        feats_ori, feat_lens = self.model.extract_content(w16, l16)
        if self.perturb is not None:
            w_alt = self.perturb(waves, wave_lens, self.model.sr)
            w16_alt, _ = self._to_16k(w_alt, wave_lens)
            feats_alt, _ = self.model.extract_content(w16_alt, l16)
        else:
            feats_alt = feats_ori
        style = self.model.compute_style(w16, l16)
        return feats_ori, feats_alt, feat_lens, style

    # ---------------------------------------------------------------- steps
    def step_bottleneck(self, batch):
        bn = self.model.bottleneck
        feats_ori, feats_alt, feat_lens, style = self._encoder_inputs(batch)
        out = bn(feats_alt, feat_lens)
        loss_entropy = bn.entropy_loss(out)
        loss_recon = bn.recon_loss(out, style, feats_ori)
        tok = self.model.text_tokenizer
        targets = [tok.encode(t) if t else [] for t in batch["texts"]]
        keep = [i for i, t in enumerate(targets) if t]
        if keep and self.lambda_ctc > 0:
            sub = {k: v[keep] for k, v in out.items()}
            flat = torch.tensor([x for i in keep for x in targets[i]], dtype=torch.long, device=self.device)
            lens = torch.tensor([len(targets[i]) for i in keep], dtype=torch.long, device=self.device)
            loss_ctc = bn.ctc_loss(sub, feat_lens[keep], flat, lens)
        else:
            loss_ctc = torch.zeros((), device=self.device)
        loss = self.lambda_ctc * loss_ctc + self.lambda_recon * loss_recon + self.lambda_entropy * loss_entropy
        self._backward_step(loss)
        used = out["indices"][out["mask"]]
        return {"loss": loss.item(), "ctc": loss_ctc.item(), "recon": loss_recon.item(),
                "entropy": loss_entropy.item(), "codes_used": int(used.unique().numel()),
                "ctc_items": len(keep)}

    def step_dit(self, batch):
        m = self.model
        mels, mel_lens = batch["mels"], batch["mel_lens"]
        feats_ori, feats_alt, feat_lens, style = self._encoder_inputs(batch)
        with torch.no_grad():
            tok_ori = m.content_tokens(feats_ori, feat_lens)
            tok_alt = tok_ori if feats_alt is feats_ori else m.content_tokens(feats_alt, feat_lens)
        cond_alt = m.regulate(tok_alt, feat_lens, mel_lens)
        cond_ori = cond_alt if tok_alt is tok_ori else m.regulate(tok_ori, feat_lens, mel_lens)
        B = mels.size(0)
        prompt_lens = (torch.rand(B, device=self.device) * (mel_lens - 1).float()).floor().long()
        prompt_lens[torch.rand(B, device=self.device) < 0.1] = 0
        # the prompt region takes the content of the *original* audio, as in v1 training
        in_prompt = torch.arange(cond_alt.size(1), device=self.device)[None, :, None] < prompt_lens[:, None, None]
        cond = torch.where(in_prompt, cond_ori, cond_alt)
        loss, mode, used_lens = m.cfm(mels, mel_lens, prompt_lens, cond, style)
        self._backward_step(loss)
        return {"loss": loss.item(), "attn_mode": mode, "cropped": int((used_lens < mel_lens).sum())}

    def _backward_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for key, module in self.trainable.items():
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)
            self.optimizer.step(key)
            self.optimizer.scheduler(key=key)

    def _move(self, batch):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def train(self):
        step_fn = self.step_bottleneck if self.stage == "bottleneck" else self.step_dit
        for module in self.trainable.values():
            module.train()
        ema = None
        while self.iters < self.max_steps:
            for batch in tqdm(self.loader, desc=f"epoch {self.epoch}"):
                metrics = step_fn(self._move(batch))
                ema = metrics["loss"] if ema is None else 0.99 * ema + 0.01 * metrics["loss"]
                self.iters += 1
                if self.iters % self.log_every == 0:
                    print(f"step {self.iters} loss(ema) {ema:.4f} {metrics}")
                if self.iters % self.save_every == 0:
                    print("saved", self.save())
                if self.iters >= self.max_steps:
                    break
            self.epoch += 1
        path = self.save(final=True)
        print(f"final model saved at {path}")
        return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/v3/seed_vc_v3_44k.yml")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--batch-size", type=int, default=0, help="0 = training.batch_size from the config")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained-ckpt", default=None, help="v3 checkpoint to start from")
    p.add_argument("--bottleneck-ckpt", default=None, help="stage A checkpoint (required for a useful stage B)")
    p.add_argument("--init-from-v1", default=None, help="v1 checkpoint path or file name in Plachta/Seed-VC")
    p.add_argument("--perturb", choices=("openvoice", "none"), default=None, help="override training.perturb")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage == "dit" and not args.bottleneck_ckpt and not args.pretrained_ckpt:
        print("WARNING: training the DiT on an untrained bottleneck; pass --bottleneck-ckpt from stage A")
    trainer = V3Trainer(args.config, args.dataset_dir, args.run_name, args.stage, args.batch_size, args.num_workers,
                        args.max_steps, args.save_every, args.log_every, args.device, args.lr, args.pretrained_ckpt,
                        args.bottleneck_ckpt, args.init_from_v1, args.perturb)
    trainer.train()


if __name__ == "__main__":
    if sys.platform == "win32":
        import torch.multiprocessing as mp

        mp.freeze_support()
        mp.set_start_method("spawn", force=True)
    main()
