import pytest
import torch

from modules.commons import recursive_munch
from modules.diffusion_transformer import DiT
from modules.v3.dit import PrefixKVStore, StreamingDiT, build_attention_mask
from modules.v3.flow_matching import CFMv3, PrefixCachedSampler


def dit_args(time_tok=False, style_tok=False, final="mlp", long_skip=False):
    return recursive_munch(dict(
        dit_type="DiT", reg_loss_type="l1", style_encoder=dict(dim=16),
        DiT=dict(hidden_dim=64, num_heads=4, depth=5, class_dropout_prob=0.1, block_size=8192, in_channels=12,
                 style_condition=True, final_layer_type=final, target="mel", content_dim=32,
                 content_codebook_size=1024, content_type="discrete", f0_condition=False, n_f0_bins=256,
                 content_codebooks=1, is_causal=False, long_skip_connection=long_skip, zero_prompt_speech_token=False,
                 time_as_token=time_tok, style_as_token=style_tok, uvit_skip_connection=True,
                 add_resblock_in_transformer=False),
        wavenet=dict(hidden_dim=64, num_layers=4, kernel_size=5, dilation_rate=1, p_dropout=0.0, style_condition=True),
    ))


def inputs(B=2, C=12, T=40, P=15, D=32, S=16):
    torch.manual_seed(1)
    x = torch.randn(B, C, T)
    prompt_x = torch.zeros(B, C, T)
    prompt_x[..., :P] = torch.randn(B, C, P)
    return x, prompt_x, torch.full((B,), T), torch.rand(B), torch.randn(B, S), torch.randn(B, T, D)


@pytest.mark.parametrize("time_tok,style_tok,final,long_skip", [
    (False, False, "mlp", False), (True, True, "mlp", False), (False, False, "mlp", True), (False, False, "wavenet", True),
])
def test_full_mode_matches_v1(time_tok, style_tok, final, long_skip):
    args = dit_args(time_tok, style_tok, final, long_skip)
    v1, v3 = DiT(args).eval(), StreamingDiT(args).eval()
    v3.load_state_dict(v1.state_dict())
    for m in (v1, v3):
        m.setup_caches(1, 8)
    x, px, lens, t, style, cond = inputs()
    with torch.no_grad():
        assert torch.equal(v1(x, px, lens, t, style, cond), v3(x, px, lens, t, style, cond, attn_mode="full"))


@pytest.mark.parametrize("time_tok,style_tok,long_skip", [(False, False, False), (True, True, False), (False, False, True)])
def test_prefix_cache_is_exact(time_tok, style_tok, long_skip):
    dit = StreamingDiT(dit_args(time_tok, style_tok, "mlp", long_skip)).eval()
    dit.setup_caches(1, 8)
    x, px, _, t, style, cond = inputs(B=1)
    P = 15
    x[..., :P] = 0
    with torch.no_grad():
        full = dit(x, px, torch.tensor([x.size(2)]), t, style, cond, prompt_lens=torch.tensor([P]),
                   attn_mode="prompt_isolated")
        store = PrefixKVStore()
        dit.forward_prefix(px[..., :P], cond[:, :P], style, t, store, step=0)
        window = dit.forward_source(x[..., P:], cond[:, P:], style, t, store, step=0)
    assert torch.allclose(full[..., P:], window, atol=1e-5)


def test_wavenet_head_differs_only_at_window_start():
    dit = StreamingDiT(dit_args(final="wavenet", long_skip=True)).eval()
    dit.setup_caches(1, 8)
    x, px, _, t, style, cond = inputs(B=1)
    P = 15
    x[..., :P] = 0
    with torch.no_grad():
        full = dit(x, px, torch.tensor([x.size(2)]), t, style, cond, prompt_lens=torch.tensor([P]),
                   attn_mode="prompt_isolated")
        store = PrefixKVStore()
        dit.forward_prefix(px[..., :P], cond[:, :P], style, t, store, 0)
        window = dit.forward_source(x[..., P:], cond[:, P:], style, t, store, 0)
    err = (full[..., P:] - window).abs().amax(dim=1)[0]
    rf = 4 * (5 // 2)  # WaveNet layers * half kernel
    assert torch.all(err[rf:] < 1e-5)


def test_prompt_isolation_blocks_source_influence():
    dit = StreamingDiT(dit_args()).eval()
    dit.setup_caches(1, 8)
    x, px, lens, t, style, cond = inputs(B=1)
    P = 15
    x2, cond2 = x.clone(), cond.clone()
    x2[..., P:] += 1.0
    cond2[:, P:] += 1.0
    kw = dict(prompt_lens=torch.tensor([P]), attn_mode="prompt_isolated")
    with torch.no_grad():
        a = dit(x, px, lens[:1], t[:1], style[:1], cond, **kw)
        b = dit(x2, px, lens[:1], t[:1], style[:1], cond2, **kw)
        c = dit(x2, px, lens[:1], t[:1], style[:1], cond2, attn_mode="full")
    assert torch.allclose(a[..., :P], b[..., :P], atol=1e-6)  # prompt outputs unaffected by the source
    assert not torch.allclose(a[..., :P], c[..., :P], atol=1e-3)  # ... unlike full attention


def test_attention_mask_rows_never_empty():
    lens = torch.tensor([10, 6])
    mask = build_attention_mask(lens, 12, prefix_lens=torch.tensor([4, 0]))
    assert mask.shape == (2, 1, 12, 12)
    assert mask.any(-1).all()
    assert not mask[0, 0, 0, 5]  # prefix query cannot see source key
    assert mask[0, 0, 5, 0] and mask[0, 0, 5, 9] and not mask[0, 0, 5, 10]


@pytest.mark.parametrize("cfg_rate", [0.0, 0.7])
def test_cached_sampler_matches_offline(cfg_rate):
    torch.manual_seed(0)
    cfm = CFMv3(dit_args()).eval()
    cfm.estimator.setup_caches(1, 8)
    P, W, C = 20, 30, 12
    prompt, mu, style = torch.randn(1, C, P), torch.randn(1, P + W, 32), torch.randn(1, 16)
    noise = torch.randn(1, C, P + W)
    offline = cfm.inference(mu, torch.tensor([P + W]), prompt, style, 6, inference_cfg_rate=cfg_rate,
                            attn_mode="prompt_isolated", sway_coef=-1.0, noise=noise)
    sampler = PrefixCachedSampler(cfm, prompt, mu[:, :P], style, 6, cfg_rate=cfg_rate, sway_coef=-1.0)
    streamed = sampler.sample(mu[:, P:], noise[..., P:])
    assert torch.allclose(offline[..., P:], streamed, atol=1e-5)
    assert sampler.prefix_len == P


def test_cfm_training_step_all_modes():
    torch.manual_seed(0)
    cfm = CFMv3(dit_args(), crop_prob=1.0, crop_min_frames=4, crop_max_frames=8).train()
    cfm.estimator.setup_caches(1, 8)
    x1, lens, pl = torch.randn(3, 12, 60), torch.tensor([60, 50, 40]), torch.tensor([10, 20, 5])
    mu, style = torch.randn(3, 60, 32), torch.randn(3, 16)
    for mode in ("full", "prompt_isolated"):
        cfm.zero_grad()
        loss, used_mode, used_lens = cfm(x1, lens, pl, mu, style, attn_mode=mode)
        loss.backward()
        assert used_mode == mode and torch.isfinite(loss)
        assert torch.all(used_lens <= pl + 8) and torch.all(used_lens > pl)
        assert cfm.estimator.cond_x_merge_linear.weight.grad is not None
