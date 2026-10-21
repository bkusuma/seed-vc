import torch

from modules.v3.bottleneck import ByteTokenizer, ContentBottleneck, bsq_entropy_loss
from modules.v3.length_regulator import TimeAlignedRegulator, time_alignment_index

MEL_RATE = 44100 / 256


def test_alignment_index_nearest_centre():
    idx = time_alignment_index(10, MEL_RATE, 50.0, n_in=100)
    # mel frame m centre (m + 0.5) * 256 / 44100 s; nearest 20 ms content frame
    expected = [round(((m + 0.5) * 256 / 44100) * 50) for m in range(10)]
    assert idx.tolist() == expected


def test_window_matches_offline_interior():
    """A window cut from a stream (absolute-time alignment) reproduces the offline regulation."""
    torch.manual_seed(0)
    reg = TimeAlignedRegulator(channels=16, sampling_ratios=(1, 1), is_discrete=True, codebook_size=32,
                               mel_rate=MEL_RATE).eval()
    content = torch.randint(0, 32, (1, 250))  # 5 s at 50 Hz
    n_mel = int(5 * MEL_RATE)
    with torch.no_grad():
        full, _ = reg(content, torch.tensor([n_mel]))
        # mel window [300, 420) = 1.74-2.44 s; content window [80, 230) = 1.6-4.6 s covers it,
        # as in the streaming engine (content left context >= DiT left context)
        g0, n_win, c0 = 300, 120, 80
        win, _ = reg(content[:, c0:c0 + 150], torch.tensor([n_win]), mel_start=g0 / MEL_RATE, content_start=c0 / 50.0)
    k = reg.receptive_half_width
    assert torch.allclose(win[:, k:n_win - k], full[:, g0 + k:g0 + n_win - k], atol=1e-6)


def test_groupnorm_regulator_is_window_dependent():
    """The v1-style sequence-wide GroupNorm breaks window equivalence (why v3 uses frame-local norm)."""
    torch.manual_seed(0)
    reg = TimeAlignedRegulator(channels=16, sampling_ratios=(1, 1), is_discrete=True, codebook_size=32,
                               mel_rate=MEL_RATE, norm="group").eval()
    content = torch.randint(0, 32, (1, 250))
    with torch.no_grad():
        full, _ = reg(content, torch.tensor([int(5 * MEL_RATE)]))
        win, _ = reg(content[:, 100:250], torch.tensor([120]), mel_start=300 / MEL_RATE, content_start=2.0)
    assert not torch.allclose(win[:, 2:118], full[:, 302:418], atol=1e-3)


def test_length_adjust_stretches_timeline():
    idx_slow = time_alignment_index(200, MEL_RATE, 50.0, n_in=1000, time_scale=0.5)
    idx = time_alignment_index(200, MEL_RATE, 50.0, n_in=1000)
    assert idx_slow[-1] < idx[-1]


def bottleneck(**kw):
    torch.manual_seed(0)
    return ContentBottleneck(input_dim=24, dim=16, intermediate_dim=32, num_blocks=2, codebook_size=64,
                             asr_blocks=1, recon_blocks=1, speaker_dim=8, **kw)


def test_bottleneck_shapes_and_range():
    bn = bottleneck().eval()
    out = bn(torch.randn(2, 30, 24), torch.tensor([30, 20]))
    assert out["indices"].shape == (2, 30) and out["quantized"].shape == (2, 30, 16)
    assert out["indices"].min() >= 0 and out["indices"].max() < 64
    assert out["u"].shape == (2, 30, 6) and not out["mask"][1, 20:].any()


def test_bottleneck_is_causal():
    bn = bottleneck().eval()
    feats = torch.randn(1, 40, 24)
    altered = feats.clone()
    altered[:, 25:] = torch.randn(1, 15, 24)
    with torch.no_grad():
        a, b = bn(feats)["indices"], bn(altered)["indices"]
    assert torch.equal(a[:, :25], b[:, :25])


def test_bottleneck_losses_backprop():
    bn = bottleneck().train()
    feats, lens = torch.randn(2, 50, 24), torch.tensor([50, 40])
    out = bn(feats, lens)
    tok = ByteTokenizer()
    texts = ["hello world", "seed vc"]
    targets = torch.tensor([b for t in texts for b in tok.encode(t)])
    target_lens = torch.tensor([len(tok.encode(t)) for t in texts])
    assert (bn.asr_vocab_size, bn.asr_blank_id) == (tok.vocab_size, tok.blank_id)
    ctc = bn.ctc_loss(out, lens, targets, target_lens)
    recon = bn.recon_loss(out, torch.randn(2, 8), feats)
    ent = bn.entropy_loss(out)
    total = ctc + recon + 0.1 * ent
    total.backward()
    assert all(torch.isfinite(v) for v in (ctc, recon, ent))
    assert bn.encoder.input_projection.weight.grad is not None  # straight-through gradient reaches the encoder
    decoded = bn.ctc_greedy_decode(out, lens)
    assert len(decoded) == 2 and all(isinstance(i, int) for i in decoded[0])


def test_entropy_loss_prefers_balanced_confident_codes():
    torch.manual_seed(0)
    collapsed = torch.nn.functional.normalize(torch.ones(500, 6) + 0.01 * torch.randn(500, 6), dim=-1)
    spread = torch.nn.functional.normalize(torch.sign(torch.randn(500, 6)), dim=-1)
    assert bsq_entropy_loss(spread, inv_temperature=10.0) < bsq_entropy_loss(collapsed, inv_temperature=10.0)


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    ids = tok.encode("  Héllo   WORLD ")
    assert max(ids) < 256 and tok.decode(ids) == "héllo world"
    assert tok.vocab_size == 257 and tok.blank_id == 256


def test_strip_training_heads():
    bn = bottleneck().strip_training_heads()
    assert bn.asr_head is None and bn.recon_decoder is None
    assert not any(k.startswith(("asr_head", "recon_decoder")) for k in bn.state_dict())
