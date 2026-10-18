"""Assertion-based checks for the agent extractor. No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from env import TEXT_TOKENS, TEXT_VOCAB, EmeraldEnv, tokenize_text  # noqa: E402
from extractor import CNN_FEATURES, TEXT_EMBED, AgentExtractor  # noqa: E402


def _space():
    env = EmeraldEnv()
    space = env.observation_space
    env.close()
    return space


def test_tokenizer_is_stable_padded_and_in_range():
    a = tokenize_text("Wild ZIGZAGOON appeared!")
    b = tokenize_text("wild zigzagoon APPEARED")
    assert a.shape == (TEXT_TOKENS,) and a.dtype == np.int64
    assert (a == b).all(), "tokens must not depend on case or punctuation"
    assert a[3:].sum() == 0, "unused slots must be zero padding"
    assert a[:3].min() >= 1 and a.max() < TEXT_VOCAB
    assert tokenize_text("").sum() == 0
    long = tokenize_text(" ".join(f"w{i}" for i in range(40)))
    assert long.shape == (TEXT_TOKENS,) and (long != 0).all(), "overflow must truncate, not grow"
    assert not (tokenize_text("Got away safely") == tokenize_text("TORCHIC fainted")).all()
    print("test_tokenizer_is_stable_padded_and_in_range PASSED")


def test_features_dim_matches_output():
    space = _space()
    ext = AgentExtractor(space, n_actions=8)
    obs = {
        "screen": torch.rand(2, 3, 80, 120),
        "state": torch.zeros(2, space["state"].shape[0]),
        "text": torch.as_tensor(np.stack([tokenize_text("hello world"), tokenize_text("")])).float(),
    }
    out = ext(obs)
    assert out.shape == (2, ext.features_dim), f"{out.shape} vs {ext.features_dim}"
    assert ext.features_dim == CNN_FEATURES + space["state"].shape[0] + TEXT_EMBED
    assert torch.isfinite(out).all()
    print("test_features_dim_matches_output PASSED")


def test_empty_text_is_a_zero_vector():
    """Padding must read as 'no text', not as a learned word."""
    space = _space()
    ext = AgentExtractor(space)
    words = ext.text(torch.as_tensor(tokenize_text("")).unsqueeze(0))
    assert words.abs().max().item() == 0.0, "empty text produced a non-zero embedding"
    some = ext.text(torch.as_tensor(tokenize_text("Got away safely")).unsqueeze(0))
    assert some.abs().max().item() > 0.0
    print("test_empty_text_is_a_zero_vector PASSED")


def test_inverse_dynamics_head_learns_from_screen_pairs():
    """Synthetic pairs: the second screen is the first shifted by the action's
    direction. The head must beat chance by a wide margin after a few hundred
    steps, with gradients reaching the shared CNN."""
    torch.manual_seed(0)
    space = _space()
    ext = AgentExtractor(space, n_actions=8)
    opt = torch.optim.Adam(list(ext.cnn.parameters()) + list(ext.idm.parameters()), lr=1e-3)

    def batch(n=64):
        s0 = torch.rand(n, 3, 80, 120)
        a = torch.randint(0, 8, (n,))
        s1 = s0.clone()
        for i in range(n):
            k = int(a[i])
            if k in (1, 2):       # up / down: vertical shift
                s1[i] = torch.roll(s0[i], shifts=(-4 if k == 1 else 4), dims=1)
            elif k in (3, 4):     # left / right: horizontal shift
                s1[i] = torch.roll(s0[i], shifts=(-4 if k == 3 else 4), dims=2)
            elif k in (5, 6):     # A / B: brightness jump of distinct sign
                s1[i] = (s0[i] + (0.3 if k == 5 else -0.3)).clamp(0, 1)
            elif k == 7:          # start: invert
                s1[i] = 1.0 - s0[i]
        return s0, s1, a            # action 0 (no-op): identical screens

    before_cnn = [p.detach().clone() for p in ext.cnn.parameters()]
    for _ in range(300):
        s0, s1, a = batch()
        loss = F.cross_entropy(ext.idm_logits(s0, s1), a)
        opt.zero_grad(); loss.backward(); opt.step()
    s0, s1, a = batch(256)
    with torch.no_grad():
        acc = (ext.idm_logits(s0, s1).argmax(1) == a).float().mean().item()
    assert acc > 0.3, f"inverse dynamics accuracy {acc:.2f}; chance is 0.125"
    moved = any((p.detach() - q).abs().max() > 0 for p, q in zip(ext.cnn.parameters(), before_cnn))
    assert moved, "gradients did not reach the shared CNN"
    print(f"test_inverse_dynamics_head_learns_from_screen_pairs PASSED (acc {acc:.2f})")


if __name__ == "__main__":
    test_tokenizer_is_stable_padded_and_in_range()
    test_features_dim_matches_output()
    test_empty_text_is_a_zero_vector()
    test_inverse_dynamics_head_learns_from_screen_pairs()
    print("\nall extractor tests passed")
