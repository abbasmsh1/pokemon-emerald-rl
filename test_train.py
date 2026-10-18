"""Checks for the training-side pieces that touch the environment. No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from env import EmeraldEnv  # noqa: E402
from extractor import AgentExtractor  # noqa: E402
from train import CuriosityWrapper  # noqa: E402


def _wrapped(attach: bool):
    venv = CuriosityWrapper(DummyVecEnv([lambda: EmeraldEnv(), lambda: EmeraldEnv()]))
    obs = venv.reset()
    if attach:
        venv.attach(AgentExtractor(venv.observation_space, n_actions=8), torch.device("cpu"))
    return venv, obs


def test_unattached_wrapper_changes_nothing():
    """Before the model exists, or on a backbone without a forward model, the
    wrapper must be a no-op so the other backbones keep their exact reward."""
    venv, _ = _wrapped(attach=False)
    inner_total = 0.0
    for _ in range(5):
        venv.step_async(np.array([1, 2]))
        _, r, _, _ = venv.step_wait()
        inner_total += float(r.sum())
    assert venv.drain_curiosity() == 0.0, "unattached wrapper booked a bonus"
    venv.close()
    print("test_unattached_wrapper_changes_nothing PASSED")


def test_curiosity_pays_a_bounded_positive_bonus():
    venv, _ = _wrapped(attach=True)
    booked = 0.0
    for i in range(8):
        venv.step_async(np.array([[1, 3, 2, 4][i % 4]] * 2))
        _, r, _, _ = venv.step_wait()
        assert np.isfinite(r).all()
    booked = venv.drain_curiosity()
    assert booked > 0.0, "attached wrapper paid nothing over 8 walking steps"
    assert booked <= 8 * 2 * CuriosityWrapper.BETA * CuriosityWrapper.CAP + 1e-9, (
        f"bonus {booked} exceeds the per-step cap"
    )
    assert venv.drain_curiosity() == 0.0, "drain did not clear"
    venv.close()
    print(f"test_curiosity_pays_a_bounded_positive_bonus PASSED (booked {booked:.4f})")


def test_curiosity_fades_as_the_forward_model_learns():
    """The point of the bonus: a screen the model has learned to predict stops
    paying. Train the forward model on one fixed transition and the error on
    that transition must fall."""
    torch.manual_seed(0)
    space = EmeraldEnv().observation_space
    fe = AgentExtractor(space, n_actions=8)
    s0 = torch.rand(4, 3, 80, 120); s1 = torch.roll(s0, 4, dims=2); a = torch.full((4,), 4)
    before = fe.forward_error(s0, s1, a).mean().item()
    opt = torch.optim.Adam(fe.fwd.parameters(), lr=1e-3)
    for _ in range(100):
        loss = fe.forward_error(s0, s1, a).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = fe.forward_error(s0, s1, a).mean().item()
    assert after < 0.5 * before, f"forward error {before:.4f} -> {after:.4f}; the model is not learning"
    print(f"test_curiosity_fades_as_the_forward_model_learns PASSED ({before:.4f} -> {after:.4f})")


if __name__ == "__main__":
    test_unattached_wrapper_changes_nothing()
    test_curiosity_pays_a_bounded_positive_bonus()
    test_curiosity_fades_as_the_forward_model_learns()
    print("\nall train tests passed")
