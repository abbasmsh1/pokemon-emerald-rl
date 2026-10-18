"""Assertion-based checks for the pretrained backbone. No framework."""

# ponytail: importing torch before any mgba core exists makes libmgba's
# run_frame() spin forever. Creating one core first inoculates the process.
# This block must stay above the torch/backbone imports below.
from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from backbone import RESNET_FEATURES, ResNetExtractor  # noqa: E402
from env import EmeraldEnv  # noqa: E402


def _space_and_obs():
    env = EmeraldEnv()
    obs, _ = env.reset()
    space = env.observation_space
    env.close()
    batched = {k: torch.as_tensor(np.stack([v, v])) for k, v in obs.items()}
    return space, batched


def _real_screens(n: int, stride: int = 25) -> np.ndarray:
    """n distinct in-game screens. Nothing synthetic reproduces the bug this
    guards against, so the emulator cost is unavoidable."""
    env = EmeraldEnv()
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    frames = [obs["screen"].copy()]
    while len(frames) < n:
        for _ in range(stride):
            obs, *_ = env.step(int(rng.integers(env.action_space.n)))
        frames.append(obs["screen"].copy())
    env.close()
    return np.stack(frames)


def test_features_dim_matches_output():
    """SB3 trusts features_dim to size the policy head; a mismatch is silent."""
    space, obs = _space_and_obs()
    ext = ResNetExtractor(space)
    out = ext(obs)
    assert out.shape[0] == 2, f"batch dim lost: {out.shape}"
    assert out.shape[1] == ext.features_dim, (
        f"declared features_dim {ext.features_dim} but produced {out.shape[1]}"
    )
    assert ext.features_dim == RESNET_FEATURES + space["state"].shape[0]
    print("test_features_dim_matches_output PASSED")


def test_frozen_backbone_has_no_gradients():
    space, _ = _space_and_obs()
    ext = ResNetExtractor(space, finetune=False)
    trainable = [n for n, p in ext.backbone.named_parameters() if p.requires_grad]
    assert not trainable, f"{len(trainable)} backbone params still trainable"
    print("test_frozen_backbone_has_no_gradients PASSED")


def test_finetune_flag_unfreezes():
    space, _ = _space_and_obs()
    ext = ResNetExtractor(space, finetune=True)
    trainable = [n for n, p in ext.backbone.named_parameters() if p.requires_grad]
    assert trainable, "finetune=True left the backbone frozen"
    print("test_finetune_flag_unfreezes PASSED")


def test_pretrained_weights_are_actually_pretrained():
    """The failure that cost 82 million steps.

    A checkpoint lineage ran that long carrying RANDOM conv weights underneath
    ImageNet BatchNorm parameters. All 20 conv tensors differed from
    torchvision's by up to 1.03; every BatchNorm weight, bias and running
    statistic matched exactly. Nothing in the run reported it: the backbone was
    frozen, so the numbers never moved and never looked unstable.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    space, _ = _space_and_obs()
    ext = ResNetExtractor(space, finetune=False)

    reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    reference.fc = torch.nn.Identity()
    ours, theirs = ext.backbone.state_dict(), reference.state_dict()

    assert set(ours) == set(theirs), "backbone no longer matches torchvision resnet18"
    worst, worst_key = 0.0, ""
    for key in theirs:
        delta = (ours[key].float() - theirs[key].float()).abs().max().item()
        if delta > worst:
            worst, worst_key = delta, key
    assert worst < 1e-6, (
        f"{worst_key} differs from the pretrained weights by {worst:.5f}; "
        f"the backbone is not the ImageNet one"
    )
    print("test_pretrained_weights_are_actually_pretrained PASSED")


def test_frozen_backbone_still_discriminates_screens():
    """Downstream symptom of the same failure, on real frames.

    ImageNet BatchNorm over random conv outputs drove the pooled features to a
    magnitude of 72 and a pairwise cosine similarity of 0.9991 across entirely
    different screens, while the raw pixels of those screens averaged -0.045.
    A healthy extractor measures about 0.85 and 0.71 on the same input.
    """
    space, _ = _space_and_obs()
    ext = ResNetExtractor(space, finetune=False)
    ext.train(True)   # as SB3 leaves the policy during an update

    # Real frames, not synthetic ones. Checked: flat colour bands and noise
    # both come out fine even with the statistics pinned, so a synthetic input
    # passes this test with the bug fully in place. Only actual Emerald
    # screens reproduce it.
    screens = _real_screens(8)
    with torch.no_grad():
        feats = ext({
            "screen": torch.as_tensor(screens),
            "state": torch.zeros(len(screens), space["state"].shape[0]),
        })[:, :RESNET_FEATURES]

    magnitude = feats.abs().mean().item()
    assert magnitude < 10.0, (
        f"feature magnitude {magnitude:.1f}; the activations are saturating "
        f"(72.1 with the bug, 0.64 without)"
    )

    unit = torch.nn.functional.normalize(feats, dim=1)
    off = (unit @ unit.T)[~torch.eye(len(feats), dtype=bool)]
    assert off.mean().item() < 0.95, (
        f"every screen looks alike to the backbone: mean cos-sim "
        f"{off.mean().item():.4f} (0.9991 with the bug)"
    )
    print(f"test_frozen_backbone_still_discriminates_screens PASSED "
          f"(cos-sim {off.mean().item():.3f}, |f| {magnitude:.2f})")


def test_channels_are_temporal_not_rgb():
    """Pin the documented mismatch so it cannot be forgotten.

    The three screen channels are consecutive frames. If someone later switches
    the observation to true RGB this fails, which is the moment to revisit the
    ImageNet normalisation in backbone.py.
    """
    env = EmeraldEnv()
    obs, _ = env.reset()
    first = obs["screen"]
    for _ in range(6):
        obs, *_ = env.step(5)  # press A
    later = obs["screen"]
    env.close()

    assert first.shape[0] == 3
    assert not np.array_equal(first[0], later[0]), (
        "screen channel 0 never changed; the stack may no longer be temporal"
    )
    print("test_channels_are_temporal_not_rgb PASSED")


def test_output_is_finite():
    space, obs = _space_and_obs()
    ext = ResNetExtractor(space)
    out = ext(obs)
    assert torch.isfinite(out).all(), "non-finite features would poison the policy"
    print("test_output_is_finite PASSED")


if __name__ == "__main__":
    test_features_dim_matches_output()
    test_frozen_backbone_has_no_gradients()
    test_finetune_flag_unfreezes()
    test_pretrained_weights_are_actually_pretrained()
    test_frozen_backbone_still_discriminates_screens()
    test_channels_are_temporal_not_rgb()
    test_output_is_finite()
    print("\nall backbone tests passed")
