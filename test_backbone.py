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


def test_frozen_backbone_stays_in_eval_mode():
    """SB3 calls .train() on the policy every update.

    Without the train() override, BatchNorm running stats inside a 'frozen'
    backbone keep updating and its features drift.
    """
    space, _ = _space_and_obs()
    ext = ResNetExtractor(space, finetune=False)
    ext.train(True)
    assert not ext.backbone.training, "frozen backbone was put back into train mode"
    print("test_frozen_backbone_stays_in_eval_mode PASSED")


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
    test_frozen_backbone_stays_in_eval_mode()
    test_channels_are_temporal_not_rgb()
    test_output_is_finite()
    print("\nall backbone tests passed")
