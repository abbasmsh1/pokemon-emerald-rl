"""A pretrained ResNet18 feature extractor, as an alternative to NatureCNN.

Two things about this are worth knowing before trusting its output.

The screen observation is three CONSECUTIVE GRAYSCALE FRAMES, not RGB. It fits
ResNet's 3-channel first conv dimensionally, but those pretrained filters encode
colour-opponent structure from photographs and are being fed temporal structure
instead. Replicating a single frame to 3 channels would respect the pretraining
but throw away the motion signal the agent uses to notice menus and battles, so
the stack is kept and the mismatch is accepted deliberately.

ImageNet is photographs; Emerald is flat-shaded pixel art at 80x120. Frozen
features from natural images may well carry less signal here than a small CNN
trained from scratch on this exact input, which is why Atari-style RL uses
NatureCNN. This exists as an option so the two can be compared on the same
reward rather than argued about.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import ResNet18_Weights, resnet18

RESNET_FEATURES = 512

# ImageNet normalisation. Applied even though the input is a grayscale stack:
# the pretrained weights expect this input distribution, and skipping it would
# add a second mismatch on top of the channel one.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetExtractor(BaseFeaturesExtractor):
    """ResNet18 over the screen, concatenated with the flattened state vector.

    Args:
        observation_space: the env's Dict space, with "screen" and "state".
        finetune: if False (default) the backbone is frozen and runs under
            no_grad. Backprop through ResNet18 at the project's batch size needs
            more VRAM than the 4GB card has spare while training runs, and this
            project has already been OOM-killed twice.
    """

    def __init__(self, observation_space: gym.spaces.Dict, finetune: bool = False):
        state_dim = int(observation_space["state"].shape[0])
        super().__init__(observation_space, features_dim=RESNET_FEATURES + state_dim)

        self.finetune = finetune
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()  # keep the 512-d pooled features, drop the classifier
        self.backbone = net

        if not finetune:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode.

        SB3 calls .train() on the whole policy each update. Without this override
        that would re-enable BatchNorm's running-statistic updates inside the
        frozen backbone, so its "frozen" features would quietly drift.
        """
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        return self

    def forward(self, observations: dict) -> torch.Tensor:
        screen = observations["screen"].float() / 255.0
        screen = (screen - self.mean) / self.std

        if self.finetune:
            visual = self.backbone(screen)
        else:
            with torch.no_grad():
                visual = self.backbone(screen)

        state = torch.flatten(observations["state"], start_dim=1)
        return torch.cat([visual, state], dim=1)
