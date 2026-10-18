"""Feature extractor with a learned text encoder and an inverse-dynamics head.

Three inputs, one feature vector:

  screen  3 stacked grayscale frames -> NatureCNN -> 512
  state   29 values from RAM and the agent's own counters -> passthrough
  text    16 hashed word ids of the on-screen text -> EmbeddingBag -> 32

The text encoder is the "small language model" in the loosest sense that
earns the name: an embedding per hashed word, mean-pooled, trained end to end
by PPO. Nothing bigger is justified by a 64-character window of mostly stale
battle fragments, and anything bigger would need text we do not have to
pretrain on. It lets the policy tell "Got away safely!" from "Wild ZIGZAGOON
appeared!" from a menu, which pixels at 120x80 do not make easy.

The inverse-dynamics head is separate from forward(). It takes the CNN
features of two consecutive screens and predicts the button pressed between
them, and is trained by InverseDynamicsCallback in train.py on pairs from the
rollout buffer, with gradients flowing into the shared CNN. That forces the
visual features to encode what actions do to the screen: a step that moved
against one that hit a wall, a menu opening, a battle starting. PPO alone has
to discover that from reward, and 54 million steps of it did not.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

CNN_FEATURES = 512
TEXT_EMBED = 32
TILE_FEATURES = 128
IDM_HIDDEN = 256


class AgentExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int = 8):
        state_dim = int(observation_space["state"].shape[0])
        vocab = int(observation_space["text"].high.max()) + 1
        tile_c, tile_h, tile_w = observation_space["tiles"].shape
        battle_dim = int(observation_space["battle"].shape[0])
        super().__init__(observation_space, features_dim=CNN_FEATURES + state_dim + TEXT_EMBED + TILE_FEATURES + battle_dim)

        self.cnn = NatureCNN(observation_space["screen"], features_dim=CNN_FEATURES)
        # padding_idx=0 keeps the pad token at a fixed zero vector, so an empty
        # window reads as "no text" rather than as a learned word.
        self.text = nn.EmbeddingBag(vocab, TEXT_EMBED, mode="mean", padding_idx=0)
        self.idm = nn.Sequential(
            nn.Linear(2 * CNN_FEATURES, IDM_HIDDEN),
            nn.ReLU(),
            nn.Linear(IDM_HIDDEN, n_actions),
        )
        # Forward model: next visual features from (features, action). Its
        # prediction error is the curiosity bonus in train.py's CuriosityWrapper:
        # the agent is paid for reaching screens the model cannot yet predict,
        # which is what "vision guiding where to go" means in practice. Trained
        # on the same rollout pairs as the inverse model. The CNN is detached on
        # the forward side so the features cannot collapse to make prediction
        # trivial; the inverse model is what shapes them.
        # The tile window is exact semantics (walkable, grass, door, NPC...)
        # on a small grid, so a small CNN over it is enough; the battle vector
        # is already features and passes straight through.
        self.tiles = nn.Sequential(
            nn.Conv2d(tile_c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * tile_h * tile_w, TILE_FEATURES), nn.ReLU(),
        )
        self.n_actions = n_actions
        self.fwd = nn.Sequential(
            nn.Linear(CNN_FEATURES + n_actions, IDM_HIDDEN),
            nn.ReLU(),
            nn.Linear(IDM_HIDDEN, CNN_FEATURES),
        )

    def encode_screen(self, screen: torch.Tensor) -> torch.Tensor:
        """screen already in [0, 1]; SB3's preprocessing divides by 255."""
        return self.cnn(screen)

    def forward(self, observations: dict) -> torch.Tensor:
        visual = self.encode_screen(observations["screen"])
        words = self.text(observations["text"].long())
        state = torch.flatten(observations["state"], start_dim=1)
        tiles = self.tiles(observations["tiles"].float())
        battle = torch.flatten(observations["battle"], start_dim=1)
        return torch.cat([visual, state, words, tiles, battle], dim=1)

    def forward_error(self, screen_t: torch.Tensor, screen_t1: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-sample squared error of the forward model, shape (batch,).
        Features are detached: this trains and scores the predictor only."""
        with torch.no_grad():
            f0 = self.encode_screen(screen_t)
            f1 = self.encode_screen(screen_t1)
        onehot = torch.nn.functional.one_hot(actions.long(), self.n_actions).float()
        pred = self.fwd(torch.cat([f0, onehot], dim=1))
        return ((pred - f1) ** 2).mean(dim=1)

    def idm_logits(self, screen_t: torch.Tensor, screen_t1: torch.Tensor) -> torch.Tensor:
        """Which action was taken between two consecutive screens, as logits."""
        return self.idm(torch.cat([self.encode_screen(screen_t), self.encode_screen(screen_t1)], dim=1))
