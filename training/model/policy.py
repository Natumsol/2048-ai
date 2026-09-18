from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


def encode_boards(boards: NDArray[np.generic]) -> Tensor:
    """Encode tile values as the browser contract's 16 one-hot planes."""
    values = torch.as_tensor(boards.astype(np.int64, copy=False))
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if tuple(values.shape[1:]) != (4, 4):
        raise ValueError("boards must have shape [N,4,4] or [4,4]")
    exponents = torch.zeros_like(values)
    nonzero = values > 0
    exponents[nonzero] = torch.log2(values[nonzero].to(torch.float32)).to(torch.int64)
    exponents.clamp_(0, 15)
    return torch.nn.functional.one_hot(exponents, num_classes=16).permute(0, 3, 1, 2).float()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.layers(inputs))


class PolicyNetwork(nn.Module):
    """Small fixed-shape residual policy network for browser inference."""

    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.blocks(self.stem(inputs)))


class SymmetryEnsemblePolicy(nn.Module):
    """Average policy logits over the eight symmetries of the square board."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer(
            "direction_permutations",
            torch.tensor(
                (
                    (0, 1, 2, 3),
                    (3, 0, 1, 2),
                    (2, 3, 0, 1),
                    (1, 2, 3, 0),
                    (0, 3, 2, 1),
                    (1, 0, 3, 2),
                    (2, 1, 0, 3),
                    (3, 2, 1, 0),
                ),
                dtype=torch.int64,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        transformed_boards: list[Tensor] = []
        for transform in range(8):
            turns = transform % 4
            if turns == 0:
                transformed = inputs
            elif turns == 1:
                transformed = torch.flip(inputs.transpose(2, 3), dims=(2,))
            elif turns == 2:
                transformed = torch.flip(inputs, dims=(2, 3))
            else:
                transformed = torch.flip(inputs.transpose(2, 3), dims=(3,))
            if transform >= 4:
                transformed = torch.flip(transformed, dims=(3,))
            transformed_boards.append(transformed)

        transformed_logits = self.policy(torch.cat(transformed_boards)).chunk(8)
        canonical_logits: list[Tensor] = []
        for transform, logits in enumerate(transformed_logits):
            permutation = self.direction_permutations[transform]
            canonical_logits.append(logits[:, permutation])
        return torch.stack(canonical_logits).mean(dim=0)
