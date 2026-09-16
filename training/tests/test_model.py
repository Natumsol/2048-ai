from __future__ import annotations

import numpy as np
import torch

from model.policy import PolicyNetwork, encode_boards


def test_encode_boards_uses_fixed_one_hot_contract() -> None:
    boards = np.array(
        [
            [
                [0, 2, 4, 8],
                [16, 32, 64, 128],
                [256, 512, 1024, 2048],
                [4096, 8192, 16384, 65536],
            ]
        ],
        dtype=np.uint32,
    )

    encoded = encode_boards(boards)

    assert encoded.shape == (1, 16, 4, 4)
    assert encoded.dtype == torch.float32
    torch.testing.assert_close(encoded.sum(dim=1), torch.ones((1, 4, 4)))
    assert encoded[0, 0, 0, 0] == 1
    assert encoded[0, 1, 0, 1] == 1
    assert encoded[0, 15, 3, 3] == 1


def test_policy_network_emits_four_finite_logits() -> None:
    model = PolicyNetwork()
    logits = model(torch.zeros((3, 16, 4, 4), dtype=torch.float32))

    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
