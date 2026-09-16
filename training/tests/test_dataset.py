from __future__ import annotations

import numpy as np

from data.dataset import (
    boards_to_exponents,
    exponents_to_boards,
    split_for_seed,
    transform_sample,
)


def test_compact_board_encoding_round_trips_and_saturates() -> None:
    boards = np.array(
        [
            [
                [0, 2, 4, 8],
                [16, 32, 64, 128],
                [256, 512, 1024, 2048],
                [4096, 8192, 32768, 65536],
            ]
        ]
    )

    encoded = boards_to_exponents(boards)
    restored = exponents_to_boards(encoded)

    assert encoded.dtype == np.uint8
    assert encoded[0, 3, 3] == 15
    np.testing.assert_array_equal(restored[0, 3], [4096, 8192, 32768, 32768])


def test_seed_split_is_stable_and_disjoint() -> None:
    assignments = {seed: split_for_seed(seed) for seed in range(100)}
    assert set(assignments.values()) == {"train", "validation", "test"}
    assert assignments == {seed: split_for_seed(seed) for seed in range(100)}


def test_symmetry_transforms_board_and_direction_together() -> None:
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 1
    target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    legal = np.array([1, 0, 0, 0], dtype=np.uint8)

    rotated, rotated_target, rotated_legal = transform_sample(board, target, legal, 1)

    assert rotated[3, 0] == 1
    np.testing.assert_array_equal(rotated_target, [0, 0, 0, 1])
    np.testing.assert_array_equal(rotated_legal, [0, 0, 0, 1])
