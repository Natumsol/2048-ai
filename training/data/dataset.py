from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SplitName = str

_VECTORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def boards_to_exponents(boards: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = boards.astype(np.uint64, copy=False)
    encoded = np.zeros(values.shape, dtype=np.uint8)
    nonzero = values > 0
    encoded[nonzero] = np.floor(np.log2(values[nonzero])).astype(np.uint8)
    return np.clip(encoded, 0, 15).astype(np.uint8, copy=False)


def exponents_to_boards(exponents: NDArray[np.uint8]) -> NDArray[np.uint32]:
    values = np.zeros(exponents.shape, dtype=np.uint32)
    nonzero = exponents > 0
    values[nonzero] = np.left_shift(np.uint32(1), exponents[nonzero].astype(np.uint32))
    return values


def split_for_seed(seed: int) -> SplitName:
    bucket = seed % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def _direction_permutation(transform: int) -> NDArray[np.int64]:
    turns = transform % 4
    mirror = transform >= 4
    mapping: list[int] = []
    for row, column in _VECTORS:
        for _ in range(turns):
            row, column = -column, row
        if mirror:
            column = -column
        mapping.append(_VECTORS.index((row, column)))
    return np.asarray(mapping, dtype=np.int64)


def transform_sample(
    board: NDArray[np.uint8],
    target: NDArray[np.float32],
    legal: NDArray[np.uint8],
    transform: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.uint8]]:
    if not 0 <= transform < 8:
        raise ValueError("transform must be in [0, 7]")
    transformed = np.rot90(board, transform % 4)
    if transform >= 4:
        transformed = np.fliplr(transformed)
    permutation = _direction_permutation(transform)
    next_target = np.zeros_like(target)
    next_legal = np.zeros_like(legal)
    next_target[permutation] = target
    next_legal[permutation] = legal
    return transformed.copy(), next_target, next_legal
