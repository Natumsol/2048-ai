from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from environment.game import Board, legal_directions, slide_board


def _with_tile(board: Board, row: int, column: int, value: int) -> Board:
    mutable = [list(line) for line in board]
    mutable[row][column] = value
    return tuple(tuple(line) for line in mutable)  # type: ignore[return-value]


def _heuristic(board: Board) -> float:
    logs = np.zeros((4, 4), dtype=np.float32)
    for row in range(4):
        for column in range(4):
            value = board[row][column]
            logs[row, column] = np.log2(value) if value else 0

    empty = float(np.count_nonzero(logs == 0))
    maximum = float(logs.max())
    smoothness = 0.0
    merge_potential = 0.0
    for row in range(4):
        for column in range(4):
            current = logs[row, column]
            if current == 0:
                continue
            if column < 3 and logs[row, column + 1] > 0:
                smoothness -= abs(float(current - logs[row, column + 1]))
                merge_potential += float(current == logs[row, column + 1])
            if row < 3 and logs[row + 1, column] > 0:
                smoothness -= abs(float(current - logs[row + 1, column]))
                merge_potential += float(current == logs[row + 1, column])

    monotonicity = 0.0
    for axis in (0, 1):
        forward = 0.0
        backward = 0.0
        lines = logs if axis == 0 else logs.T
        for line in lines:
            for index in range(3):
                delta = float(line[index] - line[index + 1])
                if delta > 0:
                    forward += delta
                else:
                    backward -= delta
        monotonicity += max(forward, backward)

    corners = (logs[0, 0], logs[0, 3], logs[3, 0], logs[3, 3])
    corner_bonus = maximum if maximum in corners else 0.0
    return (
        empty * 320.0
        + merge_potential * 95.0
        + monotonicity * 18.0
        + smoothness * 14.0
        + corner_bonus * 110.0
        + maximum * 20.0
    )


@lru_cache(maxsize=250_000)
def _max_value(board: Board, depth: int) -> float:
    directions = legal_directions(board)
    if depth <= 0 or not directions:
        return _heuristic(board)
    return max(
        earned + _chance_value(moved, depth - 1)
        for direction in directions
        for moved, earned in (slide_board(board, direction),)
    )


@lru_cache(maxsize=250_000)
def _chance_value(board: Board, depth: int) -> float:
    empty = [(row, column) for row in range(4) for column in range(4) if board[row][column] == 0]
    if not empty:
        return _max_value(board, depth)
    cell_probability = 1.0 / len(empty)
    expected = 0.0
    for row, column in empty:
        expected += cell_probability * 0.9 * _max_value(_with_tile(board, row, column, 2), depth)
        expected += cell_probability * 0.1 * _max_value(_with_tile(board, row, column, 4), depth)
    return expected


class ExpectimaxTeacher:
    def __init__(self, depth: int = 3, temperature: float = 1.0) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.depth = depth
        self.temperature = temperature

    def policy(self, board: Board) -> NDArray[np.float32]:
        scores = np.full(4, -np.inf, dtype=np.float64)
        for direction in legal_directions(board):
            moved, earned = slide_board(board, direction)
            scores[direction] = earned + _chance_value(moved, self.depth - 1)

        legal_mask = np.isfinite(scores)
        if not legal_mask.any():
            return np.zeros(4, dtype=np.float32)
        legal_scores = scores[legal_mask]
        centered = (legal_scores - legal_scores.max()) / self.temperature
        probabilities = np.exp(centered)
        probabilities /= probabilities.sum()
        result = np.zeros(4, dtype=np.float32)
        result[legal_mask] = probabilities.astype(np.float32)
        return result
