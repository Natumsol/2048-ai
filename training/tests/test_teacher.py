import numpy as np

from environment.game import Direction, legal_directions
from teacher.expectimax import ExpectimaxTeacher


def test_teacher_returns_deterministic_soft_targets_only_for_legal_moves() -> None:
    board = (
        (2, 2, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    teacher = ExpectimaxTeacher(depth=2, temperature=0.8)

    first = teacher.policy(board)
    second = teacher.policy(board)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.sum(), 1.0)
    legal = set(legal_directions(board))
    for direction in Direction:
        assert bool(first[direction] > 0) == (direction in legal)
