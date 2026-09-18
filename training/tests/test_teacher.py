from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.benchmark as benchmark_module
from environment.game import Direction, legal_directions, slide_board
from scripts.benchmark import benchmark_teacher
from teacher.expectimax import (
    ExpectimaxTeacher,
    bits_to_board,
    board_to_bits,
    slide_bits,
)


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


def test_optimized_teacher_matches_reference_policies() -> None:
    board = (
        (2, 2, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )

    actual = ExpectimaxTeacher(2, 96.0).policy(board)

    np.testing.assert_allclose(
        actual,
        [0.0, 0.20783302, 0.58433396, 0.20783302],
        rtol=1e-6,
        atol=1e-8,
    )


def test_bitboard_moves_match_authoritative_game_rules() -> None:
    boards = (
        ((2, 2, 4, 4), (8, 0, 8, 8), (0, 16, 0, 16), (32, 32, 64, 0)),
        ((2, 4, 8, 16), (32, 64, 128, 256), (2, 0, 2, 4), (0, 8, 8, 0)),
    )

    for board in boards:
        bits = board_to_bits(board)
        assert bits_to_board(bits) == board
        for direction in Direction:
            expected_board, expected_score = slide_board(board, direction)
            actual_bits, actual_score = slide_bits(bits, direction)
            assert bits_to_board(actual_bits) == expected_board
            assert actual_score == expected_score


def test_parallel_teacher_benchmark_matches_serial_results_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = range(70_000, 70_002)

    def fake_game(task: tuple[int, float, int]) -> benchmark_module.TeacherGameResult:
        _, _, seed = task
        return seed, 2048 if seed % 2 == 0 else 1024, seed, 100, False

    class ImmediateExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def submit(
            self,
            function: Any,
            *arguments: object,
        ) -> Future[benchmark_module.TeacherGameResult]:
            future: Future[benchmark_module.TeacherGameResult] = Future()
            future.set_result(function(*arguments))
            return future

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(benchmark_module, "_benchmark_teacher_game", fake_game)
    monkeypatch.setattr(benchmark_module, "ProcessPoolExecutor", ImmediateExecutor)

    serial = benchmark_teacher(1, 1.0, seeds, workers=1)
    parallel = benchmark_teacher(1, 1.0, seeds, workers=2, checkpoint_directory=tmp_path)

    assert {key: value for key, value in serial.items() if key != "workers"} == {
        key: value for key, value in parallel.items() if key != "workers"
    }
    assert serial["workers"] == 1
    assert parallel["workers"] == 2

    def fail_if_recomputed(_task: tuple[int, float, int]) -> benchmark_module.TeacherGameResult:
        raise AssertionError("completed teacher games should load from checkpoints")

    monkeypatch.setattr(benchmark_module, "_benchmark_teacher_game", fail_if_recomputed)
    resumed = benchmark_teacher(1, 1.0, seeds, workers=1, checkpoint_directory=tmp_path)
    assert {key: value for key, value in resumed.items() if key != "workers"} == {
        key: value for key, value in parallel.items() if key != "workers"
    }
