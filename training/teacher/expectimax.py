from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from environment.game import Board, Direction

ROW_MASK = 0xFFFF
BOARD_CELLS = 16
CELL_BITS = 4
ROW_BITS = 16
HEURISTIC_VERSION = "nneonneo-native-fixed-depth-v1"
CPROBABILITY_THRESHOLD = 0.0001
_NATIVE_SOURCE = Path(__file__).with_name("native_expectimax.cpp")
_NATIVE_DIRECTORY = Path(__file__).resolve().parents[1] / "artifacts" / "native"
_NATIVE_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
_NATIVE_PATH = _NATIVE_DIRECTORY / f"libexpectimax{_NATIVE_SUFFIX}"
_native_library: ctypes.CDLL | None = None


def _compile_native_teacher() -> None:
    if _NATIVE_PATH.exists() and _NATIVE_PATH.stat().st_mtime >= _NATIVE_SOURCE.stat().st_mtime:
        return
    _NATIVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary = _NATIVE_PATH.with_name(f".{_NATIVE_PATH.name}.{os.getpid()}.tmp")
    command = [
        os.environ.get("CXX", "c++"),
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        _NATIVE_SOURCE.as_posix(),
        "-o",
        temporary.as_posix(),
    ]
    if sys.platform == "darwin":
        command[3] = "-dynamiclib"
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temporary, _NATIVE_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_native_teacher() -> ctypes.CDLL:
    global _native_library
    if _native_library is None:
        _compile_native_teacher()
        library = ctypes.CDLL(_NATIVE_PATH.as_posix())
        policy = library.expectimax_policy
        policy.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_float),
        ]
        policy.restype = ctypes.c_int
        _native_library = library
    return _native_library


def _reverse_row(row: int) -> int:
    return (
        ((row & 0x000F) << 12)
        | ((row & 0x00F0) << 4)
        | ((row & 0x0F00) >> 4)
        | ((row & 0xF000) >> 12)
    )


def _build_row_tables() -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    left_moves: list[int] = []
    left_scores: list[int] = []
    reverse_rows: list[int] = []
    heuristic_scores: list[float] = []

    for row in range(1 << ROW_BITS):
        values = tuple((row >> (CELL_BITS * index)) & 0xF for index in range(4))
        tiles = [value for value in values if value]
        merged: list[int] = []
        score = 0
        source = 0
        overflow = False
        while source < len(tiles):
            value = tiles[source]
            if source + 1 < len(tiles) and tiles[source + 1] == value:
                value += 1
                if value >= 16:
                    overflow = True
                    break
                score += 1 << value
                source += 2
            else:
                source += 1
            merged.append(value)
        merged.extend([0] * (4 - len(merged)))
        moved = sum(value << (CELL_BITS * index) for index, value in enumerate(merged))
        left_moves.append(-1 if overflow else moved)
        left_scores.append(score)
        reverse_rows.append(_reverse_row(row))

        rank_sum = sum(float(value) ** 3.5 for value in values)
        empty = values.count(0)
        merges = 0
        previous = 0
        counter = 0
        for value in values:
            if value == 0:
                continue
            if previous == value:
                counter += 1
            elif counter > 0:
                merges += 1 + counter
                counter = 0
            previous = value
        if counter > 0:
            merges += 1 + counter

        monotonicity_left = 0.0
        monotonicity_right = 0.0
        for index in range(3):
            current = values[index]
            following = values[index + 1]
            if current > following:
                monotonicity_left += float(current) ** 4 - float(following) ** 4
            else:
                monotonicity_right += float(following) ** 4 - float(current) ** 4
        heuristic_scores.append(
            200_000.0
            + 270.0 * empty
            + 700.0 * merges
            - 47.0 * min(monotonicity_left, monotonicity_right)
            - 11.0 * rank_sum
        )

    right_moves = []
    right_scores = []
    for row in range(1 << ROW_BITS):
        reversed_row = reverse_rows[row]
        reversed_move = left_moves[reversed_row]
        right_moves.append(-1 if reversed_move < 0 else reverse_rows[reversed_move])
        right_scores.append(left_scores[reversed_row])

    return (
        tuple(left_moves),
        tuple(right_moves),
        tuple(left_scores),
        tuple(right_scores),
        tuple(heuristic_scores),
    )


(
    _ROW_LEFT,
    _ROW_RIGHT,
    _ROW_LEFT_SCORE,
    _ROW_RIGHT_SCORE,
    _ROW_HEURISTIC,
) = _build_row_tables()


def board_to_bits(board: Board) -> int:
    bits = 0
    for index, value in enumerate(value for row in board for value in row):
        exponent = value.bit_length() - 1 if value else 0
        if exponent >= 16:
            raise OverflowError("expectimax bitboard supports tiles up to 32768")
        bits |= exponent << (CELL_BITS * index)
    return bits


def bits_to_board(bits: int) -> Board:
    values = []
    for index in range(BOARD_CELLS):
        exponent = (bits >> (CELL_BITS * index)) & 0xF
        values.append(0 if exponent == 0 else 1 << exponent)
    return tuple(tuple(values[row * 4 : row * 4 + 4]) for row in range(4))  # type: ignore[return-value]


def _transpose_bits(bits: int) -> int:
    transposed = 0
    for row in range(4):
        for column in range(4):
            source = CELL_BITS * (row * 4 + column)
            target = CELL_BITS * (column * 4 + row)
            transposed |= ((bits >> source) & 0xF) << target
    return transposed


def _slide_rows(bits: int, right: bool) -> tuple[int, int]:
    move_table = _ROW_RIGHT if right else _ROW_LEFT
    score_table = _ROW_RIGHT_SCORE if right else _ROW_LEFT_SCORE
    moved = 0
    score = 0
    for row in range(4):
        row_value = (bits >> (ROW_BITS * row)) & ROW_MASK
        moved_row = move_table[row_value]
        if moved_row < 0:
            raise OverflowError("expectimax bitboard cannot merge tiles above 32768")
        moved |= moved_row << (ROW_BITS * row)
        score += score_table[row_value]
    return moved, score


def slide_bits(bits: int, direction: Direction) -> tuple[int, int]:
    if direction is Direction.LEFT:
        return _slide_rows(bits, False)
    if direction is Direction.RIGHT:
        return _slide_rows(bits, True)
    transposed = _transpose_bits(bits)
    moved, score = _slide_rows(transposed, direction is Direction.DOWN)
    return _transpose_bits(moved), score


@lru_cache(maxsize=250_000)
def _heuristic_bits(bits: int) -> float:
    rows = tuple((bits >> (ROW_BITS * row)) & ROW_MASK for row in range(4))
    transposed = _transpose_bits(bits)
    columns = tuple((transposed >> (ROW_BITS * row)) & ROW_MASK for row in range(4))
    return sum(_ROW_HEURISTIC[row] for row in rows) + sum(
        _ROW_HEURISTIC[column] for column in columns
    )


TranspositionTable = dict[int, tuple[int, float]]


def _move_value(
    bits: int,
    cumulative_probability: float,
    depth: int,
    depth_limit: int,
    transposition_table: TranspositionTable,
) -> float:
    best = 0.0
    for direction in Direction:
        moved, _ = slide_bits(bits, direction)
        if moved == bits:
            continue
        best = max(
            best,
            _chance_value(
                moved,
                cumulative_probability,
                depth + 1,
                depth_limit,
                transposition_table,
            ),
        )
    return best


def _chance_value(
    bits: int,
    cumulative_probability: float,
    depth: int,
    depth_limit: int,
    transposition_table: TranspositionTable,
) -> float:
    if cumulative_probability < CPROBABILITY_THRESHOLD or depth >= depth_limit:
        return _heuristic_bits(bits)
    if depth < 15:
        cached = transposition_table.get(bits)
        if cached is not None and cached[0] <= depth:
            return cached[1]
    empty_shifts = [
        CELL_BITS * index
        for index in range(BOARD_CELLS)
        if not ((bits >> (CELL_BITS * index)) & 0xF)
    ]
    if not empty_shifts:
        return _move_value(bits, cumulative_probability, depth, depth_limit, transposition_table)
    cell_probability = 1.0 / len(empty_shifts)
    expected = 0.0
    for shift in empty_shifts:
        probability_2 = cumulative_probability * cell_probability * 0.9
        probability_4 = cumulative_probability * cell_probability * 0.1
        expected += (
            cell_probability
            * 0.9
            * _move_value(
                bits | (1 << shift),
                probability_2,
                depth,
                depth_limit,
                transposition_table,
            )
        )
        expected += (
            cell_probability
            * 0.1
            * _move_value(
                bits | (2 << shift),
                probability_4,
                depth,
                depth_limit,
                transposition_table,
            )
        )
    if depth < 15:
        transposition_table[bits] = (depth, expected)
    return expected


class ExpectimaxTeacher:
    def __init__(self, depth: int = 3, temperature: float = 1.0) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.depth = depth
        self.temperature = temperature
        self._native = ensure_native_teacher()

    def policy(self, board: Board) -> NDArray[np.float32]:
        bits = board_to_bits(board)
        output = (ctypes.c_float * 4)()
        self._native.expectimax_policy(bits, self.depth, self.temperature, output)
        return np.ctypeslib.as_array(output).copy()
