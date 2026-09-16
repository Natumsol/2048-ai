from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

Row = tuple[int, int, int, int]
Board = tuple[Row, Row, Row, Row]
RandomState = tuple[int, int, int, int]
Position = tuple[int, int]
MASK32 = 0xFFFFFFFF


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass(frozen=True)
class GameState:
    board: Board
    score: int
    move_count: int
    random_state: RandomState


@dataclass(frozen=True)
class MoveResult:
    moved: bool
    state: GameState
    earned_score: int


def _rotate_left(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & MASK32


def _split_mix(seed: int) -> tuple[int, int]:
    seed = (seed + 0x9E3779B9) & MASK32
    mixed = seed
    mixed = ((mixed ^ (mixed >> 16)) * 0x21F0AAAD) & MASK32
    mixed = ((mixed ^ (mixed >> 15)) * 0x735A2D97) & MASK32
    return seed, (mixed ^ (mixed >> 15)) & MASK32


class Xoshiro128StarStar:
    @classmethod
    def from_seed(cls, seed: int) -> Xoshiro128StarStar:
        parts: list[int] = []
        rolling = seed & MASK32
        for _ in range(4):
            rolling, value = _split_mix(rolling)
            parts.append(value)
        state = tuple(parts)
        if state == (0, 0, 0, 0):
            state = (1, 0, 0, 0)
        return cls(state)  # type: ignore[arg-type]

    def __init__(self, state: RandomState) -> None:
        self.state = state

    def next_uint32(self) -> int:
        state0, state1, state2, state3 = self.state
        result = (_rotate_left((state1 * 5) & MASK32, 7) * 9) & MASK32
        shifted = (state1 << 9) & MASK32
        next2 = (state2 ^ state0) & MASK32
        next3 = (state3 ^ state1) & MASK32
        next1 = (state1 ^ next2) & MASK32
        next0 = (state0 ^ next3) & MASK32
        self.state = (next0, next1, (next2 ^ shifted) & MASK32, _rotate_left(next3, 11))
        return result

    def next_float(self) -> float:
        return self.next_uint32() / 0x1_0000_0000


def _empty_board() -> list[list[int]]:
    return [[0 for _ in range(4)] for _ in range(4)]


def _freeze(board: list[list[int]]) -> Board:
    return tuple(tuple(row) for row in board)  # type: ignore[return-value]


def _positions(direction: Direction, line: int) -> list[Position]:
    if direction is Direction.LEFT:
        return [(line, offset) for offset in range(4)]
    if direction is Direction.RIGHT:
        return [(line, 3 - offset) for offset in range(4)]
    if direction is Direction.UP:
        return [(offset, line) for offset in range(4)]
    return [(3 - offset, line) for offset in range(4)]


def slide_board(board: Board, direction: Direction) -> tuple[Board, int]:
    next_board = _empty_board()
    earned_score = 0
    for line in range(4):
        positions = _positions(direction, line)
        tiles = [(position, board[position[0]][position[1]]) for position in positions]
        tiles = [(position, value) for position, value in tiles if value]
        source = 0
        target = 0
        while source < len(tiles):
            _, value = tiles[source]
            destination = positions[target]
            if source + 1 < len(tiles) and tiles[source + 1][1] == value:
                value *= 2
                earned_score += value
                source += 2
            else:
                source += 1
            next_board[destination[0]][destination[1]] = value
            target += 1
    return _freeze(next_board), earned_score


def legal_directions(board: Board) -> tuple[Direction, ...]:
    return tuple(direction for direction in Direction if slide_board(board, direction)[0] != board)


class Game:
    @classmethod
    def start(cls, seed: int) -> Game:
        random = Xoshiro128StarStar.from_seed(seed)
        board = _empty_board()
        cls._spawn(board, random)
        cls._spawn(board, random)
        return cls(GameState(_freeze(board), 0, 0, random.state))

    @classmethod
    def from_state(cls, state: GameState) -> Game:
        return cls(state)

    def __init__(self, state: GameState) -> None:
        self.state = state

    def move(self, direction: Direction) -> MoveResult:
        moved_board, earned_score = slide_board(self.state.board, direction)
        if moved_board == self.state.board:
            return MoveResult(False, self.state, 0)
        board = [list(row) for row in moved_board]
        random = Xoshiro128StarStar(self.state.random_state)
        self._spawn(board, random)
        self.state = GameState(
            board=_freeze(board),
            score=self.state.score + earned_score,
            move_count=self.state.move_count + 1,
            random_state=random.state,
        )
        return MoveResult(True, self.state, earned_score)

    @staticmethod
    def _spawn(board: list[list[int]], random: Xoshiro128StarStar) -> None:
        empty = [
            (row, column) for row in range(4) for column in range(4) if board[row][column] == 0
        ]
        if not empty:
            return
        row, column = empty[int(random.next_float() * len(empty))]
        board[row][column] = 2 if random.next_float() < 0.9 else 4
