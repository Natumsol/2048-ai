from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import onnxruntime as ort

from environment.game import Direction, Game, legal_directions
from model.policy import encode_boards
from teacher.expectimax import ExpectimaxTeacher


def _summarize(
    maximum_tiles: list[int],
    scores: list[int],
    move_counts: list[int],
    seeds: range,
    illegal_moves: int,
    truncated_games: int,
) -> dict[str, object]:
    games = len(maximum_tiles)
    reached = sum(tile >= 2048 for tile in maximum_tiles)
    return {
        "games": games,
        "seedStart": seeds.start,
        "illegalMoves": illegal_moves,
        "truncatedGames": truncated_games,
        "reached2048": reached,
        "reached2048Rate": reached / max(games, 1),
        "medianMaximumTile": statistics.median(maximum_tiles),
        "medianScore": statistics.median(scores),
        "medianMoves": statistics.median(move_counts),
    }


def benchmark_policy(model_path: Path, seeds: range) -> dict[str, object]:
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    maximum_tiles: list[int] = []
    scores: list[int] = []
    move_counts: list[int] = []
    illegal_moves = 0
    truncated_games = 0
    for seed in seeds:
        game = Game.start(seed)
        for _ in range(20_000):
            legal = legal_directions(game.state.board)
            if not legal:
                break
            board = np.asarray(game.state.board, dtype=np.uint32)
            encoded = encode_boards(board).numpy()
            logits = np.asarray(session.run(["logits"], {"board": encoded})[0], dtype=np.float32)[0]
            masked = np.full(4, -np.inf, dtype=np.float32)
            masked[list(legal)] = logits[list(legal)]
            direction = Direction(int(masked.argmax()))
            result = game.move(direction)
            if not result.moved:
                illegal_moves += 1
                break
        maximum_tiles.append(max(max(row) for row in game.state.board))
        if legal_directions(game.state.board):
            truncated_games += 1
        scores.append(game.state.score)
        move_counts.append(game.state.move_count)

    return _summarize(maximum_tiles, scores, move_counts, seeds, illegal_moves, truncated_games)


def benchmark_teacher(depth: int, temperature: float, seeds: range) -> dict[str, object]:
    teacher = ExpectimaxTeacher(depth, temperature)
    maximum_tiles: list[int] = []
    scores: list[int] = []
    move_counts: list[int] = []
    truncated_games = 0
    for seed in seeds:
        game = Game.start(seed)
        for _ in range(20_000):
            if not legal_directions(game.state.board):
                break
            direction = Direction(int(teacher.policy(game.state.board).argmax()))
            game.move(direction)
        maximum_tiles.append(max(max(row) for row in game.state.board))
        if legal_directions(game.state.board):
            truncated_games += 1
        scores.append(game.state.score)
        move_counts.append(game.state.move_count)
    result = _summarize(maximum_tiles, scores, move_counts, seeds, 0, truncated_games)
    result["runner"] = "python-expectimax-teacher"
    return result


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("../public/models/policy.onnx"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()
    metrics = benchmark_policy(
        arguments.model, range(arguments.seed, arguments.seed + arguments.games)
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    if arguments.manifest:
        manifest = json.loads(arguments.manifest.read_text())
        manifest["metadata"]["pythonSmokeBenchmark"] = metrics
        arguments.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _main()
