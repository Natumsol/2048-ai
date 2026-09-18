from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime as ort

from environment.game import Direction, Game, legal_directions
from model.policy import encode_boards
from teacher.expectimax import HEURISTIC_VERSION, ExpectimaxTeacher, ensure_native_teacher


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


TeacherGameResult = tuple[int, int, int, int, bool]


def _benchmark_teacher_game(task: tuple[int, float, int]) -> TeacherGameResult:
    depth, temperature, seed = task
    teacher = ExpectimaxTeacher(depth, temperature)
    game = Game.start(seed)
    for _ in range(20_000):
        if not legal_directions(game.state.board):
            break
        direction = Direction(int(teacher.policy(game.state.board).argmax()))
        game.move(direction)
    truncated = bool(legal_directions(game.state.board))
    return (
        seed,
        max(max(row) for row in game.state.board),
        game.state.score,
        game.state.move_count,
        truncated,
    )


def _load_teacher_checkpoint(path: Path, seed: int) -> TeacherGameResult:
    raw = json.loads(path.read_text())
    if raw.get("seed") != seed:
        raise RuntimeError(f"teacher checkpoint seed mismatch: {path}")
    return (
        seed,
        int(raw["maximumTile"]),
        int(raw["score"]),
        int(raw["moves"]),
        bool(raw["truncated"]),
    )


def _save_teacher_checkpoint(directory: Path, result: TeacherGameResult) -> None:
    seed, maximum, score, moves, truncated = result
    path = directory / f"seed-{seed}.json"
    temporary = directory / f".seed-{seed}.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "seed": seed,
                "maximumTile": maximum,
                "score": score,
                "moves": moves,
                "truncated": truncated,
            }
        )
        + "\n"
    )
    temporary.replace(path)


def benchmark_teacher(
    depth: int,
    temperature: float,
    seeds: range,
    workers: int = 1,
    checkpoint_directory: Path | None = None,
) -> dict[str, object]:
    ensure_native_teacher()
    records: dict[int, tuple[int, int, int, bool]] = {}
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        configuration = {
            "depth": depth,
            "temperature": temperature,
            "heuristicVersion": HEURISTIC_VERSION,
            "seedStart": seeds.start,
            "seedStop": seeds.stop,
            "seedStep": seeds.step,
        }
        config_path = checkpoint_directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise RuntimeError("existing teacher checkpoints use a different benchmark config")
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
        for seed in seeds:
            path = checkpoint_directory / f"seed-{seed}.json"
            if path.exists():
                _, maximum, score, moves, truncated = _load_teacher_checkpoint(path, seed)
                records[seed] = (maximum, score, moves, truncated)

    tasks = [(depth, temperature, seed) for seed in seeds if seed not in records]
    worker_count = max(1, workers)
    total_games = len(seeds)
    report_interval = max(1, total_games // 100)

    if worker_count == 1:
        results = map(_benchmark_teacher_game, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        futures: list[Future[TeacherGameResult]] = [
            executor.submit(_benchmark_teacher_game, task) for task in tasks
        ]
        results = (future.result() for future in as_completed(futures))

    try:
        for result in results:
            seed, maximum, score, moves, truncated = result
            records[seed] = (maximum, score, moves, truncated)
            if checkpoint_directory is not None:
                _save_teacher_checkpoint(checkpoint_directory, result)
            completed = len(records)
            if completed % report_interval == 0 or completed == total_games:
                print(
                    f"teacher benchmark: {completed}/{total_games} games",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    maximum_tiles = [records[seed][0] for seed in seeds]
    scores = [records[seed][1] for seed in seeds]
    move_counts = [records[seed][2] for seed in seeds]
    truncated_games = sum(int(records[seed][3]) for seed in seeds)
    result = _summarize(maximum_tiles, scores, move_counts, seeds, 0, truncated_games)
    result["runner"] = "native-expectimax-teacher"
    result["workers"] = worker_count
    result["heuristicVersion"] = HEURISTIC_VERSION
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
