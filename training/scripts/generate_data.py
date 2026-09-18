from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data.dataset import boards_to_exponents, split_for_seed
from environment.game import Direction, Game, legal_directions
from teacher.expectimax import HEURISTIC_VERSION, ExpectimaxTeacher, ensure_native_teacher


@dataclass(frozen=True)
class GenerationConfig:
    profile: str
    seed: int
    samples: int
    teacher_depth: int
    teacher_temperature: float
    workers: int = 1
    teacher_version: str = HEURISTIC_VERSION


@dataclass(frozen=True)
class EpisodeData:
    episode: int
    boards: NDArray[np.uint8]
    targets: NDArray[np.float32]
    legal: NDArray[np.uint8]
    seeds: NDArray[np.uint32]
    steps: NDArray[np.uint16]
    sources: NDArray[np.uint8]
    priorities: NDArray[np.float32]


def _episode_seed(config: GenerationConfig, episode: int) -> int:
    split_order = (0, 8, 9, 1, 2, 3, 4, 5, 6, 7)
    cycle, offset = divmod(episode, len(split_order))
    return config.seed + cycle * 10 + split_order[offset]


def _generate_episode(task: tuple[GenerationConfig, int]) -> EpisodeData:
    config, episode = task
    teacher = ExpectimaxTeacher(config.teacher_depth, config.teacher_temperature)
    seed_episode, source = divmod(episode, 2)
    seed = _episode_seed(config, seed_episode)
    game = Game.start(seed)
    boards: list[NDArray[np.uint8]] = []
    targets: list[NDArray[np.float32]] = []
    masks: list[NDArray[np.uint8]] = []
    for step in range(20_000):
        board = game.state.board
        directions = legal_directions(board)
        if not directions:
            break
        target = teacher.policy(board)
        legal = np.zeros(4, dtype=np.uint8)
        legal[list(directions)] = 1
        boards.append(boards_to_exponents(np.asarray(board, dtype=np.uint32)))
        targets.append(target)
        masks.append(legal)
        ranked = [Direction(index) for index in np.argsort(target)[::-1] if legal[index]]
        choice = ranked[1] if source == 1 and step % 5 == 2 and len(ranked) > 1 else ranked[0]
        game.move(choice)

    if legal_directions(game.state.board):
        raise RuntimeError(f"episode {episode} exceeded the 20,000 move safety limit")

    count = len(boards)
    return EpisodeData(
        episode=episode,
        boards=np.stack(boards).astype(np.uint8),
        targets=np.stack(targets).astype(np.float32),
        legal=np.stack(masks).astype(np.uint8),
        seeds=np.full(count, seed, dtype=np.uint32),
        steps=np.arange(count, dtype=np.uint16),
        sources=np.full(count, source, dtype=np.uint8),
        priorities=np.ones(count, dtype=np.float32),
    )


def save_episode_shard(path: Path, episode: EpisodeData) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        boards=episode.boards,
        targets=episode.targets,
        legal=episode.legal,
        seeds=episode.seeds,
        steps=episode.steps,
        sources=episode.sources,
        priorities=episode.priorities,
    )
    temporary.replace(path)


def episode_sample_count(path: Path) -> int:
    with np.load(path) as archive:
        return len(archive["boards"])


class _InlineExecutor:
    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def map(
        self,
        function: Callable[[tuple[GenerationConfig, int]], EpisodeData],
        tasks: list[tuple[GenerationConfig, int]],
    ) -> Iterable[EpisodeData]:
        return [function(task) for task in tasks]


def _generate_resumable_shards(config: GenerationConfig, output_directory: Path) -> list[Path]:
    ensure_native_teacher()
    shard_directory = output_directory / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    generated = 0
    episode = 0
    while True:
        path = shard_directory / f"episode-{episode:06d}.npz"
        if not path.exists():
            break
        shards.append(path)
        generated += episode_sample_count(path)
        episode += 1

    workers = max(1, config.workers)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else _InlineExecutor()
    with executor:
        while generated < config.samples:
            tasks = [(config, index) for index in range(episode, episode + workers)]
            for result in executor.map(_generate_episode, tasks):
                path = shard_directory / f"episode-{result.episode:06d}.npz"
                save_episode_shard(path, result)
                shards.append(path)
                generated += len(result.boards)
                episode += 1
            (output_directory / "progress.json").write_text(
                json.dumps({"completedEpisodes": episode, "generatedSamples": generated}) + "\n"
            )
    return shards


def merge_episode_shards(
    shards: list[Path], sample_limit: int, output_directory: Path
) -> tuple[dict[str, int], dict[str, str], int, int]:
    selected: list[tuple[Path, int, str]] = []
    counts = {name: 0 for name in ("train", "validation", "test")}
    remaining = sample_limit
    for path in shards:
        if remaining <= 0:
            break
        with np.load(path) as archive:
            take = min(remaining, len(archive["boards"]))
            seed = int(archive["seeds"][0])
        split = split_for_seed(seed)
        selected.append((path, take, split))
        counts[split] += take
        remaining -= take
    if remaining:
        raise RuntimeError(f"dataset is short by {remaining} samples")

    arrays: dict[str, dict[str, NDArray[np.generic]]] = {}
    for split, count in counts.items():
        arrays[split] = {
            "boards": np.empty((count, 4, 4), dtype=np.uint8),
            "targets": np.empty((count, 4), dtype=np.float32),
            "legal": np.empty((count, 4), dtype=np.uint8),
            "seeds": np.empty(count, dtype=np.uint32),
            "steps": np.empty(count, dtype=np.uint16),
            "sources": np.empty(count, dtype=np.uint8),
            "priorities": np.empty(count, dtype=np.float32),
        }
    offsets = {name: 0 for name in counts}
    for path, take, split in selected:
        offset = offsets[split]
        with np.load(path) as archive:
            for name, destination in arrays[split].items():
                if name == "priorities" and name not in archive:
                    destination[offset : offset + take] = 1.0
                else:
                    destination[offset : offset + take] = archive[name][:take]
        offsets[split] += take

    hashes: dict[str, str] = {}
    normal = 0
    perturbed = 0
    for split, split_arrays in arrays.items():
        path = output_directory / f"{split}.npz"
        np.savez_compressed(path, **split_arrays)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources = split_arrays["sources"]
        normal += int(np.count_nonzero(sources == 0))
        perturbed += int(np.count_nonzero(sources == 1))
    return counts, hashes, normal, perturbed


def generate_dataset(config: GenerationConfig, output_directory: Path) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "generation-config.json"
    serialized_config = asdict(config)
    if config_path.exists() and json.loads(config_path.read_text()) != serialized_config:
        raise RuntimeError("existing resumable shards use a different generation config")
    config_path.write_text(json.dumps(serialized_config, indent=2) + "\n")
    shards = _generate_resumable_shards(config, output_directory)
    counts, hashes, normal, perturbed = merge_episode_shards(
        shards, config.samples, output_directory
    )
    manifest: dict[str, object] = {
        "version": 1,
        "profile": config.profile,
        "seed": config.seed,
        "samples": config.samples,
        "normalSamples": normal,
        "perturbedSamples": perturbed,
        "workers": max(1, config.workers),
        "resumableShards": len(shards),
        "teacher": {
            "kind": "deterministic-expectimax",
            "depth": config.teacher_depth,
            "temperature": config.teacher_temperature,
            "heuristicVersion": config.teacher_version,
        },
        "split": "seed modulo 10 bucketed 80/10/10 by complete episode",
        "counts": counts,
        "sha256": hashes,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.config.read_text())
    config = GenerationConfig(
        profile=raw["profile"],
        seed=raw["seed"],
        samples=raw["samples"],
        teacher_depth=raw["teacherDepth"],
        teacher_temperature=raw["teacherTemperature"],
        workers=raw["dataWorkers"],
        teacher_version=raw.get("teacherVersion", HEURISTIC_VERSION),
    )
    print(json.dumps(generate_dataset(config, arguments.output), indent=2))


if __name__ == "__main__":
    _main()
