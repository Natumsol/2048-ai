from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from data.dataset import boards_to_exponents
from environment.game import Board, Direction, Game, Xoshiro128StarStar, legal_directions
from model.policy import PolicyNetwork, encode_boards
from scripts.generate_data import (
    EpisodeData,
    episode_sample_count,
    merge_episode_shards,
    save_episode_shard,
)
from teacher.expectimax import HEURISTIC_VERSION, ExpectimaxTeacher, ensure_native_teacher

ARRAY_NAMES = ("boards", "targets", "legal", "seeds", "steps", "sources", "priorities")
DAGGER_SOURCE = 2


@dataclass(frozen=True)
class DaggerCollectionConfig:
    profile: str
    round: int
    seed: int
    samples: int
    teacher_depth: int
    teacher_temperature: float
    student_checkpoint: str
    student_checkpoint_sha256: str
    expert_action_rate: float
    workers: int = 1
    teacher_version: str = HEURISTIC_VERSION


def _episode_seed(config: DaggerCollectionConfig, episode: int) -> int:
    split_order = (0, 8, 9, 1, 2, 3, 4, 5, 6, 7)
    cycle, offset = divmod(episode, len(split_order))
    return config.seed + cycle * 10 + split_order[offset]


_student_model_path: str | None = None
_student_model: PolicyNetwork | None = None


def _load_student(path: str) -> PolicyNetwork:
    global _student_model_path, _student_model
    if _student_model is None or _student_model_path != path:
        torch.set_num_threads(1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        stem_weight = state.get("stem.0.weight")
        if not isinstance(stem_weight, torch.Tensor):
            raise RuntimeError("student checkpoint has no stem weight")
        block_indices = {
            int(name.split(".")[1])
            for name in state
            if name.startswith("blocks.") and name.split(".")[1].isdigit()
        }
        model = PolicyNetwork(int(stem_weight.shape[0]), max(block_indices) + 1)
        model.load_state_dict(state)
        model.eval()
        _student_model = model
        _student_model_path = path
    return _student_model


def _student_action(
    model: PolicyNetwork, board: Board, directions: tuple[Direction, ...]
) -> Direction:
    encoded = encode_boards(np.asarray(board, dtype=np.uint32))
    with torch.inference_mode():
        logits = model(encoded)[0].numpy()
    masked = np.full(4, -np.inf, dtype=np.float32)
    masked[list(directions)] = logits[list(directions)]
    return Direction(int(masked.argmax()))


def _generate_dagger_episode(task: tuple[DaggerCollectionConfig, int]) -> EpisodeData:
    config, episode = task
    teacher = ExpectimaxTeacher(config.teacher_depth, config.teacher_temperature)
    student = _load_student(config.student_checkpoint)
    seed = _episode_seed(config, episode)
    rollout_random = Xoshiro128StarStar.from_seed(seed ^ 0xDA66_EE12)
    game = Game.start(seed)
    boards: list[NDArray[np.uint8]] = []
    targets: list[NDArray[np.float32]] = []
    masks: list[NDArray[np.uint8]] = []
    priorities: list[float] = []

    for _ in range(20_000):
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

        expert_choice = Direction(int(target.argmax()))
        student_choice = _student_action(student, board, directions)
        priority = 2.0 if student_choice != expert_choice else 1.0
        priorities.append(priority)
        choice = (
            expert_choice
            if rollout_random.next_float() < config.expert_action_rate
            else student_choice
        )
        game.move(choice)

    if legal_directions(game.state.board):
        raise RuntimeError(f"DAgger episode {episode} exceeded the 20,000 move safety limit")

    count = len(boards)
    return EpisodeData(
        episode=episode,
        boards=np.stack(boards).astype(np.uint8),
        targets=np.stack(targets).astype(np.float32),
        legal=np.stack(masks).astype(np.uint8),
        seeds=np.full(count, seed, dtype=np.uint32),
        steps=np.arange(count, dtype=np.uint16),
        sources=np.full(count, DAGGER_SOURCE, dtype=np.uint8),
        priorities=np.asarray(priorities, dtype=np.float32),
    )


class _InlineExecutor:
    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def map(
        self,
        function: Callable[[tuple[DaggerCollectionConfig, int]], EpisodeData],
        tasks: list[tuple[DaggerCollectionConfig, int]],
    ) -> Iterable[EpisodeData]:
        return [function(task) for task in tasks]


def _validate_collection_config(config: DaggerCollectionConfig) -> None:
    if config.round < 1:
        raise ValueError("DAgger round must be at least 1")
    if config.samples < 1:
        raise ValueError("DAgger samples must be positive")
    if not 0.0 <= config.expert_action_rate <= 1.0:
        raise ValueError("expert action rate must be in [0, 1]")
    checkpoint = Path(config.student_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"student checkpoint does not exist: {checkpoint}")
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != config.student_checkpoint_sha256:
        raise RuntimeError("student checkpoint does not match configured sha256")


def collect_dagger_dataset(
    config: DaggerCollectionConfig, output_directory: Path
) -> dict[str, object]:
    _validate_collection_config(config)
    ensure_native_teacher()
    output_directory.mkdir(parents=True, exist_ok=True)
    serialized_config = asdict(config)
    config_path = output_directory / "collection-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != serialized_config:
        raise RuntimeError("existing DAgger shards use a different collection config")
    config_path.write_text(json.dumps(serialized_config, indent=2) + "\n")

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
            for result in executor.map(_generate_dagger_episode, tasks):
                path = shard_directory / f"episode-{result.episode:06d}.npz"
                save_episode_shard(path, result)
                shards.append(path)
                generated += len(result.boards)
                episode += 1
            (output_directory / "progress.json").write_text(
                json.dumps({"completedEpisodes": episode, "generatedSamples": generated}) + "\n"
            )

    counts, hashes, _, _ = merge_episode_shards(shards, config.samples, output_directory)
    manifest: dict[str, object] = {
        "version": 1,
        "kind": "dagger-student-rollout-teacher-labels",
        "profile": config.profile,
        "round": config.round,
        "seed": config.seed,
        "samples": config.samples,
        "expertActionRate": config.expert_action_rate,
        "studentCheckpointSha256": config.student_checkpoint_sha256,
        "teacher": {
            "depth": config.teacher_depth,
            "temperature": config.teacher_temperature,
            "heuristicVersion": config.teacher_version,
        },
        "workers": workers,
        "resumableShards": len(shards),
        "counts": counts,
        "sha256": hashes,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def aggregate_datasets(
    input_directories: Sequence[Path], output_directory: Path, seed: int
) -> dict[str, object]:
    if not input_directories:
        raise ValueError("at least one input dataset is required")
    output_directory.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for split_index, split in enumerate(("train", "validation", "test")):
        chunks: dict[str, list[NDArray[np.generic]]] = {name: [] for name in ARRAY_NAMES}
        for directory in input_directories:
            with np.load(directory / f"{split}.npz") as archive:
                for name in ARRAY_NAMES:
                    if name == "priorities" and name not in archive:
                        chunks[name].append(np.ones(len(archive["boards"]), dtype=np.float32))
                    else:
                        chunks[name].append(archive[name])
        arrays = {name: np.concatenate(values) for name, values in chunks.items()}
        if split == "train":
            permutation = np.random.default_rng(seed + split_index).permutation(
                len(arrays["boards"])
            )
            arrays = {name: values[permutation] for name, values in arrays.items()}
        path = output_directory / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = len(arrays["boards"])
        unique, occurrences = np.unique(arrays["sources"], return_counts=True)
        for source, occurrence in zip(unique, occurrences, strict=True):
            key = str(int(source))
            source_counts[key] = source_counts.get(key, 0) + int(occurrence)

    inputs = []
    for directory in input_directories:
        manifest_path = directory / "manifest.json"
        inputs.append(
            {
                "path": directory.as_posix(),
                "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        )
    manifest: dict[str, object] = {
        "version": 1,
        "kind": "aggregated-dagger-dataset",
        "seed": seed,
        "samples": sum(counts.values()),
        "counts": counts,
        "sourceCounts": source_counts,
        "inputs": inputs,
        "sha256": hashes,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.config.read_text())
    checkpoint = arguments.checkpoint.resolve()
    config = DaggerCollectionConfig(
        profile=raw["profile"],
        round=int(raw.get("daggerRound", 1)),
        seed=int(raw.get("daggerSeed", raw["seed"] + 100_000)),
        samples=int(raw["daggerSamplesPerRound"]),
        teacher_depth=int(raw["teacherDepth"]),
        teacher_temperature=float(raw["teacherTemperature"]),
        student_checkpoint=checkpoint.as_posix(),
        student_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        expert_action_rate=float(raw["daggerExpertActionRate"]),
        workers=int(raw.get("daggerWorkers", raw["dataWorkers"])),
        teacher_version=raw.get("teacherVersion", HEURISTIC_VERSION),
    )
    print(json.dumps(collect_dagger_dataset(config, arguments.output), indent=2))


if __name__ == "__main__":
    _main()
