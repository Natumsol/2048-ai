from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from model.exporter import export_policy
from model.policy import PolicyNetwork
from scripts.dagger import DaggerCollectionConfig, aggregate_datasets, collect_dagger_dataset
from scripts.pipeline import benchmark_candidate, publish_candidate, student_quality_gate
from scripts.train import TrainingConfig, train_policy
from teacher.expectimax import HEURISTIC_VERSION


def _checkpoint_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _best_gameplay_rate(training_metrics: dict[str, object]) -> float:
    best_rank = training_metrics.get("bestRank")
    if not isinstance(best_rank, list | tuple) or not best_rank:
        return 0.0
    value = best_rank[0]
    return float(value) if isinstance(value, int | float) else 0.0


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--publish", type=Path, default=Path("../public/models"))
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    project_root = root.parent
    raw = json.loads((root / "configs" / f"{arguments.profile}.json").read_text())
    run_version = str(raw.get("daggerRunVersion", "v1"))
    rounds = int(raw["daggerRounds"])
    if rounds < 1:
        raise ValueError("daggerRounds must be positive")

    base_data_directory = root / "datasets" / arguments.profile
    base_artifact_directory = root / "artifacts" / arguments.profile
    previous_checkpoint = base_artifact_directory / "policy.pt"
    if not previous_checkpoint.is_file():
        raise FileNotFoundError("baseline policy.pt is missing; run train:full before DAgger")

    rollout_directories: list[Path] = []
    final_model_path: Path | None = None
    final_manifest_path: Path | None = None

    for round_number in range(1, rounds + 1):
        round_root = base_artifact_directory / f"dagger-{run_version}" / f"round-{round_number}"
        rollout_directory = (
            root
            / "datasets"
            / f"{arguments.profile}-dagger-{run_version}"
            / f"round-{round_number}-rollouts"
        )
        aggregate_directory = (
            root
            / "datasets"
            / f"{arguments.profile}-dagger-{run_version}"
            / f"round-{round_number}-aggregate"
        )
        expert_action_rate = float(raw["daggerExpertActionRate"]) * (0.5 ** (round_number - 1))
        collection_config = DaggerCollectionConfig(
            profile=arguments.profile,
            round=round_number,
            seed=int(raw["daggerSeed"]) + (round_number - 1) * 10_000,
            samples=int(raw["daggerSamplesPerRound"]),
            teacher_depth=int(raw["teacherDepth"]),
            teacher_temperature=float(raw["teacherTemperature"]),
            student_checkpoint=previous_checkpoint.resolve().as_posix(),
            student_checkpoint_sha256=_checkpoint_hash(previous_checkpoint),
            expert_action_rate=expert_action_rate,
            workers=int(raw.get("daggerWorkers", raw["dataWorkers"])),
            teacher_version=raw.get("teacherVersion", HEURISTIC_VERSION),
        )
        rollout_manifest = collect_dagger_dataset(collection_config, rollout_directory)
        rollout_directories.append(rollout_directory)
        aggregate_manifest = aggregate_datasets(
            [base_data_directory, *rollout_directories],
            aggregate_directory,
            int(raw["seed"]) + round_number,
        )

        checkpoint, training_metrics = train_policy(
            TrainingConfig(
                seed=int(raw["seed"]) + round_number,
                epochs=int(raw["daggerEpochsPerRound"]),
                batch_size=int(raw["batchSize"]),
                learning_rate=float(raw["daggerLearningRate"]),
                weight_decay=float(raw["weightDecay"]),
                evaluation_games=int(raw["evaluationGames"]),
                initial_checkpoint=previous_checkpoint,
                model_channels=int(raw.get("modelChannels", 64)),
                model_blocks=int(raw.get("modelBlocks", 4)),
            ),
            aggregate_directory,
            round_root,
        )
        previous_checkpoint = checkpoint

        model = PolicyNetwork(int(raw.get("modelChannels", 64)), int(raw.get("modelBlocks", 4)))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        candidate_directory = round_root / "release-candidate"
        final_model_path, final_manifest_path = export_policy(
            model,
            candidate_directory,
            {
                "profile": arguments.profile,
                "seed": raw["seed"],
                "dagger": {
                    "round": round_number,
                    "expertActionRate": expert_action_rate,
                    "rollout": rollout_manifest,
                    "aggregate": aggregate_manifest,
                },
                "training": training_metrics,
            },
        )
        print(
            json.dumps(
                {
                    "daggerRound": round_number,
                    "bestEvaluationReached2048Rate": _best_gameplay_rate(training_metrics),
                    "checkpoint": checkpoint.as_posix(),
                }
            ),
            flush=True,
        )

    if final_model_path is None or final_manifest_path is None:
        raise RuntimeError("DAgger produced no candidate model")

    final_benchmark = benchmark_candidate(
        final_model_path,
        int(raw["benchmarkGames"]),
        project_root,
        int(raw.get("benchmarkWorkers", 1)),
    )
    final_benchmark["qualityGatePassed"] = student_quality_gate(final_benchmark)
    manifest = json.loads(final_manifest_path.read_text())
    manifest["metadata"]["benchmark"] = final_benchmark
    final_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if not final_benchmark["qualityGatePassed"]:
        raise RuntimeError(f"DAgger student quality gate failed: {final_benchmark}")

    published = publish_candidate(
        final_model_path, final_manifest_path, arguments.publish.resolve()
    )
    print(json.dumps({"benchmark": final_benchmark, "published": published.as_posix()}, indent=2))


if __name__ == "__main__":
    _main()
