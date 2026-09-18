from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.exporter import export_policy
from model.policy import PolicyNetwork, SymmetryEnsemblePolicy
from scripts.pipeline import benchmark_candidate, publish_candidate, student_quality_gate
from scripts.train import TrainingConfig, train_policy


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--publish", type=Path, default=Path("../public/models"))
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    project_root = root.parent
    raw = json.loads((root / "configs" / f"{arguments.profile}.json").read_text())
    run_version = str(raw["daggerRunVersion"])
    source_root = root / "artifacts" / arguments.profile / f"dagger-{run_version}" / "round-2"
    source_checkpoint = source_root / "policy.pt"
    data_directory = (
        root / "datasets" / f"{arguments.profile}-dagger-{run_version}" / "round-2-aggregate"
    )
    output_directory = root / "artifacts" / arguments.profile / f"dagger-{run_version}" / "polish"
    checkpoint, training_metrics = train_policy(
        TrainingConfig(
            seed=int(raw["seed"]) + 10_000,
            epochs=int(raw["daggerPolishEpochs"]),
            batch_size=int(raw["batchSize"]),
            learning_rate=float(raw["daggerPolishLearningRate"]),
            weight_decay=float(raw["weightDecay"]),
            evaluation_games=int(raw["evaluationGames"]),
            initial_checkpoint=source_checkpoint,
            model_channels=int(raw["modelChannels"]),
            model_blocks=int(raw["modelBlocks"]),
        ),
        data_directory,
        output_directory,
    )
    model = PolicyNetwork(int(raw["modelChannels"]), int(raw["modelBlocks"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model_path, manifest_path = export_policy(
        SymmetryEnsemblePolicy(model),
        output_directory / "release-candidate",
        {
            "profile": arguments.profile,
            "seed": raw["seed"],
            "daggerRunVersion": run_version,
            "stage": "low-learning-rate-polish",
            "inferenceEnsemble": "eight-square-symmetries",
            "training": training_metrics,
        },
    )
    benchmark = benchmark_candidate(
        model_path,
        int(raw["benchmarkGames"]),
        project_root,
        int(raw["benchmarkWorkers"]),
    )
    benchmark["qualityGatePassed"] = student_quality_gate(benchmark)
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["benchmark"] = benchmark
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if not benchmark["qualityGatePassed"]:
        raise RuntimeError(f"polished student quality gate failed: {benchmark}")
    published = publish_candidate(model_path, manifest_path, arguments.publish.resolve())
    print(json.dumps({"benchmark": benchmark, "published": published.as_posix()}, indent=2))


if __name__ == "__main__":
    _main()
