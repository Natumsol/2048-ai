from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from model.exporter import export_policy
from model.policy import PolicyNetwork
from scripts.benchmark import benchmark_teacher
from scripts.generate_data import GenerationConfig, generate_dataset
from scripts.train import TrainingConfig, train_policy


def _numeric_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, int | float):
        raise RuntimeError(f"benchmark metric {name} is not numeric")
    return float(value)


def student_quality_gate(metrics: dict[str, object]) -> bool:
    return (
        _numeric_metric(metrics, "illegalMoves") == 0
        and _numeric_metric(metrics, "truncatedGames") == 0
        and _numeric_metric(metrics, "reached2048Rate") >= 0.5
        and _numeric_metric(metrics, "medianMaximumTile") >= 1024
        and _numeric_metric(metrics, "decisionP95Ms") <= 50
    )


def publish_candidate(model_path: Path, manifest_path: Path, publish_directory: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    sha256 = manifest.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise RuntimeError("candidate manifest has no valid sha256")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != sha256:
        raise RuntimeError("candidate model does not match manifest sha256")
    publish_directory.mkdir(parents=True, exist_ok=True)
    model_name = f"policy.{sha256[:12]}.onnx"
    manifest["model"] = model_name
    temporary_model = publish_directory / f".{model_name}.tmp"
    temporary_manifest = publish_directory / ".policy.manifest.json.tmp"
    shutil.copy2(model_path, temporary_model)
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary_model, publish_directory / model_name)
    os.replace(temporary_manifest, publish_directory / "policy.manifest.json")
    return publish_directory / model_name


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--publish", type=Path, default=Path("../public/models"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / f"{arguments.profile}.json"
    raw = json.loads(config_path.read_text())
    data_directory = root / "datasets" / arguments.profile
    artifact_directory = root / "artifacts" / arguments.profile
    teacher_benchmark: dict[str, object] | None = None
    if arguments.profile == "full":
        teacher_benchmark = benchmark_teacher(
            int(raw["teacherDepth"]),
            float(raw["teacherTemperature"]),
            range(40_000, 41_000),
        )
        teacher_passed = (
            _numeric_metric(teacher_benchmark, "reached2048Rate") >= 0.9
            and _numeric_metric(teacher_benchmark, "medianMaximumTile") >= 2048
            and _numeric_metric(teacher_benchmark, "truncatedGames") == 0
        )
        if not teacher_passed:
            raise RuntimeError(f"teacher quality gate failed: {teacher_benchmark}")
    data_manifest = generate_dataset(
        GenerationConfig(
            raw["profile"],
            raw["seed"],
            raw["samples"],
            raw["teacherDepth"],
            raw["teacherTemperature"],
            raw["dataWorkers"],
        ),
        data_directory,
    )
    checkpoint, training_metrics = train_policy(
        TrainingConfig(
            raw["seed"],
            raw["epochs"],
            raw["batchSize"],
            raw["learningRate"],
            raw["weightDecay"],
            raw["evaluationGames"],
        ),
        data_directory,
        artifact_directory,
    )
    model = PolicyNetwork()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    candidate_directory = artifact_directory / "release-candidate"
    model_path, manifest_path = export_policy(
        model,
        candidate_directory,
        {
            "profile": arguments.profile,
            "seed": raw["seed"],
            "data": data_manifest,
            "training": training_metrics,
            "teacherBenchmark": teacher_benchmark,
        },
    )
    project_root = root.parent
    completed = subprocess.run(
        [
            (project_root / "node_modules" / ".bin" / "tsx").as_posix(),
            (project_root / "scripts" / "benchmark-model.ts").as_posix(),
            "--model",
            model_path.as_posix(),
            "--games",
            str(raw["benchmarkGames"]),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    benchmark = json.loads(completed.stdout)
    quality_gate_passed = student_quality_gate(benchmark)
    benchmark["qualityGatePassed"] = quality_gate_passed
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["benchmark"] = benchmark
    if arguments.profile == "full" and not quality_gate_passed:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise RuntimeError(f"student quality gate failed: {benchmark}")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    publish_candidate(model_path, manifest_path, arguments.publish.resolve())
    print(json.dumps({"benchmark": benchmark}, indent=2))


if __name__ == "__main__":
    _main()
