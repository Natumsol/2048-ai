from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from model.exporter import export_policy
from model.policy import PolicyNetwork, SymmetryEnsemblePolicy
from scripts.pipeline import benchmark_candidate, publish_candidate, student_quality_gate


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--publish", type=Path, default=Path("../public/models"))
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    project_root = root.parent
    raw = json.loads((root / "configs" / f"{arguments.profile}.json").read_text())
    run_version = str(raw["daggerRunVersion"])
    source_directory = root / "artifacts" / arguments.profile / f"dagger-{run_version}" / "polish"
    checkpoint = source_directory / "policy.pt"
    training_metrics = json.loads((source_directory / "training-metrics.json").read_text())

    policy = PolicyNetwork(int(raw["modelChannels"]), int(raw["modelBlocks"]))
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    output_directory = (
        root / "artifacts" / arguments.profile / f"dagger-{run_version}" / "symmetry-ensemble"
    )
    model_path, manifest_path = export_policy(
        SymmetryEnsemblePolicy(policy),
        output_directory / "release-candidate",
        {
            "profile": arguments.profile,
            "seed": raw["seed"],
            "daggerRunVersion": run_version,
            "stage": "eight-square-symmetry-ensemble",
            "sourceCheckpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
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
        raise RuntimeError(f"symmetry ensemble quality gate failed: {benchmark}")
    published = publish_candidate(model_path, manifest_path, arguments.publish.resolve())
    print(json.dumps({"benchmark": benchmark, "published": published.as_posix()}, indent=2))


if __name__ == "__main__":
    _main()
