from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.pipeline import publish_candidate, student_quality_gate


def test_student_quality_gate_rejects_each_failed_threshold() -> None:
    passing: dict[str, object] = {
        "illegalMoves": 0,
        "truncatedGames": 0,
        "reached2048Rate": 0.5,
        "medianMaximumTile": 1024,
        "decisionP95Ms": 50,
    }
    assert student_quality_gate(passing)
    for name, failed in {
        "illegalMoves": 1,
        "truncatedGames": 1,
        "reached2048Rate": 0.49,
        "medianMaximumTile": 512,
        "decisionP95Ms": 50.1,
    }.items():
        metrics = {**passing, name: failed}
        assert not student_quality_gate(metrics)


def test_publish_candidate_uses_content_hash_and_updates_manifest_last(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    published = tmp_path / "published"
    candidate.mkdir()
    model = candidate / "policy.onnx"
    model.write_bytes(b"onnx fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = candidate / "policy.manifest.json"
    manifest.write_text(json.dumps({"sha256": digest, "model": "policy.onnx"}))

    published_model = publish_candidate(model, manifest, published)

    assert published_model.name == f"policy.{digest[:12]}.onnx"
    assert published_model.read_bytes() == model.read_bytes()
    published_manifest = json.loads((published / "policy.manifest.json").read_text())
    assert published_manifest["model"] == published_model.name
    assert not list(published.glob(".*.tmp"))
