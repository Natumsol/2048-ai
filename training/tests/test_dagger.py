from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model.policy import PolicyNetwork
from scripts.dagger import DaggerCollectionConfig, aggregate_datasets, collect_dagger_dataset


def _write_dataset(directory: Path, source: int) -> None:
    directory.mkdir(parents=True)
    for split_index, split in enumerate(("train", "validation", "test")):
        count = split_index + 1
        np.savez_compressed(
            directory / f"{split}.npz",
            boards=np.full((count, 4, 4), source, dtype=np.uint8),
            targets=np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (count, 1)),
            legal=np.ones((count, 4), dtype=np.uint8),
            seeds=np.arange(count, dtype=np.uint32),
            steps=np.arange(count, dtype=np.uint16),
            sources=np.full(count, source, dtype=np.uint8),
        )
    (directory / "manifest.json").write_text(json.dumps({"source": source}) + "\n")


def test_aggregate_datasets_keeps_base_and_dagger_samples(tmp_path: Path) -> None:
    base = tmp_path / "base"
    dagger = tmp_path / "dagger"
    output = tmp_path / "aggregate"
    _write_dataset(base, 0)
    _write_dataset(dagger, 2)

    manifest = aggregate_datasets([base, dagger], output, seed=2048)

    assert manifest["samples"] == 12
    assert manifest["counts"] == {"train": 2, "validation": 4, "test": 6}
    assert manifest["sourceCounts"] == {"0": 6, "2": 6}
    with np.load(output / "train.npz") as archive:
        assert set(archive["sources"].tolist()) == {0, 2}


def test_collect_dagger_dataset_uses_student_rollouts_and_teacher_labels(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "student.pt"
    torch.save(PolicyNetwork().state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "rollouts"
    config = DaggerCollectionConfig(
        profile="test",
        round=1,
        seed=100,
        samples=32,
        teacher_depth=1,
        teacher_temperature=64.0,
        student_checkpoint=checkpoint.as_posix(),
        student_checkpoint_sha256=digest,
        expert_action_rate=0.0,
    )

    manifest = collect_dagger_dataset(config, output)

    assert manifest["samples"] == 32
    assert manifest["kind"] == "dagger-student-rollout-teacher-labels"
    with np.load(output / "train.npz") as archive:
        assert set(archive["sources"].tolist()) == {2}
        np.testing.assert_allclose(archive["targets"].sum(axis=1), 1.0)
        assert np.all(archive["targets"][archive["legal"] == 0] == 0)
