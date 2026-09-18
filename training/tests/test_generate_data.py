from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from scripts.generate_data import GenerationConfig, generate_dataset


def test_generation_uses_multiple_seed_splits_and_both_sources(tmp_path: Path) -> None:
    config = GenerationConfig("test", 100, 3072, 1, 64.0, workers=2)
    manifest = generate_dataset(config, tmp_path)

    assert manifest["samples"] == 3072
    assert cast(int, manifest["normalSamples"]) > 0
    assert cast(int, manifest["perturbedSamples"]) > 0
    observed_sources: set[int] = set()
    for split in ("train", "validation", "test"):
        archive = np.load(tmp_path / f"{split}.npz")
        assert len(archive["boards"]) > 0
        assert archive["boards"].dtype == np.uint8
        observed_sources.update(archive["sources"].tolist())
    assert observed_sources == {0, 1}

    resumed = generate_dataset(config, tmp_path)
    assert resumed["sha256"] == manifest["sha256"]
    assert resumed["resumableShards"] == manifest["resumableShards"]
