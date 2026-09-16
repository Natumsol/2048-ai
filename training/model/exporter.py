from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


def export_policy(
    model: nn.Module,
    output_directory: Path,
    metadata: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "policy.onnx"
    manifest_path = output_directory / "policy.manifest.json"
    model = model.cpu().eval()
    example = torch.zeros((1, 16, 4, 4), dtype=torch.float32)
    example[:, 0] = 1
    torch.onnx.export(
        model,
        example,
        model_path.as_posix(),
        input_names=["board"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    graph = onnx.load(model_path)
    graph = onnx.version_converter.convert_version(graph, 18)
    onnx.save(graph, model_path)
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    fixture = torch.rand((1, 16, 4, 4), generator=torch.Generator().manual_seed(2048))
    fixture = fixture / fixture.sum(dim=1, keepdim=True)
    with torch.inference_mode():
        expected = model(fixture).numpy()
    actual = np.asarray(session.run(["logits"], {"board": fixture.numpy()})[0], dtype=np.float32)
    maximum_error = float(np.max(np.abs(expected - actual)))
    if maximum_error > 1e-5:
        raise RuntimeError(f"ONNX parity error {maximum_error:.3g} exceeds 1e-5")

    manifest: dict[str, object] = {
        "contractVersion": 1,
        "model": model_path.name,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input": {"name": "board", "shape": [1, 16, 4, 4], "dtype": "float32"},
        "output": {"name": "logits", "shape": [1, 4], "dtype": "float32"},
        "opset": 18,
        "exportedAt": datetime.now(UTC).isoformat(),
        "parityMaximumAbsoluteError": maximum_error,
        "metadata": metadata or {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return model_path, manifest_path
