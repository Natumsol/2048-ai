from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model.exporter import export_policy
from model.policy import PolicyNetwork


def test_exported_policy_matches_pytorch_and_writes_manifest(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = PolicyNetwork().eval()
    board = torch.zeros((1, 16, 4, 4), dtype=torch.float32)
    board[:, 0] = 1

    model_path, manifest_path = export_policy(model, tmp_path, {"profile": "test"})

    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(["logits"], {"board": board.numpy()})[0], dtype=np.float32)
    with torch.inference_mode():
        expected = model(board).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["contractVersion"] == 1
    assert manifest["input"]["name"] == "board"
    assert manifest["input"]["shape"] == [1, 16, 4, 4]
    assert manifest["output"]["name"] == "logits"
    assert manifest["output"]["shape"] == [1, 4]
    assert onnx.load(model_path).opset_import[0].version == 18
    assert manifest["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
