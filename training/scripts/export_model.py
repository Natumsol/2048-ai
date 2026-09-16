from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.exporter import export_policy
from model.policy import PolicyNetwork


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    arguments = parser.parse_args()
    model = PolicyNetwork()
    model.load_state_dict(torch.load(arguments.checkpoint, map_location="cpu", weights_only=True))
    metadata: dict[str, object] = {}
    if arguments.metadata:
        metadata = json.loads(arguments.metadata.read_text())
    paths = export_policy(model, arguments.output, metadata)
    print("\n".join(path.as_posix() for path in paths))


if __name__ == "__main__":
    _main()
