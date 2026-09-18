from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from environment.game import Direction, Game, legal_directions
from model.policy import PolicyNetwork, encode_boards


class PolicyDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, path: Path) -> None:
        with np.load(path) as archive:
            self.boards = torch.from_numpy(archive["boards"].astype(np.uint8))
            self.targets = torch.from_numpy(archive["targets"].astype(np.float32))
            sources = archive["sources"].astype(np.uint8)
            priorities = (
                archive["priorities"].astype(np.float32)
                if "priorities" in archive
                else np.ones(len(self.boards), dtype=np.float32)
            )
        empty = (self.boards.numpy() == 0).sum(axis=(1, 2))
        maximum_rank = self.boards.numpy().max(axis=(1, 2))
        priorities *= np.where(sources == 2, 2.0, 1.0)
        priorities *= np.where(empty <= 2, 1.5, 1.0)
        priorities *= np.where(np.isin(maximum_rank, (9, 10)), 1.5, 1.0)
        self.weights = torch.from_numpy(np.minimum(priorities, 3.0).astype(np.float32))

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.boards[index], self.targets[index], self.weights[index]


_DIRECTION_PERMUTATIONS = torch.tensor(
    (
        (0, 1, 2, 3),
        (3, 0, 1, 2),
        (2, 3, 0, 1),
        (1, 2, 3, 0),
        (0, 3, 2, 1),
        (1, 0, 3, 2),
        (2, 1, 0, 3),
        (3, 2, 1, 0),
    ),
    dtype=torch.int64,
)


def _encode_exponents(exponents: Tensor) -> Tensor:
    return (
        torch.nn.functional.one_hot(exponents.to(torch.int64), num_classes=16)
        .permute(0, 3, 1, 2)
        .float()
    )


def _augment_batch(boards: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    transforms = torch.randint(0, 8, (len(boards),))
    augmented_boards = torch.empty_like(boards)
    augmented_targets = torch.empty_like(targets)
    for transform in range(8):
        selected = transforms == transform
        if not selected.any():
            continue
        transformed_boards = torch.rot90(boards[selected], transform % 4, dims=(1, 2))
        if transform >= 4:
            transformed_boards = torch.flip(transformed_boards, dims=(2,))
        augmented_boards[selected] = transformed_boards
        permutation = _DIRECTION_PERMUTATIONS[transform]
        transformed_targets = torch.empty_like(targets[selected])
        transformed_targets[:, permutation] = targets[selected]
        augmented_targets[selected] = transformed_targets
    return augmented_boards, augmented_targets


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    evaluation_games: int
    initial_checkpoint: Path | None = None
    model_channels: int = 64
    model_blocks: int = 4


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loss(logits: Tensor, targets: Tensor, weights: Tensor | None = None) -> Tensor:
    losses = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
    if weights is None:
        return losses.mean()
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _load_initial_weights(model: PolicyNetwork, checkpoint: Path) -> str:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise RuntimeError("policy checkpoint does not contain a state dictionary")
    current = model.state_dict()
    if not loaded.keys() <= current.keys():
        raise RuntimeError("policy checkpoint contains keys outside the configured architecture")
    if loaded.keys() == current.keys() and all(
        loaded[name].shape == tensor.shape for name, tensor in current.items()
    ):
        model.load_state_dict(loaded)
        return "strict"

    expanded = {}
    for name, target in current.items():
        if name not in loaded:
            expanded[name] = target
            continue
        source = loaded[name]
        if source.ndim != target.ndim or any(
            source_size > target_size
            for source_size, target_size in zip(source.shape, target.shape, strict=True)
        ):
            raise RuntimeError(f"cannot expand checkpoint tensor {name}")
        value = target.clone()
        if value.ndim >= 2:
            value *= 0.1
        slices = tuple(slice(0, size) for size in source.shape)
        value[slices] = source
        expanded[name] = value
    model.load_state_dict(expanded)
    return "expanded"


def _evaluate_gameplay(
    model: PolicyNetwork, device: torch.device, seeds: range
) -> dict[str, float]:
    games = [Game.start(seed) for seed in seeds]
    active = games.copy()
    model.eval()
    with torch.inference_mode():
        for _ in range(20_000):
            evaluable: list[tuple[Game, tuple[Direction, ...]]] = []
            for game in active:
                legal = legal_directions(game.state.board)
                if legal:
                    evaluable.append((game, legal))
            if not evaluable:
                break
            boards = np.asarray([game.state.board for game, _ in evaluable], dtype=np.uint32)
            logits_batch = model(encode_boards(boards).to(device)).cpu().numpy()
            active = []
            for (game, legal), logits in zip(evaluable, logits_batch, strict=True):
                masked = np.full(4, -np.inf, dtype=np.float32)
                masked[list(legal)] = logits[list(legal)]
                game.move(Direction(int(masked.argmax())))
                active.append(game)
        else:
            raise RuntimeError("gameplay evaluation exceeded the 20,000 move safety limit")
    maximum_tiles = [max(max(row) for row in game.state.board) for game in games]
    reached = sum(tile >= 2048 for tile in maximum_tiles)
    return {
        "evaluationReached2048Rate": reached / max(len(maximum_tiles), 1),
        "evaluationMedianMaximumTile": float(statistics.median(maximum_tiles)),
    }


def _evaluate_dataset(
    model: PolicyNetwork,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    total_loss = 0.0
    items = 0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for exponents, targets, _ in loader:
            boards = _encode_exponents(exponents).to(device)
            targets = targets.to(device)
            logits = model(boards)
            total_loss += float(_loss(logits, targets)) * len(boards)
            items += len(boards)
            correct += int((logits.argmax(dim=1) == targets.argmax(dim=1)).sum())
    return total_loss / max(items, 1), correct / max(items, 1)


def train_policy(
    config: TrainingConfig, data_directory: Path, output_directory: Path
) -> tuple[Path, dict[str, object]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device()
    model = PolicyNetwork(config.model_channels, config.model_blocks).to(device)
    initial_checkpoint_hash: str | None = None
    initial_checkpoint_mode: str | None = None
    if config.initial_checkpoint is not None:
        initial_checkpoint_mode = _load_initial_weights(model, config.initial_checkpoint)
        model.to(device)
        initial_checkpoint_hash = hashlib.sha256(config.initial_checkpoint.read_bytes()).hexdigest()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = DataLoader(
        PolicyDataset(data_directory / "train.npz"),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        PolicyDataset(data_directory / "validation.npz"),
        batch_size=config.batch_size,
    )
    best_rank = (-1.0, -1.0, float("-inf"))
    history: list[dict[str, float]] = []
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = output_directory / "policy.pt"
    for epoch in range(config.epochs):
        model.train()
        training_loss = 0.0
        training_items = 0
        for exponents, targets, weights in train_loader:
            exponents, targets = _augment_batch(exponents, targets)
            boards = _encode_exponents(exponents).to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(boards), targets, weights)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(boards)
            training_items += len(boards)

        model.eval()
        validation_loss = 0.0
        validation_items = 0
        correct = 0
        with torch.inference_mode():
            for exponents, targets, _ in validation_loader:
                boards = _encode_exponents(exponents).to(device)
                targets = targets.to(device)
                logits = model(boards)
                loss = _loss(logits, targets)
                validation_loss += float(loss) * len(boards)
                validation_items += len(boards)
                correct += int((logits.argmax(dim=1) == targets.argmax(dim=1)).sum())
        metrics = {
            "epoch": float(epoch + 1),
            "trainingLoss": training_loss / max(training_items, 1),
            "validationLoss": validation_loss / max(validation_items, 1),
            "validationTop1": correct / max(validation_items, 1),
            **_evaluate_gameplay(
                model,
                device,
                range(30_000, 30_000 + config.evaluation_games),
            ),
        }
        history.append(metrics)
        print(json.dumps(metrics))
        rank = (
            metrics["evaluationReached2048Rate"],
            metrics["evaluationMedianMaximumTile"],
            -metrics["validationLoss"],
        )
        if rank > best_rank:
            best_rank = rank
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loss, test_top1 = _evaluate_dataset(
        model,
        DataLoader(
            PolicyDataset(data_directory / "test.npz"),
            batch_size=config.batch_size,
        ),
        device,
    )

    summary: dict[str, object] = {
        "device": str(device),
        "seed": config.seed,
        "epochs": config.epochs,
        "batchSize": config.batch_size,
        "learningRate": config.learning_rate,
        "weightDecay": config.weight_decay,
        "initialCheckpoint": (
            {
                "path": config.initial_checkpoint.as_posix(),
                "sha256": initial_checkpoint_hash,
                "mode": initial_checkpoint_mode,
            }
            if config.initial_checkpoint is not None
            else None
        ),
        "checkpointSelection": "gameplay rate, median maximum tile, then validation loss",
        "modelChannels": config.model_channels,
        "modelBlocks": config.model_blocks,
        "bestRank": best_rank,
        "testSoftTargetLoss": test_loss,
        "testTop1": test_top1,
        "history": history,
    }
    (output_directory / "training-metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return checkpoint, summary


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.config.read_text())
    config = TrainingConfig(
        seed=raw["seed"],
        epochs=raw["epochs"],
        batch_size=raw["batchSize"],
        learning_rate=raw["learningRate"],
        weight_decay=raw["weightDecay"],
        evaluation_games=raw["evaluationGames"],
        initial_checkpoint=(
            Path(raw["initialCheckpoint"]) if raw.get("initialCheckpoint") else None
        ),
        model_channels=int(raw.get("modelChannels", 64)),
        model_blocks=int(raw.get("modelBlocks", 4)),
    )
    train_policy(config, arguments.data, arguments.output)


if __name__ == "__main__":
    _main()
