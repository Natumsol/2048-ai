from __future__ import annotations

import argparse
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

from data.dataset import exponents_to_boards, transform_sample
from environment.game import Direction, Game, legal_directions
from model.policy import PolicyNetwork, encode_boards


class PolicyDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, path: Path, augment: bool) -> None:
        archive = np.load(path)
        self.boards = archive["boards"].astype(np.uint8)
        self.targets = archive["targets"].astype(np.float32)
        self.legal = archive["legal"].astype(np.uint8)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        board = self.boards[index]
        target = self.targets[index]
        legal = self.legal[index]
        if self.augment:
            board, target, _ = transform_sample(board, target, legal, random.randrange(8))
        values = exponents_to_boards(board)
        return encode_boards(values)[0], torch.from_numpy(target.copy())


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    evaluation_games: int


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loss(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _evaluate_gameplay(
    model: PolicyNetwork, device: torch.device, seeds: range
) -> dict[str, float]:
    maximum_tiles: list[int] = []
    model.eval()
    with torch.inference_mode():
        for seed in seeds:
            game = Game.start(seed)
            for _ in range(20_000):
                legal = legal_directions(game.state.board)
                if not legal:
                    break
                board = np.asarray(game.state.board, dtype=np.uint32)
                logits = model(encode_boards(board).to(device))[0].cpu().numpy()
                masked = np.full(4, -np.inf, dtype=np.float32)
                masked[list(legal)] = logits[list(legal)]
                game.move(Direction(int(masked.argmax())))
            maximum_tiles.append(max(max(row) for row in game.state.board))
    reached = sum(tile >= 2048 for tile in maximum_tiles)
    return {
        "evaluationReached2048Rate": reached / max(len(maximum_tiles), 1),
        "evaluationMedianMaximumTile": float(statistics.median(maximum_tiles)),
    }


def _evaluate_dataset(
    model: PolicyNetwork, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device
) -> tuple[float, float]:
    total_loss = 0.0
    items = 0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for boards, targets in loader:
            boards, targets = boards.to(device), targets.to(device)
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
    model = PolicyNetwork().to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = DataLoader(
        PolicyDataset(data_directory / "train.npz", augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        PolicyDataset(data_directory / "validation.npz", augment=False),
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
        for boards, targets in train_loader:
            boards, targets = boards.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(boards), targets)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(boards)
            training_items += len(boards)

        model.eval()
        validation_loss = 0.0
        validation_items = 0
        correct = 0
        with torch.inference_mode():
            for boards, targets in validation_loader:
                boards, targets = boards.to(device), targets.to(device)
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
            PolicyDataset(data_directory / "test.npz", augment=False),
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
        "checkpointSelection": "gameplay rate, median maximum tile, then validation loss",
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
    )
    train_policy(config, arguments.data, arguments.output)


if __name__ == "__main__":
    _main()
