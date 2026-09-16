from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from environment.game import Board, Direction, Game, GameState, RandomState


def test_python_rules_match_shared_browser_contract() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "rules.json"
    fixtures = json.loads(fixture_path.read_text())

    for fixture in fixtures:
        state = GameState(
            board=cast(Board, tuple(tuple(row) for row in fixture["board"])),
            score=fixture["score"],
            move_count=fixture["moveCount"],
            random_state=cast(RandomState, tuple(fixture["randomState"])),
        )
        result = Game.from_state(state).move(Direction[fixture["direction"].upper()])
        assert [list(row) for row in result.state.board] == fixture["expected"]["board"]
        assert result.state.score == fixture["expected"]["score"]
        assert result.state.move_count == fixture["expected"]["moveCount"]
        assert list(result.state.random_state) == fixture["expected"]["randomState"]


def test_python_replays_shared_fixed_seed_trajectory() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "trajectory.json"
    fixture = json.loads(fixture_path.read_text())
    game = Game.start(fixture["seed"])

    for direction in fixture["directions"]:
        game.move(Direction[direction.upper()])

    assert [list(row) for row in game.state.board] == fixture["expected"]["board"]
    assert game.state.score == fixture["expected"]["score"]
    assert game.state.move_count == fixture["expected"]["moveCount"]
    assert list(game.state.random_state) == fixture["expected"]["randomState"]
