from __future__ import annotations

import json
from pathlib import Path

from environment.game import Xoshiro128StarStar


def test_python_prng_matches_shared_browser_contract() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "prng.json"
    fixtures = json.loads(fixture_path.read_text())

    for fixture in fixtures:
        random = Xoshiro128StarStar.from_seed(fixture["seed"])
        assert [random.next_uint32() for _ in fixture["outputs"]] == fixture["outputs"]
