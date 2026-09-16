from environment.game import Direction, Game, GameState


def test_move_merges_each_tile_once_and_scores_known_example() -> None:
    game = Game.from_state(
        GameState(
            board=(
                (2, 2, 4, 8),
                (16, 32, 64, 128),
                (256, 512, 1024, 2),
                (4, 8, 16, 32),
            ),
            score=10,
            move_count=7,
            random_state=(1, 2, 3, 4),
        )
    )

    transition = game.move(Direction.LEFT)

    assert transition.moved is True
    assert transition.state.score == 14
    assert transition.state.move_count == 8
    assert transition.state.board[0][:3] == (4, 4, 8)
    assert transition.state.board[0][3] in (2, 4)
