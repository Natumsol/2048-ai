import { describe, expect, it } from 'vitest';
import fc from 'fast-check';

import { GameSession, legalDirections, type Board } from './game-session';

describe('GameSession', () => {
  it('starts a reproducible 对局 with two tiles and no score', () => {
    const first = GameSession.start({ seed: 2048 }).snapshot();
    const replay = GameSession.start({ seed: 2048 }).snapshot();

    expect(first).toEqual(replay);
    expect(first.score).toBe(0);
    expect(first.moveCount).toBe(0);
    expect(first.phase).toBe('playing');
    expect(first.board.flat().filter((tile) => tile !== 0)).toHaveLength(2);
  });

  it('merges each tile once, scores the merge, and spawns after a legal move', () => {
    const session = GameSession.restore({
      board: [
        [2, 2, 4, 8],
        [16, 32, 64, 128],
        [256, 512, 1024, 2],
        [4, 8, 16, 32],
      ],
      score: 10,
      moveCount: 7,
      phase: 'playing',
      randomState: [1, 2, 3, 4],
      hasShown2048: false,
    });

    const transition = session.move('left');

    expect(transition.moved).toBe(true);
    expect(transition.snapshot.score).toBe(14);
    expect(transition.snapshot.moveCount).toBe(8);
    expect(transition.snapshot.board[0]?.slice(0, 3)).toEqual([4, 4, 8]);
    expect([2, 4]).toContain(transition.snapshot.board[0]?.[3]);
    expect(transition.events).toContainEqual({
      type: 'merge',
      from: [
        [0, 0],
        [0, 1],
      ],
      to: [0, 0],
      value: 4,
    });
    expect(transition.events.at(-1)).toMatchObject({ type: 'spawn', at: [0, 3] });
  });

  it('does not chain-merge a tile twice in one move', () => {
    const session = GameSession.restore({
      board: [
        [2, 2, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      score: 0,
      moveCount: 0,
      phase: 'playing',
      randomState: [1, 2, 3, 4],
      hasShown2048: false,
    });

    const transition = session.move('left');

    expect(transition.snapshot.board[0]?.slice(0, 2)).toEqual([4, 4]);
    expect(transition.snapshot.score).toBe(4);
  });

  it('changes total tile value only by the spawned 2 or 4 for arbitrary legal moves', () => {
    const tile = fc
      .integer({ min: 0, max: 11 })
      .map((exponent) => (exponent === 0 ? 0 : 2 ** exponent));
    fc.assert(
      fc.property(fc.array(tile, { minLength: 16, maxLength: 16 }), (values) => {
        const board = Array.from({ length: 4 }, (_, row) =>
          values.slice(row * 4, row * 4 + 4),
        ) as Board;
        const direction = legalDirections(board)[0];
        if (!direction) return;
        const before = board.flat().reduce((sum, value) => sum + value, 0);
        const session = GameSession.restore({
          board,
          score: 0,
          moveCount: 0,
          phase: 'playing',
          randomState: [11, 22, 33, 44],
          hasShown2048: false,
        });

        const transition = session.move(direction);
        const after = transition.snapshot.board.flat().reduce((sum, value) => sum + value, 0);

        expect(transition.moved).toBe(true);
        expect([2, 4]).toContain(after - before);
      }),
      { numRuns: 200 },
    );
  });
});
