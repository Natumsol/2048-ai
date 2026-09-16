import { describe, expect, it } from 'vitest';

import fixtures from '../../fixtures/rules.json';
import trajectory from '../../fixtures/trajectory.json';
import { GameSession, type Direction } from './game-session';
import type { RandomState } from './prng';

const asRandomState = (values: readonly number[]): RandomState => {
  if (values.length !== 4) throw new Error('PRNG fixture state must have four values');
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0, values[3] ?? 0];
};

describe('shared TypeScript and Python rule contract', () => {
  it.each(fixtures)('$name', (fixture) => {
    const session = GameSession.restore({
      board: fixture.board,
      score: fixture.score,
      moveCount: fixture.moveCount,
      phase: 'playing',
      randomState: asRandomState(fixture.randomState),
      hasShown2048: false,
    });

    const transition = session.move(fixture.direction as Direction);

    expect(transition.moved).toBe(true);
    expect(transition.snapshot).toMatchObject({
      board: fixture.expected.board,
      score: fixture.expected.score,
      moveCount: fixture.expected.moveCount,
      randomState: asRandomState(fixture.expected.randomState),
    });
  });

  it('replays a fixed-seed multi-move trajectory', () => {
    const session = GameSession.start({ seed: trajectory.seed });

    for (const direction of trajectory.directions) session.move(direction as Direction);

    expect(session.snapshot()).toMatchObject({
      board: trajectory.expected.board,
      score: trajectory.expected.score,
      moveCount: trajectory.expected.moveCount,
      randomState: asRandomState(trajectory.expected.randomState),
    });
  });
});
