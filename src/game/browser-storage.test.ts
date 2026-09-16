import { beforeEach, describe, expect, it, vi } from 'vitest';

import { BrowserGameStorage } from './browser-storage';

const key = 'test-session';

describe('BrowserGameStorage', () => {
  let values: Map<string, string>;

  beforeEach(() => {
    values = new Map();
    vi.stubGlobal('localStorage', {
      getItem: (name: string) => values.get(name) ?? null,
      setItem: (name: string, value: string) => values.set(name, value),
    });
  });

  it('accepts a valid persisted session', () => {
    values.set(
      key,
      JSON.stringify({
        game: {
          board: [
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
          ],
          score: 4,
          moveCount: 2,
          phase: 'playing',
          randomState: [0, 1, 2, 0xffff_ffff],
          hasShown2048: false,
        },
        bestScore: 8,
        speed: 'standard',
      }),
    );

    expect(new BrowserGameStorage(key).load()).toMatchObject({
      game: { score: 4 },
      bestScore: 8,
      speed: 'standard',
    });
  });

  it.each([
    { path: 'tile', tile: 3, score: 0, bestScore: 0, randomPart: 1 },
    { path: 'score', tile: 2, score: -1, bestScore: 0, randomPart: 1 },
    { path: 'best score', tile: 2, score: 4, bestScore: 2, randomPart: 1 },
    { path: 'PRNG', tile: 2, score: 0, bestScore: 0, randomPart: 2 ** 32 },
  ])(
    'rejects an invalid $path at the storage boundary',
    ({ tile, score, bestScore, randomPart }) => {
      values.set(
        key,
        JSON.stringify({
          game: {
            board: [
              [tile, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
            ],
            score,
            moveCount: 0,
            phase: 'playing',
            randomState: [randomPart, 1, 2, 3],
            hasShown2048: false,
          },
          bestScore,
          speed: 'standard',
        }),
      );

      expect(new BrowserGameStorage(key).load()).toBeUndefined();
    },
  );
});
