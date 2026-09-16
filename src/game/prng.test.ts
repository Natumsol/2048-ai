import { describe, expect, it } from 'vitest';

import fixtures from '../../fixtures/prng.json';
import { Xoshiro128StarStar } from './prng';

describe('Xoshiro128StarStar shared contract', () => {
  it.each(fixtures)('replays seed $seed exactly', ({ seed, outputs }) => {
    const random = Xoshiro128StarStar.fromSeed(seed);

    expect(outputs.map(() => random.nextUint32())).toEqual(outputs);
  });
});
