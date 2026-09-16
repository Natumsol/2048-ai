import { describe, expect, it } from 'vitest';

import { PolicyModel, type PolicyBackend } from './policy-model';

describe('PolicyModel', () => {
  it('encodes the model contract and ignores the highest-scoring illegal move', async () => {
    let observedInput: Float32Array | undefined;
    const backend: PolicyBackend = {
      name: 'test',
      infer(input) {
        observedInput = input;
        return Promise.resolve(new Float32Array([99, 2, 1, 0]));
      },
    };
    const model = new PolicyModel(backend);
    const board = [
      [2, 0, 4, 8],
      [16, 0, 32, 64],
      [128, 0, 256, 512],
      [1024, 0, 2, 4],
    ] as const;

    const decision = await model.decide(board);

    expect(decision.direction).toBe('right');
    expect(decision.backend).toBe('test');
    expect(observedInput).toHaveLength(16 * 4 * 4);
    expect(observedInput?.[1 * 16]).toBe(1);
    expect(observedInput?.reduce((sum, value) => sum + value, 0)).toBe(16);
  });
});
