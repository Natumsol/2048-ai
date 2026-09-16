import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';

import { legalDirections } from '../game/game-session';
import { PolicyModel } from './policy-model';

describe('published ONNX policy artifact', () => {
  it('matches its hash and actual runtime contract, then chooses a legal direction', async () => {
    const directory = new URL('../../public/models/', import.meta.url);
    const manifest = JSON.parse(
      await readFile(new URL('policy.manifest.json', directory), 'utf8'),
    ) as { model: string; sha256: string };
    const bytes = await readFile(new URL(manifest.model, directory));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(manifest.sha256);

    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    expect(session.inputMetadata).toMatchObject([
      { name: 'board', isTensor: true, type: 'float32', shape: [1, 16, 4, 4] },
    ]);
    expect(session.outputMetadata).toMatchObject([
      { name: 'logits', isTensor: true, type: 'float32', shape: [1, 4] },
    ]);
    const policy = new PolicyModel({
      name: 'artifact-test',
      async infer(input) {
        const output = await session.run({
          board: new ort.Tensor('float32', input, [1, 16, 4, 4]),
        });
        const logits = output.logits;
        if (!logits || !(logits.data instanceof Float32Array)) {
          throw new Error('invalid fixture output');
        }
        return new Float32Array(logits.data);
      },
    });
    const board = [
      [2, 0, 4, 8],
      [16, 0, 32, 64],
      [128, 0, 256, 512],
      [1024, 0, 2, 4],
    ] as const;

    const decision = await policy.decide(board);

    expect(legalDirections(board)).toContain(decision.direction);
    await session.release();
  });
});
