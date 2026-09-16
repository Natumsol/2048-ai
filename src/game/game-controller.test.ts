import { describe, expect, it } from 'vitest';

import { PolicyModel } from '../ai/policy-model';
import { GameController } from './game-controller';

describe('GameController', () => {
  it('persists a newly created zero-move session immediately', () => {
    let saved: unknown;

    GameController.start(2048, {
      storage: {
        load: () => undefined,
        save: (snapshot) => {
          saved = snapshot;
        },
      },
    });

    expect(saved).toMatchObject({ game: { moveCount: 0, score: 0 }, bestScore: 0 });
  });

  it('lets the policy model advance an existing 对局 and exposes the decision', async () => {
    const policy = new PolicyModel({
      name: 'wasm',
      infer() {
        return Promise.resolve(new Float32Array([50, 8, 1, 0]));
      },
    });
    const controller = GameController.restore(
      {
        board: [
          [2, 0, 4, 8],
          [16, 0, 32, 64],
          [128, 0, 256, 512],
          [1024, 0, 2, 4],
        ],
        score: 0,
        moveCount: 0,
        phase: 'playing',
        randomState: [1, 2, 3, 4],
        hasShown2048: false,
      },
      { policy },
    );

    controller.startAutoPlay();
    const transition = await controller.stepAutoPlay();

    expect(transition?.moved).toBe(true);
    expect(controller.snapshot()).toMatchObject({
      control: 'ai',
      aiStatus: 'running',
      lastDirection: 'right',
      backend: 'wasm',
      game: { moveCount: 1 },
    });
  });

  it('persists a legal manual move and keeps the best score', () => {
    let saved: unknown;
    const controller = GameController.restore(
      {
        board: [
          [2, 2, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
        ],
        score: 40,
        moveCount: 3,
        phase: 'playing',
        randomState: [8, 6, 7, 5],
        hasShown2048: false,
      },
      {
        storage: {
          load: () => undefined,
          save: (snapshot) => {
            saved = snapshot;
          },
        },
      },
    );

    controller.move('left');

    expect(saved).toMatchObject({ game: { score: 44, moveCount: 4 }, bestScore: 44 });
    expect(controller.snapshot().bestScore).toBe(44);
  });

  it('restores the saved 对局 but always resumes under manual control', () => {
    const stored = {
      game: {
        board: [
          [2, 4, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
        ],
        score: 28,
        moveCount: 9,
        phase: 'playing' as const,
        randomState: [10, 20, 30, 40] as const,
        hasShown2048: false,
      },
      bestScore: 400,
      speed: 'fast' as const,
    };

    const controller = GameController.resumeOrStart(2048, {
      storage: { load: () => stored, save: () => undefined },
    });

    expect(controller.snapshot()).toMatchObject({
      control: 'manual',
      game: { score: 28, moveCount: 9 },
      bestScore: 400,
      speed: 'fast',
    });
  });

  it('starts a fresh 对局 without clearing the highest score', () => {
    const controller = GameController.restore({
      board: [
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      score: 80,
      moveCount: 12,
      phase: 'playing',
      randomState: [4, 3, 2, 1],
      hasShown2048: false,
    });

    controller.restart(99);

    expect(controller.snapshot()).toMatchObject({
      control: 'manual',
      game: { score: 0, moveCount: 0, phase: 'playing' },
      bestScore: 80,
    });
  });

  it('pauses safely and preserves the board when model inference fails', async () => {
    const policy = new PolicyModel({
      name: 'broken',
      infer() {
        return Promise.reject(new Error('推理后端不可用'));
      },
    });
    const controller = GameController.start(2048, { policy });
    const before = controller.snapshot().game;

    controller.startAutoPlay();
    const transition = await controller.stepAutoPlay();

    expect(transition).toBeUndefined();
    expect(controller.snapshot()).toMatchObject({
      control: 'manual',
      aiStatus: 'error',
      error: '推理后端不可用',
      game: before,
    });
  });

  it('discards an in-flight decision after pause or restart', async () => {
    let resolveInference: ((logits: Float32Array) => void) | undefined;
    const policy = new PolicyModel({
      name: 'deferred',
      infer: () =>
        new Promise((resolve) => {
          resolveInference = resolve;
        }),
    });
    const controller = GameController.start(2048, { policy });
    const original = controller.snapshot().game;

    controller.startAutoPlay();
    const pending = controller.stepAutoPlay();
    controller.pauseAutoPlay();
    resolveInference?.(new Float32Array([1, 2, 3, 4]));

    expect(await pending).toBeUndefined();
    expect(controller.snapshot().game).toEqual(original);

    controller.startAutoPlay();
    const pendingBeforeRestart = controller.stepAutoPlay();
    controller.restart(99);
    resolveInference?.(new Float32Array([1, 2, 3, 4]));

    expect(await pendingBeforeRestart).toBeUndefined();
    expect(controller.snapshot().game.moveCount).toBe(0);
  });

  it('shares one in-flight inference between concurrent step requests', async () => {
    let calls = 0;
    let resolveInference: ((logits: Float32Array) => void) | undefined;
    const policy = new PolicyModel({
      name: 'single-flight',
      infer: () => {
        calls += 1;
        return new Promise((resolve) => {
          resolveInference = resolve;
        });
      },
    });
    const controller = GameController.start(2048, { policy });
    controller.startAutoPlay();

    const first = controller.stepAutoPlay();
    const second = controller.stepAutoPlay();
    resolveInference?.(new Float32Array([1, 2, 3, 4]));
    await Promise.all([first, second]);

    expect(calls).toBe(1);
    expect(controller.snapshot().game.moveCount).toBe(1);
  });
});
