import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

import * as ort from 'onnxruntime-web';

import { PolicyModel } from '../src/ai/policy-model';
import { GameSession, legalDirections } from '../src/game/game-session';

const argument = (name: string, fallback: string): string => {
  const index = process.argv.indexOf(name);
  return index >= 0 ? (process.argv[index + 1] ?? fallback) : fallback;
};

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((first, second) => first - second);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle] ?? 0;
  return sorted.length % 2 === 0 ? (upper + (sorted[middle - 1] ?? upper)) / 2 : upper;
};

const requestedModelPath = argument('--model', '');
let modelPath = requestedModelPath;
if (!modelPath) {
  const manifestPath = resolve('public/models/policy.manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as { model?: unknown };
  if (typeof manifest.model !== 'string') throw new Error('模型清单缺少 model 文件名');
  modelPath = resolve(dirname(manifestPath), manifest.model);
}
const games = Number.parseInt(argument('--games', '100'), 10);
const seedStart = Number.parseInt(argument('--seed', '50000'), 10);
if (!Number.isSafeInteger(games) || games <= 0) throw new Error('--games 必须是正整数');

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(await readFile(modelPath), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
const policy = new PolicyModel({
  name: 'ONNX Runtime Web WASM',
  async infer(input) {
    const result = await session.run({
      board: new ort.Tensor('float32', input, [1, 16, 4, 4]),
    });
    const logits = result.logits;
    if (!logits || !(logits.data instanceof Float32Array)) {
      throw new Error('模型没有返回 float32 logits');
    }
    return new Float32Array(logits.data);
  },
});

const maximumTiles: number[] = [];
const scores: number[] = [];
const moveCounts: number[] = [];
let illegalMoves = 0;
let truncatedGames = 0;
const decisionDurations: number[] = [];
for (let offset = 0; offset < games; offset += 1) {
  const game = GameSession.start({ seed: seedStart + offset });
  for (let step = 0; step < 20_000; step += 1) {
    const before = game.snapshot();
    if (legalDirections(before.board).length === 0) break;
    const startedAt = performance.now();
    const decision = await policy.decide(before.board);
    if (decisionDurations.length >= 25) {
      decisionDurations.push(performance.now() - startedAt);
    } else {
      decisionDurations.push(Number.NaN);
    }
    const transition = game.move(decision.direction);
    if (!transition.moved) {
      illegalMoves += 1;
      break;
    }
  }
  const snapshot = game.snapshot();
  if (legalDirections(snapshot.board).length > 0) truncatedGames += 1;
  maximumTiles.push(Math.max(...snapshot.board.flat()));
  scores.push(snapshot.score);
  moveCounts.push(snapshot.moveCount);
}
await session.release();

const reached2048 = maximumTiles.filter((tile) => tile >= 2048).length;
const measuredDurations = decisionDurations.filter(Number.isFinite).sort((a, b) => a - b);
const p95Index = Math.max(0, Math.ceil(measuredDurations.length * 0.95) - 1);
process.stdout.write(
  `${JSON.stringify(
    {
      runner: 'typescript-engine+onnxruntime-web-wasm',
      games,
      seedStart,
      illegalMoves,
      truncatedGames,
      reached2048,
      reached2048Rate: reached2048 / games,
      medianMaximumTile: median(maximumTiles),
      medianScore: median(scores),
      medianMoves: median(moveCounts),
      decisionP95Ms: measuredDurations[p95Index] ?? 0,
    },
    undefined,
    2,
  )}\n`,
);
