import { legalDirections, type Board, type Direction } from '../game/game-session';

export interface PolicyBackend {
  readonly name: string;
  infer(input: Float32Array): Promise<Float32Array>;
}

export interface PolicyDecision {
  readonly direction: Direction;
  readonly logits: readonly number[];
  readonly backend: string;
  readonly durationMs: number;
}

export class PolicyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PolicyError';
  }
}

const DIRECTION_ORDER: readonly Direction[] = ['up', 'right', 'down', 'left'];
const BOARD_CELLS = 16;
const CHANNELS = 16;

const encodeBoard = (board: Board): Float32Array => {
  const encoded = new Float32Array(CHANNELS * BOARD_CELLS);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      const value = board[row]?.[column] ?? 0;
      const channel = value === 0 ? 0 : Math.min(15, Math.log2(value));
      if (!Number.isInteger(channel)) throw new PolicyError(`棋盘包含非 2 的幂：${value}`);
      encoded[channel * BOARD_CELLS + row * 4 + column] = 1;
    }
  }
  return encoded;
};

export class PolicyModel {
  constructor(private readonly backend: PolicyBackend) {}

  async decide(board: Board): Promise<PolicyDecision> {
    const legal = new Set(legalDirections(board));
    if (legal.size === 0) throw new PolicyError('当前局面没有合法移动');

    const startedAt = performance.now();
    const output = await this.backend.infer(encodeBoard(board));
    const durationMs = performance.now() - startedAt;
    if (output.length !== DIRECTION_ORDER.length) {
      throw new PolicyError(`模型输出长度应为 4，实际为 ${output.length}`);
    }
    if ([...output].some((value) => !Number.isFinite(value))) {
      throw new PolicyError('模型输出包含非有限值');
    }

    let direction: Direction | undefined;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const [index, candidate] of DIRECTION_ORDER.entries()) {
      const score = output[index];
      if (legal.has(candidate) && score !== undefined && score > bestScore) {
        direction = candidate;
        bestScore = score;
      }
    }
    if (!direction) throw new PolicyError('模型没有返回可执行方向');

    return {
      direction,
      logits: [...output],
      backend: this.backend.name,
      durationMs,
    };
  }
}
