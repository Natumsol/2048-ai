import type { AutoPlaySpeed, GameStorage, StoredControllerState } from './game-controller';
import type { GameSnapshot } from './game-session';

const speeds = new Set<AutoPlaySpeed>(['slow', 'standard', 'fast']);
const isNonNegativeSafeInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
const isTile = (value: unknown): value is number =>
  isNonNegativeSafeInteger(value) &&
  (value === 0 || (value >= 2 && Number.isInteger(Math.log2(value))));

const isGameSnapshot = (value: unknown): value is GameSnapshot => {
  if (!value || typeof value !== 'object') return false;
  const snapshot = value as Record<string, unknown>;
  const board = snapshot.board;
  const randomState = snapshot.randomState;
  return (
    Array.isArray(board) &&
    board.length === 4 &&
    board.every((row) => Array.isArray(row) && row.length === 4 && row.every(isTile)) &&
    isNonNegativeSafeInteger(snapshot.score) &&
    isNonNegativeSafeInteger(snapshot.moveCount) &&
    (snapshot.phase === 'playing' || snapshot.phase === 'over') &&
    Array.isArray(randomState) &&
    randomState.length === 4 &&
    randomState.every(
      (part) =>
        typeof part === 'number' && Number.isInteger(part) && part >= 0 && part <= 0xffff_ffff,
    ) &&
    typeof snapshot.hasShown2048 === 'boolean'
  );
};

export class BrowserGameStorage implements GameStorage {
  constructor(private readonly key = '2048-ai:session:v1') {}

  load(): StoredControllerState | undefined {
    try {
      const raw = localStorage.getItem(this.key);
      if (!raw) return undefined;
      const stored = JSON.parse(raw) as Record<string, unknown>;
      if (
        !isGameSnapshot(stored.game) ||
        !isNonNegativeSafeInteger(stored.bestScore) ||
        stored.bestScore < stored.game.score ||
        typeof stored.speed !== 'string' ||
        !speeds.has(stored.speed as AutoPlaySpeed)
      ) {
        return undefined;
      }
      return {
        game: stored.game,
        bestScore: stored.bestScore,
        speed: stored.speed as AutoPlaySpeed,
      };
    } catch {
      return undefined;
    }
  }

  save(snapshot: StoredControllerState): void {
    try {
      localStorage.setItem(this.key, JSON.stringify(snapshot));
    } catch {
      // Storage can be unavailable in private or quota-limited contexts; the game remains usable.
    }
  }
}
