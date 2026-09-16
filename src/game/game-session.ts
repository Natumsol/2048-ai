import { Xoshiro128StarStar, type RandomState } from './prng';

export type TileValue = number;
export type Board = readonly (readonly TileValue[])[];
export type Direction = 'up' | 'right' | 'down' | 'left';
export type Position = readonly [row: number, column: number];

export interface GameSnapshot {
  readonly board: Board;
  readonly score: number;
  readonly moveCount: number;
  readonly phase: 'playing' | 'over';
  readonly randomState: RandomState;
  readonly hasShown2048: boolean;
}

export type GameEvent =
  | {
      readonly type: 'move';
      readonly from: Position;
      readonly to: Position;
      readonly value: number;
    }
  | {
      readonly type: 'merge';
      readonly from: readonly [Position, Position];
      readonly to: Position;
      readonly value: number;
    }
  | { readonly type: 'spawn'; readonly at: Position; readonly value: 2 | 4 }
  | { readonly type: 'win' }
  | { readonly type: 'game-over' };

export interface MoveTransition {
  readonly moved: boolean;
  readonly events: readonly GameEvent[];
  readonly snapshot: GameSnapshot;
}

const BOARD_SIZE = 4;

const emptyBoard = (): number[][] =>
  Array.from({ length: BOARD_SIZE }, () => Array<number>(BOARD_SIZE).fill(0));

const cloneBoard = (board: Board): number[][] => board.map((row) => [...row]);

const spawnTile = (
  board: number[][],
  random: Xoshiro128StarStar,
): Extract<GameEvent, { type: 'spawn' }> | undefined => {
  const emptyCells: Position[] = [];
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let column = 0; column < BOARD_SIZE; column += 1) {
      if (board[row]?.[column] === 0) emptyCells.push([row, column]);
    }
  }

  const selected = emptyCells[Math.floor(random.nextFloat() * emptyCells.length)];
  if (!selected) return undefined;
  const [row, column] = selected;
  const value = random.nextFloat() < 0.9 ? 2 : 4;
  const targetRow = board[row];
  if (!targetRow) return undefined;
  targetRow[column] = value;
  return { type: 'spawn', at: selected, value };
};

const positionsFor = (direction: Direction, line: number): Position[] => {
  const positions: Position[] = [];
  for (let offset = 0; offset < BOARD_SIZE; offset += 1) {
    switch (direction) {
      case 'left':
        positions.push([line, offset]);
        break;
      case 'right':
        positions.push([line, BOARD_SIZE - 1 - offset]);
        break;
      case 'up':
        positions.push([offset, line]);
        break;
      case 'down':
        positions.push([BOARD_SIZE - 1 - offset, line]);
        break;
    }
  }
  return positions;
};

const tileAt = (board: Board, [row, column]: Position): number => board[row]?.[column] ?? 0;

const setTile = (board: number[][], [row, column]: Position, value: number): void => {
  const targetRow = board[row];
  if (targetRow) targetRow[column] = value;
};

const boardsEqual = (first: Board, second: Board): boolean =>
  first.every((row, rowIndex) =>
    row.every((value, columnIndex) => value === second[rowIndex]?.[columnIndex]),
  );

export const legalDirections = (board: Board): Direction[] => {
  const directions: Direction[] = ['up', 'right', 'down', 'left'];
  return directions.filter((direction) => {
    for (let line = 0; line < BOARD_SIZE; line += 1) {
      const values = positionsFor(direction, line).map((position) => tileAt(board, position));
      const compact = values.filter((value) => value !== 0);
      const shifted = [...compact, ...Array<number>(BOARD_SIZE - compact.length).fill(0)];
      for (let index = 0; index < shifted.length - 1; index += 1) {
        if (shifted[index] !== 0 && shifted[index] === shifted[index + 1]) return true;
      }
      if (values.some((value, index) => value !== shifted[index])) return true;
    }
    return false;
  });
};

export class GameSession {
  static start({ seed }: { readonly seed: number }): GameSession {
    const random = Xoshiro128StarStar.fromSeed(seed);
    const board = emptyBoard();
    spawnTile(board, random);
    spawnTile(board, random);
    return new GameSession(board, random, 0, 0, 'playing', false);
  }

  static restore(snapshot: GameSnapshot): GameSession {
    return new GameSession(
      cloneBoard(snapshot.board),
      new Xoshiro128StarStar(snapshot.randomState),
      snapshot.score,
      snapshot.moveCount,
      snapshot.phase,
      snapshot.hasShown2048,
    );
  }

  private constructor(
    private board: number[][],
    private readonly random: Xoshiro128StarStar,
    private score: number,
    private moveCount: number,
    private phase: GameSnapshot['phase'],
    private hasShown2048: boolean,
  ) {}

  move(direction: Direction): MoveTransition {
    if (this.phase === 'over') return { moved: false, events: [], snapshot: this.snapshot() };

    const nextBoard = emptyBoard();
    const events: GameEvent[] = [];
    let earnedScore = 0;

    for (let line = 0; line < BOARD_SIZE; line += 1) {
      const positions = positionsFor(direction, line);
      const tiles = positions
        .map((position) => ({ position, value: tileAt(this.board, position) }))
        .filter(({ value }) => value !== 0);

      let sourceIndex = 0;
      let targetIndex = 0;
      while (sourceIndex < tiles.length) {
        const current = tiles[sourceIndex];
        const following = tiles[sourceIndex + 1];
        const destination = positions[targetIndex];
        if (!current || !destination) break;

        if (following?.value === current.value) {
          const mergedValue = current.value * 2;
          setTile(nextBoard, destination, mergedValue);
          events.push({
            type: 'merge',
            from: [current.position, following.position],
            to: destination,
            value: mergedValue,
          });
          earnedScore += mergedValue;
          sourceIndex += 2;
        } else {
          setTile(nextBoard, destination, current.value);
          if (current.position[0] !== destination[0] || current.position[1] !== destination[1]) {
            events.push({
              type: 'move',
              from: current.position,
              to: destination,
              value: current.value,
            });
          }
          sourceIndex += 1;
        }
        targetIndex += 1;
      }
    }

    if (boardsEqual(this.board, nextBoard)) {
      return { moved: false, events: [], snapshot: this.snapshot() };
    }

    this.board = nextBoard;
    this.score += earnedScore;
    this.moveCount += 1;
    const spawn = spawnTile(this.board, this.random);
    if (spawn) events.push(spawn);

    if (!this.hasShown2048 && this.board.some((row) => row.some((value) => value >= 2048))) {
      this.hasShown2048 = true;
      events.push({ type: 'win' });
    }

    if (legalDirections(this.board).length === 0) {
      this.phase = 'over';
      events.push({ type: 'game-over' });
    }

    return { moved: true, events, snapshot: this.snapshot() };
  }

  snapshot(): GameSnapshot {
    return {
      board: cloneBoard(this.board),
      score: this.score,
      moveCount: this.moveCount,
      phase: this.phase,
      randomState: this.random.snapshot(),
      hasShown2048: this.hasShown2048,
    };
  }
}
