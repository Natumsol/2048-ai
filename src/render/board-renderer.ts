import { Application, Container, Graphics, Text, TextStyle } from 'pixi.js';

import type { Board, GameEvent, GameSnapshot, Position } from '../game/game-session';

const SIZE = 520;
const PADDING = 18;
const GAP = 12;
const CELL = (SIZE - PADDING * 2 - GAP * 3) / 4;

const TILE_COLORS = new Map<number, { background: number; text: number }>([
  [2, { background: 0xeee4da, text: 0x67574d }],
  [4, { background: 0xead8bd, text: 0x67574d }],
  [8, { background: 0xf2b178, text: 0xfffaf2 }],
  [16, { background: 0xee925f, text: 0xfffaf2 }],
  [32, { background: 0xe97852, text: 0xfffaf2 }],
  [64, { background: 0xdc583d, text: 0xfffaf2 }],
  [128, { background: 0xe4bd5b, text: 0x4c3823 }],
  [256, { background: 0xd9a83e, text: 0x44301f }],
  [512, { background: 0xc9932c, text: 0x3d2a1c }],
  [1024, { background: 0xb97823, text: 0xfffaf2 }],
  [2048, { background: 0x9e5e21, text: 0xfffaf2 }],
]);

const positionKey = ([row, column]: Position): string => `${row}:${column}`;
const coordinate = (index: number): number => PADDING + index * (CELL + GAP);
const tileCoordinate = (index: number): number => coordinate(index) + CELL / 2;

const tileColor = (value: number): { background: number; text: number } =>
  TILE_COLORS.get(value) ?? { background: 0x44352d, text: 0xfff8ec };

const tileFontSize = (value: number): number => {
  const digits = String(value).length;
  if (digits <= 2) return 52;
  if (digits === 3) return 46;
  if (digits === 4) return 38;
  return 30;
};

export class BoardRenderer {
  static async create(host: HTMLElement): Promise<BoardRenderer> {
    const app = new Application();
    await app.init({
      width: SIZE,
      height: SIZE,
      antialias: true,
      autoDensity: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
      backgroundAlpha: 0,
      preference: 'webgl',
    });
    app.canvas.setAttribute('aria-hidden', 'true');
    app.canvas.style.width = '100%';
    app.canvas.style.height = '100%';
    host.append(app.canvas);
    return new BoardRenderer(app);
  }

  private readonly tiles = new Container();

  private constructor(private readonly app: Application) {
    const board = new Graphics().roundRect(0, 0, SIZE, SIZE, 28).fill({ color: 0x9d8875 });
    this.app.stage.addChild(board);
    const slots = new Graphics();
    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        slots
          .roundRect(coordinate(column), coordinate(row), CELL, CELL, 16)
          .fill({ color: 0xc6b6a4 });
      }
    }
    this.app.stage.addChild(slots, this.tiles);
  }

  async render(
    snapshot: GameSnapshot,
    events: readonly GameEvent[] = [],
    reducedMotion = false,
  ): Promise<void> {
    this.tiles.removeChildren().forEach((child) => child.destroy({ children: true }));
    const spawnPositions = new Set(
      events.filter((event) => event.type === 'spawn').map((event) => positionKey(event.at)),
    );
    const mergePositions = new Set(
      events.filter((event) => event.type === 'merge').map((event) => positionKey(event.to)),
    );
    const moveOrigins = new Map(
      events
        .filter((event) => event.type === 'move')
        .map((event) => [positionKey(event.to), event.from] as const),
    );

    const animations: Promise<void>[] = [];
    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        const value = snapshot.board[row]?.[column] ?? 0;
        if (value === 0) continue;
        const destination: Position = [row, column];
        const tile = this.createTile(value);
        const targetX = tileCoordinate(column);
        const targetY = tileCoordinate(row);
        const origin = moveOrigins.get(positionKey(destination));
        tile.position.set(
          origin ? tileCoordinate(origin[1]) : targetX,
          origin ? tileCoordinate(origin[0]) : targetY,
        );
        const isSpawn = spawnPositions.has(positionKey(destination));
        const isMerge = mergePositions.has(positionKey(destination));
        const initialScale = isSpawn ? 0.2 : isMerge ? 0.82 : 1;
        tile.scale.set(initialScale);
        tile.alpha = isSpawn ? 0 : 1;
        this.tiles.addChild(tile);

        if (!reducedMotion && (origin || isSpawn || isMerge)) {
          animations.push(
            this.animate(tile, targetX, targetY, initialScale, 1, isSpawn ? 170 : 145),
          );
        } else {
          tile.position.set(targetX, targetY);
          tile.scale.set(1);
          tile.alpha = 1;
        }
      }
    }
    await Promise.all(animations);
  }

  destroy(): void {
    this.app.destroy({ removeView: true }, { children: true });
  }

  private createTile(value: number): Container {
    const container = new Container();
    const colors = tileColor(value);
    const surface = new Graphics()
      .roundRect(0, 0, CELL, CELL, 16)
      .fill({ color: colors.background });
    const label = new Text({
      text: String(value),
      style: new TextStyle({
        fill: colors.text,
        fontFamily: 'Avenir Next, PingFang SC, sans-serif',
        fontSize: tileFontSize(value),
        fontWeight: '700',
      }),
    });
    label.anchor.set(0.5);
    label.position.set(CELL / 2, CELL / 2 + 2);
    container.pivot.set(CELL / 2);
    container.addChild(surface, label);
    return container;
  }

  private animate(
    tile: Container,
    targetX: number,
    targetY: number,
    fromScale: number,
    toScale: number,
    durationMs: number,
  ): Promise<void> {
    const fromX = tile.x;
    const fromY = tile.y;
    const startedAt = performance.now();
    return new Promise((resolve) => {
      const frame = (now: number): void => {
        const progress = Math.min(1, (now - startedAt) / durationMs);
        const eased = 1 - Math.pow(1 - progress, 4);
        tile.position.set(fromX + (targetX - fromX) * eased, fromY + (targetY - fromY) * eased);
        tile.scale.set(fromScale + (toScale - fromScale) * eased);
        tile.alpha = Math.min(1, progress * 2);
        if (progress < 1) requestAnimationFrame(frame);
        else resolve();
      };
      requestAnimationFrame(frame);
    });
  }
}

export const boardAsText = (board: Board): string =>
  board
    .map((row) => row.map((value) => (value === 0 ? '空' : String(value))).join('，'))
    .join('；');
