import type { PolicyModel } from '../ai/policy-model';
import {
  GameSession,
  type Direction,
  type GameSnapshot,
  type MoveTransition,
} from './game-session';

export type AutoPlaySpeed = 'slow' | 'standard' | 'fast';
export type AiStatus = 'unavailable' | 'ready' | 'running' | 'paused' | 'error';

export interface ControllerSnapshot {
  readonly game: GameSnapshot;
  readonly bestScore: number;
  readonly control: 'manual' | 'ai';
  readonly aiStatus: AiStatus;
  readonly speed: AutoPlaySpeed;
  readonly lastDirection?: Direction;
  readonly backend?: string;
  readonly error?: string;
}

export interface StoredControllerState {
  readonly game: GameSnapshot;
  readonly bestScore: number;
  readonly speed: AutoPlaySpeed;
}

export interface GameStorage {
  load(): StoredControllerState | undefined;
  save(snapshot: StoredControllerState): void;
}

export interface GameControllerOptions {
  readonly policy?: PolicyModel;
  readonly storage?: GameStorage;
}

export class GameController {
  static start(seed: number, options: GameControllerOptions = {}): GameController {
    const controller = new GameController(
      GameSession.start({ seed }),
      options.policy,
      options.storage,
    );
    controller.persist();
    return controller;
  }

  static resumeOrStart(seed: number, options: GameControllerOptions = {}): GameController {
    const stored = options.storage?.load();
    if (!stored) return GameController.start(seed, options);
    return new GameController(
      GameSession.restore(stored.game),
      options.policy,
      options.storage,
      stored.bestScore,
      stored.speed,
    );
  }

  static restore(snapshot: GameSnapshot, options: GameControllerOptions = {}): GameController {
    return new GameController(GameSession.restore(snapshot), options.policy, options.storage);
  }

  private control: ControllerSnapshot['control'] = 'manual';
  private aiStatus: AiStatus;
  private speed: AutoPlaySpeed = 'standard';
  private lastDirection: Direction | undefined;
  private backend: string | undefined;
  private error: string | undefined;
  private bestScore: number;
  private controlRevision = 0;
  private inFlightStep: Promise<MoveTransition | undefined> | undefined;

  private constructor(
    private session: GameSession,
    private policy: PolicyModel | undefined,
    private readonly storage: GameStorage | undefined,
    initialBestScore?: number,
    initialSpeed?: AutoPlaySpeed,
  ) {
    this.aiStatus = policy ? 'ready' : 'unavailable';
    this.bestScore = Math.max(initialBestScore ?? 0, session.snapshot().score);
    if (initialSpeed) this.speed = initialSpeed;
  }

  setPolicy(policy: PolicyModel): void {
    this.controlRevision += 1;
    this.policy = policy;
    this.aiStatus = 'ready';
    this.error = undefined;
  }

  startAutoPlay(): void {
    if (!this.policy || this.aiStatus === 'error') return;
    this.controlRevision += 1;
    this.control = 'ai';
    this.aiStatus = 'running';
    this.error = undefined;
  }

  pauseAutoPlay(): void {
    this.controlRevision += 1;
    this.control = 'manual';
    if (this.policy) this.aiStatus = 'paused';
  }

  setSpeed(speed: AutoPlaySpeed): void {
    this.speed = speed;
    this.persist();
  }

  restart(seed: number): void {
    this.controlRevision += 1;
    this.bestScore = Math.max(this.bestScore, this.session.snapshot().score);
    this.session = GameSession.start({ seed });
    this.control = 'manual';
    this.aiStatus = this.policy ? 'ready' : 'unavailable';
    this.lastDirection = undefined;
    this.backend = undefined;
    this.error = undefined;
    this.persist();
  }

  move(direction: Direction): MoveTransition {
    if (this.control === 'ai')
      return { moved: false, events: [], snapshot: this.session.snapshot() };
    const transition = this.session.move(direction);
    if (transition.moved) this.persist();
    return transition;
  }

  stepAutoPlay(): Promise<MoveTransition | undefined> {
    if (this.inFlightStep) return this.inFlightStep;
    if (this.control !== 'ai' || !this.policy || this.session.snapshot().phase === 'over') {
      return Promise.resolve(undefined);
    }
    const revision = this.controlRevision;
    const session = this.session;
    const pending = this.performAutoPlayStep(revision, session).finally(() => {
      if (this.inFlightStep === pending) this.inFlightStep = undefined;
    });
    this.inFlightStep = pending;
    return pending;
  }

  private async performAutoPlayStep(
    revision: number,
    session: GameSession,
  ): Promise<MoveTransition | undefined> {
    try {
      const policy = this.policy;
      if (!policy) return undefined;
      const decision = await policy.decide(session.snapshot().board);
      if (revision !== this.controlRevision || session !== this.session || this.control !== 'ai') {
        return undefined;
      }
      this.lastDirection = decision.direction;
      this.backend = decision.backend;
      const transition = session.move(decision.direction);
      if (transition.moved) this.persist();
      if (transition.snapshot.phase === 'over') this.pauseAutoPlay();
      return transition;
    } catch (error) {
      if (revision !== this.controlRevision || session !== this.session) return undefined;
      this.control = 'manual';
      this.aiStatus = 'error';
      this.error = error instanceof Error ? error.message : '策略模型无法完成本次决策';
      return undefined;
    }
  }

  snapshot(): ControllerSnapshot {
    const optional = {
      ...(this.lastDirection ? { lastDirection: this.lastDirection } : {}),
      ...(this.backend ? { backend: this.backend } : {}),
      ...(this.error ? { error: this.error } : {}),
    };
    return {
      game: this.session.snapshot(),
      bestScore: this.bestScore,
      control: this.control,
      aiStatus: this.aiStatus,
      speed: this.speed,
      ...optional,
    };
  }

  private persist(): void {
    const game = this.session.snapshot();
    this.bestScore = Math.max(this.bestScore, game.score);
    this.storage?.save({ game, bestScore: this.bestScore, speed: this.speed });
  }
}
