import { OnnxWebBackend } from './ai/onnx-web-backend';
import { PolicyModel } from './ai/policy-model';
import styles from './app.module.css';
import { BrowserGameStorage } from './game/browser-storage';
import { GameController, type AutoPlaySpeed } from './game/game-controller';
import type { Direction, GameEvent, MoveTransition } from './game/game-session';
import { BoardRenderer, boardAsText } from './render/board-renderer';

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('缺少应用挂载节点');

root.innerHTML = `
  <a class="${styles.skipLink}" href="#game-board">跳到棋盘</a>
  <div class="${styles.app}">
    <div class="${styles.shell}">
      <header class="${styles.identity}">
        <p class="${styles.eyebrow}">本地策略模型</p>
        <h1 class="${styles.title}">2048<small>AI PLAY</small></h1>
        <p class="${styles.tagline}">每一步都在你的浏览器里决定。你来走，或把当前局面交给神经网络。</p>
        <div class="${styles.scores}" aria-label="对局分数">
          <div class="${styles.score}"><span class="${styles.scoreLabel}">分数</span><strong class="${styles.scoreValue}" id="score">0</strong></div>
          <div class="${styles.score}"><span class="${styles.scoreLabel}">最高分</span><strong class="${styles.scoreValue}" id="best-score">0</strong></div>
        </div>
        <p class="${styles.privacyNote}">纯本地运行，不收集对局。方向键、WASD 或滑动都可以移动。</p>
      </header>

      <main class="${styles.gameRegion}" id="main-content">
        <div class="${styles.boardHeader}">
          <div class="${styles.status}" aria-live="polite">
            <span class="${styles.statusDot}" id="status-dot"></span>
            <span id="status-copy">正在准备模型</span>
            <button class="${styles.retry}" id="retry-model" type="button" hidden>重试</button>
          </div>
          <span class="${styles.backend}" id="backend">人工模式</span>
        </div>

        <section class="${styles.boardShell}" id="game-board" aria-label="2048 棋盘" tabindex="0">
          <div class="${styles.boardCanvas}" id="board-canvas"></div>
          <span class="${styles.directionSignal}" id="direction-signal" aria-hidden="true"></span>
          <div class="${styles.messageOverlay}" id="game-over" data-visible="false">
            <div><strong>对局结束</strong><span id="game-over-copy"></span></div>
          </div>
          <div class="${styles.winToast}" id="win-toast" data-visible="false">已到达 2048，继续向前</div>
        </section>

        <p class="${styles.srOnly}" id="board-description" aria-live="polite"></p>

        <div class="${styles.controls}">
          <button class="${styles.primary}" id="auto-play" type="button" disabled>开始自动游玩</button>
          <button class="${styles.secondary}" id="restart" type="button">重新开始</button>
        </div>

        <div class="${styles.settings}">
          <div class="${styles.speedControl}" role="group" aria-label="自动游玩速度">
            <button class="${styles.speedButton}" type="button" data-speed="slow">慢</button>
            <button class="${styles.speedButton}" type="button" data-speed="standard">标准</button>
            <button class="${styles.speedButton}" type="button" data-speed="fast">快</button>
          </div>
          <label class="${styles.motionToggle}"><input id="reduce-motion" type="checkbox" />减少动态</label>
        </div>
      </main>
    </div>
  </div>

  <dialog class="${styles.dialog}" id="restart-dialog">
    <div class="${styles.dialogBody}">
      <h2>重新开始当前对局？</h2>
      <p>棋盘与当前分数会被清除，最高分会保留。</p>
      <div class="${styles.dialogActions}">
        <button class="${styles.dialogAction}" id="cancel-restart" type="button">继续当前对局</button>
        <button class="${styles.dialogAction}" data-danger="true" id="confirm-restart" type="button">重新开始</button>
      </div>
    </div>
  </dialog>
`;

const required = <ElementType extends Element>(selector: string): ElementType => {
  const element = document.querySelector<ElementType>(selector);
  if (!element) throw new Error(`缺少界面元素：${selector}`);
  return element;
};

const boardHost = required<HTMLDivElement>('#board-canvas');
const boardShell = required<HTMLElement>('#game-board');
const renderer = await BoardRenderer.create(boardHost);
const storage = new BrowserGameStorage();
const randomSeed = (): number => crypto.getRandomValues(new Uint32Array(1))[0] ?? Date.now();
const controller = GameController.resumeOrStart(randomSeed(), { storage });

const score = required<HTMLElement>('#score');
const bestScore = required<HTMLElement>('#best-score');
const statusDot = required<HTMLElement>('#status-dot');
const statusCopy = required<HTMLElement>('#status-copy');
const backendCopy = required<HTMLElement>('#backend');
const retryModel = required<HTMLButtonElement>('#retry-model');
const autoPlay = required<HTMLButtonElement>('#auto-play');
const restart = required<HTMLButtonElement>('#restart');
const boardDescription = required<HTMLElement>('#board-description');
const directionSignal = required<HTMLElement>('#direction-signal');
const gameOver = required<HTMLElement>('#game-over');
const gameOverCopy = required<HTMLElement>('#game-over-copy');
const winToast = required<HTMLElement>('#win-toast');
const restartDialog = required<HTMLDialogElement>('#restart-dialog');
const cancelRestart = required<HTMLButtonElement>('#cancel-restart');
const confirmRestart = required<HTMLButtonElement>('#confirm-restart');
const reduceMotionInput = required<HTMLInputElement>('#reduce-motion');
const speedButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-speed]')];

const directionGlyph: Record<Direction, string> = {
  up: '↑',
  right: '→',
  down: '↓',
  left: '←',
};
const directionName: Record<Direction, string> = {
  up: '上',
  right: '右',
  down: '下',
  left: '左',
};
const speedInterval: Record<AutoPlaySpeed, number> = { slow: 600, standard: 250, fast: 80 };

let modelLoading = true;
let modelError: string | undefined;
let animationRunning = false;
let bufferedDirection: Direction | undefined;
let autoRunId = 0;
let winTimer: number | undefined;
let resumeAiAfterDialog = false;
const reduceMotionKey = '2048-ai:reduce-motion';
const systemReducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
reduceMotionInput.checked = localStorage.getItem(reduceMotionKey) === 'true' || systemReducedMotion;

const updateDom = (): void => {
  const snapshot = controller.snapshot();
  score.textContent = String(snapshot.game.score);
  bestScore.textContent = String(snapshot.bestScore);
  boardDescription.textContent = `棋盘：${boardAsText(snapshot.game.board)}。分数 ${snapshot.game.score}。`;
  boardShell.dataset.ai = snapshot.aiStatus === 'running' ? 'running' : 'manual';
  directionSignal.textContent = snapshot.lastDirection
    ? directionGlyph[snapshot.lastDirection]
    : '';
  directionSignal.dataset.visible = String(
    snapshot.aiStatus === 'running' && Boolean(snapshot.lastDirection),
  );

  gameOver.dataset.visible = String(snapshot.game.phase === 'over');
  gameOverCopy.textContent = `分数 ${snapshot.game.score}，最大方块 ${Math.max(...snapshot.game.board.flat())}，共 ${snapshot.game.moveCount} 步。`;

  speedButtons.forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.speed === snapshot.speed));
  });

  if (modelLoading) {
    statusDot.dataset.status = 'loading';
    statusCopy.textContent = '正在加载本地策略模型';
    backendCopy.textContent = '人工模式可用';
    retryModel.hidden = true;
    autoPlay.disabled = true;
    return;
  }
  if (modelError) {
    statusDot.dataset.status = 'error';
    statusCopy.textContent = modelError;
    backendCopy.textContent = '人工模式可用';
    retryModel.hidden = false;
    autoPlay.disabled = true;
    return;
  }

  const runtimeModelError = snapshot.aiStatus === 'error';
  retryModel.hidden = !runtimeModelError;
  autoPlay.disabled = snapshot.game.phase === 'over' || runtimeModelError;
  statusDot.dataset.status = snapshot.aiStatus;
  statusCopy.textContent =
    snapshot.aiStatus === 'running'
      ? `AI 正在游玩${snapshot.lastDirection ? `，最近向${directionName[snapshot.lastDirection]}` : ''}`
      : snapshot.aiStatus === 'error'
        ? (snapshot.error ?? '策略模型已暂停')
        : snapshot.aiStatus === 'paused'
          ? 'AI 已暂停，可以人工操作'
          : '策略模型已就绪';
  backendCopy.textContent = snapshot.backend ?? '等待首次推理';
  autoPlay.textContent = snapshot.control === 'ai' ? '暂停自动游玩' : '开始自动游玩';
};

const showWin = (): void => {
  winToast.dataset.visible = 'true';
  if (winTimer) window.clearTimeout(winTimer);
  winTimer = window.setTimeout(() => {
    winToast.dataset.visible = 'false';
  }, 2200);
};

const presentTransition = async (transition: MoveTransition | undefined): Promise<void> => {
  updateDom();
  if (!transition?.moved) return;
  if (transition.events.some((event: GameEvent) => event.type === 'win')) showWin();
  animationRunning = true;
  await renderer.render(transition.snapshot, transition.events, reduceMotionInput.checked);
  animationRunning = false;
};

const moveManually = async (direction: Direction): Promise<void> => {
  if (controller.snapshot().control === 'ai' || restartDialog.open) return;
  if (animationRunning) {
    bufferedDirection = direction;
    return;
  }
  await presentTransition(controller.move(direction));
  const pending = bufferedDirection;
  bufferedDirection = undefined;
  if (pending) await moveManually(pending);
};

const wait = async (durationMs: number): Promise<void> =>
  new Promise((resolve) => window.setTimeout(resolve, durationMs));

const runAutoPlay = async (runId: number): Promise<void> => {
  while (runId === autoRunId && controller.snapshot().control === 'ai') {
    const startedAt = performance.now();
    const transition = await controller.stepAutoPlay();
    await presentTransition(transition);
    const snapshot = controller.snapshot();
    if (!transition) {
      if (snapshot.control === 'ai') {
        await wait(16);
        continue;
      }
      break;
    }
    if (snapshot.control !== 'ai' || snapshot.game.phase === 'over') break;
    await wait(Math.max(0, speedInterval[snapshot.speed] - (performance.now() - startedAt)));
  }
  updateDom();
};

const toggleAutoPlay = (): void => {
  if (controller.snapshot().control === 'ai') {
    autoRunId += 1;
    controller.pauseAutoPlay();
    updateDom();
    return;
  }
  controller.startAutoPlay();
  autoRunId += 1;
  updateDom();
  void runAutoPlay(autoRunId);
};

const loadModel = async (): Promise<void> => {
  modelLoading = true;
  modelError = undefined;
  updateDom();
  try {
    const backend = await OnnxWebBackend.load('/models/policy.manifest.json');
    controller.setPolicy(new PolicyModel(backend));
  } catch (error) {
    modelError = error instanceof Error ? error.message : '无法加载本地策略模型';
  } finally {
    modelLoading = false;
    updateDom();
  }
};

const restartGame = async (): Promise<void> => {
  autoRunId += 1;
  controller.restart(randomSeed());
  updateDom();
  await renderer.render(controller.snapshot().game, [], reduceMotionInput.checked);
  boardShell.focus();
};

autoPlay.addEventListener('click', toggleAutoPlay);
retryModel.addEventListener('click', () => void loadModel());
restart.addEventListener('click', () => {
  const snapshot = controller.snapshot();
  if (snapshot.game.moveCount === 0 || snapshot.game.phase === 'over') {
    void restartGame();
    return;
  }
  resumeAiAfterDialog = snapshot.control === 'ai';
  autoRunId += 1;
  controller.pauseAutoPlay();
  updateDom();
  restartDialog.showModal();
});
cancelRestart.addEventListener('click', () => restartDialog.close('cancel'));
confirmRestart.addEventListener('click', () => restartDialog.close('confirm'));
restartDialog.addEventListener('close', () => {
  if (restartDialog.returnValue === 'confirm') {
    void restartGame();
  } else if (resumeAiAfterDialog) {
    resumeAiAfterDialog = false;
    toggleAutoPlay();
  }
});

speedButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const speed = button.dataset.speed as AutoPlaySpeed;
    controller.setSpeed(speed);
    updateDom();
  });
});

reduceMotionInput.addEventListener('change', () => {
  localStorage.setItem(reduceMotionKey, String(reduceMotionInput.checked));
});

const keyDirections: Partial<Record<string, Direction>> = {
  ArrowUp: 'up',
  w: 'up',
  W: 'up',
  ArrowRight: 'right',
  d: 'right',
  D: 'right',
  ArrowDown: 'down',
  s: 'down',
  S: 'down',
  ArrowLeft: 'left',
  a: 'left',
  A: 'left',
};

window.addEventListener('keydown', (event) => {
  const direction = keyDirections[event.key];
  if (!direction || restartDialog.open) return;
  event.preventDefault();
  void moveManually(direction);
});

let touchStart: { readonly x: number; readonly y: number } | undefined;
boardShell.addEventListener(
  'touchstart',
  (event) => {
    const touch = event.changedTouches[0];
    if (touch) touchStart = { x: touch.clientX, y: touch.clientY };
  },
  { passive: true },
);
boardShell.addEventListener(
  'touchmove',
  (event) => {
    const touch = event.changedTouches[0];
    if (!touch || !touchStart) return;
    const deltaX = touch.clientX - touchStart.x;
    const deltaY = touch.clientY - touchStart.y;
    if (Math.max(Math.abs(deltaX), Math.abs(deltaY)) >= 24) event.preventDefault();
  },
  { passive: false },
);
boardShell.addEventListener(
  'touchend',
  (event) => {
    const touch = event.changedTouches[0];
    if (!touch || !touchStart) return;
    const deltaX = touch.clientX - touchStart.x;
    const deltaY = touch.clientY - touchStart.y;
    touchStart = undefined;
    if (Math.max(Math.abs(deltaX), Math.abs(deltaY)) < 24) return;
    event.preventDefault();
    const direction: Direction =
      Math.abs(deltaX) > Math.abs(deltaY)
        ? deltaX > 0
          ? 'right'
          : 'left'
        : deltaY > 0
          ? 'down'
          : 'up';
    void moveManually(direction);
  },
  { passive: false },
);

await renderer.render(controller.snapshot().game, [], reduceMotionInput.checked);
updateDom();
void loadModel();
