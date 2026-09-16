import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    if (sessionStorage.getItem('e2e-seeded') === 'true') return;
    sessionStorage.setItem('e2e-seeded', 'true');
    localStorage.setItem(
      '2048-ai:session:v1',
      JSON.stringify({
        game: {
          board: [
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
          ],
          score: 0,
          moveCount: 0,
          phase: 'playing',
          randomState: [1, 2, 3, 4],
          hasShown2048: false,
        },
        bestScore: 0,
        speed: 'standard',
      }),
    );
  });
  await page.goto('/');
  await expect(page.getByText('策略模型已就绪')).toBeVisible({ timeout: 20_000 });
});

test('supports WASD and restores the exact persisted session after reload', async ({ page }) => {
  const boardDescription = page.locator('#board-description');
  const initial = await boardDescription.textContent();

  await page.keyboard.press('d');
  await expect(boardDescription).not.toHaveText(initial ?? '');
  const moved = await boardDescription.textContent();
  await page.reload();

  await expect(boardDescription).toHaveText(moved ?? '');
});

test('supports keyboard play, restart confirmation, and AI handoff', async ({ page }) => {
  const boardDescription = page.locator('#board-description');
  const initial = await boardDescription.textContent();
  await page.keyboard.press('ArrowLeft');
  await expect(boardDescription).not.toHaveText(initial ?? '');

  await page.getByRole('button', { name: '重新开始' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('button', { name: '继续当前对局' }).click();

  const beforeAi = await boardDescription.textContent();
  await page.getByRole('button', { name: '开始自动游玩' }).click();
  await expect(page.getByRole('button', { name: '暂停自动游玩' })).toBeVisible();
  await expect(page.getByText(/AI 正在游玩/u)).toBeVisible();
  await expect(boardDescription).not.toHaveText(beforeAi ?? '', { timeout: 10_000 });
  await page.getByRole('button', { name: '暂停自动游玩' }).click();
  await expect(page.getByText('AI 已暂停，可以人工操作')).toBeVisible();
});

test('applies only an intentional swipe and persists reduced motion', async ({ page }) => {
  const boardDescription = page.locator('#board-description');
  const board = page.locator('#game-board');
  const initial = await boardDescription.textContent();
  const swipe = async (distance: number): Promise<void> => {
    await board.evaluate((element, delta) => {
      const touch = (x: number) =>
        new Touch({
          identifier: 1,
          target: element,
          clientX: x,
          clientY: 80,
          screenX: x,
          screenY: 80,
          pageX: x,
          pageY: 80,
          radiusX: 1,
          radiusY: 1,
          rotationAngle: 0,
          force: 1,
        });
      element.dispatchEvent(
        new TouchEvent('touchstart', { bubbles: true, changedTouches: [touch(20)] }),
      );
      element.dispatchEvent(
        new TouchEvent('touchend', { bubbles: true, changedTouches: [touch(20 + delta)] }),
      );
    }, distance);
  };

  await swipe(10);
  await expect(boardDescription).toHaveText(initial ?? '');
  await swipe(60);
  await expect(boardDescription).not.toHaveText(initial ?? '');

  const reducedMotion = page.getByLabel('减少动态');
  await reducedMotion.check();
  await page.reload();
  await expect(reducedMotion).toBeChecked();
});

test('keeps the complete game surface inside a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 720 });
  await page.reload();
  await expect(page.getByText('策略模型已就绪')).toBeVisible({ timeout: 20_000 });

  const board = page.locator('#game-board');
  const box = await board.boundingBox();
  expect(box).not.toBeNull();
  expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(320);
  await expect(page.getByRole('button', { name: '开始自动游玩' })).toBeVisible();
  await expect(page.getByRole('button', { name: '重新开始' })).toBeVisible();
});

test('keeps manual play available when the model cannot be loaded', async ({ page }) => {
  await page.route('**/models/*.onnx', (route) => route.abort());
  await page.reload();

  await expect(page.locator('#backend')).toHaveText('人工模式可用');
  await expect(page.getByRole('button', { name: '重试' })).toBeVisible();
  await expect(page.getByRole('button', { name: '开始自动游玩' })).toBeDisabled();
  await expect(page.getByRole('button', { name: '重新开始' })).toBeEnabled();
});
