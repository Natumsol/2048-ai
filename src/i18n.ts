export type Language = 'en' | 'zh-CN';
export const languageKey = '2048-ai:language';

const english: Record<string, string> = {
  跳到棋盘: 'Skip to board',
  本地策略模型: 'LOCAL POLICY MODEL',
  '每一步都在你的浏览器里决定。你来走，或把当前局面交给神经网络。':
    'Every move stays in your browser. Play yourself, or let the neural network take over.',
  对局分数: 'Game scores',
  分数: 'Score',
  最高分: 'Best',
  '纯本地运行，不收集对局。方向键、WASD 或滑动都可以移动。':
    'Runs locally. No gameplay data collected. Move with arrow keys, WASD, or swipes.',
  正在准备模型: 'Preparing model',
  重试: 'Retry',
  人工模式: 'Manual play',
  '2048 棋盘': '2048 board',
  对局结束: 'Game over',
  '已到达 2048，继续向前': '2048 reached. Keep going!',
  开始自动游玩: 'Start AI autoplay',
  暂停自动游玩: 'Pause AI autoplay',
  重新开始: 'Restart',
  自动游玩速度: 'Autoplay speed',
  慢: 'Slow',
  标准: 'Normal',
  快: 'Fast',
  减少动态: 'Reduce motion',
  '重新开始当前对局？': 'Restart this game?',
  '棋盘与当前分数会被清除，最高分会保留。':
    'The board and score will reset. Your best score will be kept.',
  继续当前对局: 'Keep playing',
  正在加载本地策略模型: 'Loading local policy model',
  人工模式可用: 'Manual play available',
  策略模型已暂停: 'Policy model paused',
  'AI 已暂停，可以人工操作': 'AI paused. Your turn.',
  策略模型已就绪: 'Policy model ready',
  等待首次推理: 'Awaiting first inference',
  无法加载本地策略模型: 'Could not load the local policy model',
  '模型推理失败，可以人工操作': 'AI inference failed. Manual play available.',
  'AI 正在游玩': 'AI playing',
};

export const translate = (text: string, language: Language): string =>
  language === 'en' ? (english[text] ?? text) : text;

export const readLanguage = (): Language => {
  try {
    return localStorage.getItem(languageKey) === 'zh-CN' ? 'zh-CN' : 'en';
  } catch {
    return 'en';
  }
};

/** Remember original copy so switching languages never changes game state. */
export const bindStaticTranslations = (root: HTMLElement): ((language: Language) => void) => {
  const nodes: { node: Text; original: string }[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    if (english[node.textContent?.trim() ?? '']) {
      nodes.push({ node: node as Text, original: node.textContent?.trim() ?? '' });
    }
    node = walker.nextNode();
  }
  const labels = [...root.querySelectorAll<HTMLElement>('[aria-label]')].map((element) => ({
    element,
    original: element.getAttribute('aria-label') ?? '',
  }));
  return (language) => {
    for (const { node, original } of nodes) node.textContent = translate(original, language);
    for (const { element, original } of labels) {
      element.setAttribute('aria-label', translate(original, language));
    }
  };
};
