# 2048 AI

一个完全在浏览器本地运行的 2048：PixiJS 负责棋盘与动画，ONNX Runtime Web 执行由 PyTorch 训练的策略网络。人工操作、自动游玩、当前对局存档和最高分都不依赖后端，也不上传数据。

## 本地运行

需要 Node.js 22、npm 10、Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
npm install
npm run dev
```

方向键、WASD 和棋盘滑动都可以移动。模型会在页面可玩后异步加载；WebGPU 不可用时自动回退到单线程 WASM。

## 验证

```bash
npm run check
npm run check:python
npm run build
npm run test:e2e
```

`npm run check:all` 会依次执行以上全部检查。规则测试包含固定种子重放、合并语义、属性测试、控制器状态和模型异常降级；Python 测试覆盖教师、数据、网络与 ONNX 导出契约。

## 训练与发布模型

快速端到端验证：

```bash
uv sync --directory training --all-groups
npm run train:smoke
```

正式预算训练：

```bash
npm run train:full
npm run train:dagger
npm run train:polish
npm run train:ensemble
```

流水线依次生成 Expectimax 软目标、按完整种子拆分数据、训练残差 CNN，再通过 DAgger 收集学生实际访问的困难局面并低学习率微调。发布模型在 ONNX 图内集成棋盘的八种旋转与镜像视角，输入输出契约仍保持固定形状。浏览器在创建会话前会验证清单契约和模型 SHA-256。

正式质量基准直接使用 TypeScript 权威规则引擎和 ONNX Runtime Web WASM：

```bash
npm run benchmark:model -- --games 100
```

`full` 流水线先验证教师门槛，再在候选目录中训练与导出；只有 TypeScript + ONNX Runtime Web 的 1,000 局基准达到「无非法移动、至少 50% 到达 2048、中位最大方块至少 1024、单路决策 P95 不超过 50ms」才会原子发布内容哈希命名的模型。当前正式模型达到 2048 的比例为 73.7%，中位最大方块为 2048，决策 P95 为 19.45ms，且没有非法动作或截断局。

模型输入固定为 `float32[1,16,4,4]`，输出为 `[上, 右, 下, 左]` 顺序的 `float32[1,4]` logits。浏览器始终先屏蔽非法方向，再选择最高 logit；模型异常会暂停 AI，但不会影响人工对局。
