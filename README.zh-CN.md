# 2048 AI

[English](README.md) | 简体中文

## 演示

![桌面端 AI 自动游玩](docs/images/2048-desktop.png)

<details>
<summary>移动端视图</summary>

<img src="docs/images/2048-mobile.png" alt="2048 AI 移动端页面" width="390" />

</details>

截图来自实际运行的中文界面，展示 AI 自动游玩后的局面。UI 默认英文，通过 English / 中文 按钮切换，语言选择会保存在本地。

一个完全在浏览器本地运行的 2048：PixiJS 负责棋盘与动画，ONNX Runtime Web 执行由 PyTorch 训练的策略网络。人工操作、自动游玩、当前对局存档和最高分都不依赖后端，也不上传数据。

## 本地运行

试玩需要 Node.js 22 和 npm 10。仓库已包含训练好的模型，不需要 Python 或重新训练。训练和 Python 验证另需 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
npm ci
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

## 使用说明与功能

打开开发服务器显示的地址，通常为 `http://localhost:5173/`。点击「开始自动游玩」交给 AI，点击「暂停自动游玩」恢复人工操作。支持三档速度、本地存档、最高分、响应式布局和减少动态效果。达到 2048 后可以继续对局。

## 基准数据

| 指标                | 结果                 |
| ------------------- | -------------------- |
| 达到 2048           | 737 / 1,000（73.7%） |
| 最大方块中位数      | 2048                 |
| 分数中位数          | 32,438               |
| 移动步数中位数      | 1,682                |
| 决策延迟 P95        | 19.45 ms             |
| 非法移动 / 截断对局 | 0 / 0                |
| 模型大小            | 约 4.6 MiB           |

对局种子为 50,000–50,999。延迟另用五个单进程对局测量，避免并行 CPU 争用。耗时取决于机器。详见[模型清单](public/models/policy.manifest.json)。

## 训练注意事项

完整数据生成需要 C++ 编译器，以编译原生 Expectimax 教师；macOS 可安装 Xcode Command Line Tools。`train:smoke` 会发布小规模测试模型，可能替换自带模型清单，且不受正式门禁约束。实验后可恢复 Git 跟踪的清单以使用自带模型。

正式训练阶段即使发布门禁失败，也会保留候选检查点。命令以错误退出后，下一阶段仍可使用已保存的候选模型。发布门禁还要求无截断对局。

[完整配置](training/configs/full.json)使用 96 通道、六个残差块和两轮 DAgger。DAgger 由教师标注学生访问的局面，通过加权困难局面回放训练，再低学习率微调。最终 ONNX 图平均八种旋转与镜像视角。数据和中间产物位于 `training/datasets/` 与 `training/artifacts/`，不进入 Git。

## 代码结构

| 目录             | 职责                             |
| ---------------- | -------------------------------- |
| `src/game/`      | 权威规则、确定性随机数与对局控制 |
| `src/ai/`        | 棋盘编码、合法移动选择与推理     |
| `src/render/`    | PixiJS 渲染与动画                |
| `training/`      | 教师、数据、训练与导出           |
| `public/models/` | 模型与清单                       |
| `tests/e2e/`     | 桌面与移动端测试                 |
| `docs/adr/`      | 架构决策                         |

## 生产构建与验证准备

```bash
npm run build
npm run preview
```

完整验证前安装依赖和测试浏览器：

```bash
uv sync --directory training --all-groups
npx playwright install chromium
npm run check:all
```
