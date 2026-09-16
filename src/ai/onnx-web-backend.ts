import * as ort from 'onnxruntime-web/webgpu';

import type { PolicyBackend } from './policy-model';

interface ModelManifest {
  readonly contractVersion: number;
  readonly model: string;
  readonly sha256: string;
  readonly input: { readonly name: string; readonly shape: readonly number[] };
  readonly output: { readonly name: string; readonly shape: readonly number[] };
}

const arraysEqual = (
  first: readonly (number | string)[],
  second: readonly (number | string)[],
): boolean =>
  first.length === second.length && first.every((value, index) => value === second[index]);

const toHex = (buffer: ArrayBuffer): string =>
  [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, '0')).join('');

const loadManifest = async (url: string): Promise<ModelManifest> => {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`无法读取模型清单（${response.status}）`);
  const manifest = (await response.json()) as ModelManifest;
  if (
    manifest.contractVersion !== 1 ||
    manifest.input.name !== 'board' ||
    !arraysEqual(manifest.input.shape, [1, 16, 4, 4]) ||
    manifest.output.name !== 'logits' ||
    !arraysEqual(manifest.output.shape, [1, 4]) ||
    !/^[a-f\d]{64}$/u.test(manifest.sha256)
  ) {
    throw new Error('模型清单与 contractVersion 1 不兼容');
  }
  return manifest;
};

const resolveModelUrl = (manifestUrl: string, model: string): string =>
  new URL(model, new URL(manifestUrl, window.location.href)).toString();

const validateSessionContract = (session: ort.InferenceSession): void => {
  const input = session.inputMetadata[0];
  const output = session.outputMetadata[0];
  if (
    session.inputNames.length !== 1 ||
    session.inputNames[0] !== 'board' ||
    !input?.isTensor ||
    input.type !== 'float32' ||
    !arraysEqual(input.shape, [1, 16, 4, 4]) ||
    session.outputNames.length !== 1 ||
    session.outputNames[0] !== 'logits' ||
    !output?.isTensor ||
    output.type !== 'float32' ||
    !arraysEqual(output.shape, [1, 4])
  ) {
    throw new Error('ONNX 模型的实际输入输出与 contractVersion 1 不兼容');
  }
};

const createValidatedSession = async (
  modelBytes: Uint8Array,
  executionProvider: 'webgpu' | 'wasm',
): Promise<ort.InferenceSession> => {
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [executionProvider],
    graphOptimizationLevel: 'all',
  });
  try {
    validateSessionContract(session);
    return session;
  } catch (error) {
    await session.release();
    throw error;
  }
};

export class OnnxWebBackend implements PolicyBackend {
  static async load(manifestUrl: string): Promise<OnnxWebBackend> {
    const manifest = await loadManifest(manifestUrl);
    const response = await fetch(resolveModelUrl(manifestUrl, manifest.model));
    if (!response.ok) throw new Error(`无法下载策略模型（${response.status}）`);
    const bytes = await response.arrayBuffer();
    const actualHash = toHex(await crypto.subtle.digest('SHA-256', bytes));
    if (actualHash !== manifest.sha256) throw new Error('策略模型哈希校验失败');

    const modelBytes = new Uint8Array(bytes);
    if ('gpu' in navigator) {
      try {
        const session = await createValidatedSession(modelBytes, 'webgpu');
        return new OnnxWebBackend(session, 'WebGPU');
      } catch {
        // Fall through to the compatibility baseline.
      }
    }

    ort.env.wasm.numThreads = 1;
    const session = await createValidatedSession(modelBytes, 'wasm');
    return new OnnxWebBackend(session, 'WASM');
  }

  private constructor(
    private readonly session: ort.InferenceSession,
    readonly name: string,
  ) {}

  async infer(input: Float32Array): Promise<Float32Array> {
    const output = await this.session.run({
      board: new ort.Tensor('float32', input, [1, 16, 4, 4]),
    });
    const logits = output.logits;
    if (!logits) throw new Error('模型没有返回 logits');
    if (!(logits.data instanceof Float32Array)) throw new Error('模型 logits 不是 float32');
    return new Float32Array(logits.data);
  }
}
