/* XLSR CNN 超分 —— 实验性档位
 *
 * 与 EASU / RAVU 的定位差异（M1 实测）：
 *   EASU  1080p→2K   0.94ms   —— 6% 帧预算
 *   XLSR  1080p 全帧 28.3ms   —— 169% 的 60fps 预算，只够 30fps
 *
 * 也就是说它比纯 shader 慢约 30 倍，仅适用于 30fps 内容 + 较强 GPU。
 * 之所以仍然值得做：它是唯一能"生成"细节的路线（28K 参数的学习先验），
 * 而 EASU/RAVU 只能重建已有边缘。
 *
 * 能做到 28ms 的前提是 **全卷积、一次前向、不分块**。同一台机器上
 * ESPCN 因输入尺寸硬编码需分 45 块，单帧 372ms —— 差 13 倍。瓶颈是
 * 分块调度而非算力，这一点由批处理增益仅 1.05× 反证。
 *
 * 本模块把 ORT 的复杂性全部隔离：外部只需 upscale(encoder, srcTex) 。
 */

const ORT_MJS = chrome.runtime.getURL("vendor/ort/ort.webgpu.min.mjs");
const ORT_WASM_DIR = chrome.runtime.getURL("vendor/ort/");
const MODEL_URL = chrome.runtime.getURL("models/xlsr-dynamic.onnx");

/** XLSR 固定 4x 放大（模型结构决定，非可配置） */
export const XLSR_SCALE = 4;

let ortPromise = null;

async function loadOrt() {
  if (!ortPromise) {
    ortPromise = (async () => {
      const ort = await import(ORT_MJS);
      // 必须是绝对 URL —— 相对路径会被拼到模块自身目录下，
      // 变成 .../vendor/ort/vendor/ort/... 这种错误路径
      ort.env.wasm.wasmPaths = ORT_WASM_DIR;
      // 多线程依赖 blob worker，扩展 CSP 下会踩
      // "URL.createObjectURL is not a function"
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = "error";
      return ort;
    })();
    ortPromise.catch(() => { ortPromise = null; });
  }
  return ortPromise;
}

/* 取得 ORT 自己创建的 WebGPU device。
 *
 * 这是整个 CNN 档最关键的约束：ORT 在首次创建 webgpu session 时会自行
 * requestDevice()，它不接受外部传入的 device。而 WebGPU 规定 buffer/纹理
 * 只能用于创建它的那个 device —— 跨 device 使用会报
 *   [Buffer] is associated with [Device], and cannot be used with [Device]
 *
 * 因此不能让 ORT 适配我们的 device，只能反过来：先建一个 session 让 ORT
 * 初始化好 device，然后整条渲染管线都改用 ort.env.webgpu.device。
 *
 * 返回 null 表示 ORT 不可用，调用方应回落纯 shader 路径。
 */
export async function getOrtDevice() {
  try {
    const ort = await loadOrt();
    // 创建一次 session 以触发 ORT 内部的 device 初始化
    const probe = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    const device = ort.env.webgpu?.device ?? null;
    if (!device) {
      await probe.release?.();
      return null;
    }
    // session 留着复用，避免再解析一次模型
    return { device, ort, session: probe };
  } catch (err) {
    console.warn("[VidSharp] 无法取得 ORT 的 WebGPU device:", err);
    return null;
  }
}

/** 单个视频会话的 CNN 上采样器。尺寸变化时无需重建 session（模型是全卷积）。 */
export class CnnUpscaler {
  /** @param ortCtx getOrtDevice() 的返回值 —— device 必须来自 ORT */
  constructor(ortCtx) {
    this.device = ortCtx.device;
    this.ort = ortCtx.ort;
    this.session = ortCtx.session;
    this.inSize = { w: 0, h: 0 };
    this.inBuffer = null;
    this.outBuffer = null;
    this.dimsIn = null;
    this.dimsOut = null;
    this.toTensorPipeline = null;
    this.fromTensorPipeline = null;
    this.ready = false;
    this.failed = false;
  }

  /** 编译桥接 shader。ORT 与模型已在 getOrtDevice 阶段就绪。 */
  async init(shaders, interFormat) {
    if (this.failed) return false;
    try {
      const dev = this.device;
      const toMod = dev.createShaderModule({
        code: shaders.toTensor, label: "to-tensor",
      });
      this.toTensorPipeline = dev.createComputePipeline({
        label: "to-tensor",
        layout: "auto",
        compute: { module: toMod, entryPoint: "main" },
      });

      const fromMod = dev.createShaderModule({
        code: shaders.fromTensor, label: "from-tensor",
      });
      this.fromTensorPipeline = dev.createRenderPipeline({
        label: "from-tensor",
        layout: "auto",
        vertex: { module: fromMod, entryPoint: "vs_main" },
        fragment: {
          module: fromMod,
          entryPoint: "fs_main",
          targets: [{ format: interFormat }],
        },
        primitive: { topology: "triangle-list" },
      });

      this.ready = true;
      return true;
    } catch (err) {
      console.warn("[VidSharp] CNN 桥接 shader 编译失败，回退 EASU:", err);
      this.failed = true;
      this.ready = false;
      return false;
    }
  }

  /** 按输入尺寸分配 GPU buffer。尺寸未变则复用。 */
  ensureBuffers(w, h) {
    if (w === this.inSize.w && h === this.inSize.h && this.inBuffer) return;

    this.releaseBuffers();
    const dev = this.device;
    const inElems = 3 * w * h;
    const outElems = 3 * w * XLSR_SCALE * h * XLSR_SCALE;

    // ORT 要求 STORAGE | COPY_SRC | COPY_DST，且 size 需 16 字节对齐
    const align = (n) => Math.ceil((n * 4) / 16) * 16;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
                | GPUBufferUsage.COPY_DST;

    this.inBuffer = dev.createBuffer({
      label: "cnn-in", size: align(inElems), usage,
    });
    this.outBuffer = dev.createBuffer({
      label: "cnn-out", size: align(outElems), usage,
    });

    this.dimsIn = dev.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.dimsIn, 0, new Uint32Array([w, h, 0, 0]));

    this.dimsOut = dev.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(this.dimsOut, 0, new Uint32Array([
      w * XLSR_SCALE, h * XLSR_SCALE, 0, 0,
    ]));

    this.inSize = { w, h };
  }

  /* 一帧的完整流程。
   *
   * 必须分三次提交，因为 ORT 的 run() 是独立的命令提交，无法插进调用方的
   * encoder：
   *   1) 调用方 encoder 之外单独提交 texture→buffer（compute）
   *   2) await session.run() —— ORT 内部自行提交
   *   3) 把 buffer→texture 挂进调用方的 encoder，与后续 pass 一起提交
   *
   * 返回 true 表示成功写入 dstTex；false 表示应降级。
   */
  async upscale(srcTex, srcW, srcH, dstTex, encoder) {
    if (!this.ready) return false;

    try {
      this.ensureBuffers(srcW, srcH);
      const dev = this.device;

      // --- 1. 纹理 → NCHW buffer（单独提交，ORT 需要已完成的数据） ---
      const preEnc = dev.createCommandEncoder({ label: "cnn-pre" });
      const cpass = preEnc.beginComputePass();
      cpass.setPipeline(this.toTensorPipeline);
      cpass.setBindGroup(0, dev.createBindGroup({
        layout: this.toTensorPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcTex.createView() },
          { binding: 1, resource: { buffer: this.inBuffer } },
          { binding: 2, resource: { buffer: this.dimsIn } },
        ],
      }));
      cpass.dispatchWorkgroups(
        Math.ceil(srcW / 8), Math.ceil(srcH / 8), 1,
      );
      cpass.end();
      dev.queue.submit([preEnc.finish()]);

      // --- 2. 推理，输入输出都留在 GPU ---
      const inTensor = this.ort.Tensor.fromGpuBuffer(this.inBuffer, {
        dataType: "float32",
        dims: [1, 3, srcH, srcW],
      });
      const outTensor = this.ort.Tensor.fromGpuBuffer(this.outBuffer, {
        dataType: "float32",
        dims: [1, 3, srcH * XLSR_SCALE, srcW * XLSR_SCALE],
      });

      const feeds = {};
      feeds[this.session.inputNames[0]] = inTensor;
      const fetches = {};
      fetches[this.session.outputNames[0]] = outTensor;
      await this.session.run(feeds, fetches);

      // --- 3. buffer → 纹理，挂进调用方 encoder ---
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: dstTex.createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.fromTensorPipeline);
      pass.setBindGroup(0, dev.createBindGroup({
        layout: this.fromTensorPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.outBuffer } },
          { binding: 1, resource: { buffer: this.dimsOut } },
        ],
      }));
      pass.draw(3);
      pass.end();

      return true;
    } catch (err) {
      console.warn("[VidSharp] CNN 推理失败，回退 EASU:", err);
      this.failed = true;
      this.ready = false;
      return false;
    }
  }

  releaseBuffers() {
    this.inBuffer?.destroy();
    this.outBuffer?.destroy();
    this.dimsIn?.destroy();
    this.dimsOut?.destroy();
    this.inBuffer = null;
    this.outBuffer = null;
    this.dimsIn = null;
    this.dimsOut = null;
    this.inSize = { w: 0, h: 0 };
  }

  async dispose() {
    this.releaseBuffers();
    this.ready = false;
    // session 属于共享的 ortCtx（多个视频会话复用同一个模型），
    // 不在此处释放 —— 否则第二个视频会拿到已释放的 session
    this.session = null;
  }
}
