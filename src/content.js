/* VidSharp — WebGPU 视频增强
 *
 * 管线（顺序有讲究，pass 数随设置变化）：
 *
 *   <video> --importExternalTexture-->
 *     [1] enhance    去块 + 去色带 + 局部对比度   (源分辨率)
 *     [2] 放大       easu / ravu / xlsr           (→ 目标分辨率)
 *     [3] rcas       自适应锐化
 *     [4] grain      合成胶片颗粒（可选）
 *     [5] badge      状态角标合成（可选）
 *   --> <canvas>
 *
 * XLSR 档是两段式：先 downscale 到目标÷4，再由 CNN 放大 4 倍。这样任意
 * 目标倍率都能用 CNN，且 CNN 负担只由目标尺寸决定、不随源分辨率膨胀。
 *
 * 为什么是这个顺序：
 *   - 去块必须在放大前：否则 EASU 会把块边界当真实边缘"保护"，块效应被锐化。
 *   - 锐化必须在放大后：放大前锐化会被插值重新糊掉。
 *   - 颗粒必须在锐化后：否则 RCAS 会把颗粒当细节放大，变成噪点。
 *   - 角标始终最后：避免被后续 pass 处理。
 *
 * 关键约束（改代码前先读）：
 *   - external texture 单帧有效：每帧重新 import，bind group 也必须重建。
 *   - 覆盖层不能压过弹幕/控制条：插到 video 的直接后继位置，继承 video 的
 *     层级，而不是把 z-index 顶到最大（那样会盖住播放器 UI）。
 *   - 可选 pass 用 sharpened/pong 两张纹理乒乓串联 —— 同一张纹理不能在一个
 *     pass 里既读又写；最后一个 pass 直接写 canvas 以省一次全屏绘制。
 *   - DRM 视频取不到像素，靠 consecutiveFailures 检测后自动退出。
 */

const SHADERS = {
  enhance: chrome.runtime.getURL("src/enhance.wgsl"),
  easu: chrome.runtime.getURL("src/easu.wgsl"),
  ravu: chrome.runtime.getURL("src/ravu.wgsl"),
  rcas: chrome.runtime.getURL("src/rcas.wgsl"),
  downscale: chrome.runtime.getURL("src/downscale.wgsl"),
  toTensor: chrome.runtime.getURL("src/to-tensor.wgsl"),
  fromTensor: chrome.runtime.getURL("src/from-tensor.wgsl"),
  grain: chrome.runtime.getURL("src/grain.wgsl"),
  badge: chrome.runtime.getURL("src/badge.wgsl"),
};

const DEFAULTS = {
  enabled: false,
  strength: 50,      // RCAS 锐化 0~100
  denoise: true,     // RCAS 噪声抑制
  deblock: 40,       // 去块 0~100
  deband: 30,        // 去色带 0~100
  contrast: 25,      // 局部对比度 0~100
  grain: 0,          // 胶片颗粒 0~100（默认关，需按片源调）
  grainSize: 1.0,    // 颗粒粗细 1~3
  upscale: "2k",     // off | 2k | 4k | 2x
  upscaler: "easu",  // easu | ravu | xlsr — 放大算法
  compare: false,    // 拖动对比模式
  badge: "corner",   // off | corner | detail | perf — 增强状态标识
  preSuperRes: false,      // 预超分（实验性）
  preSuperResMode: "raw",  // encode（省显存，需硬件编码器）| raw（吃显存，零编解码开销）
  preSuperResBudget: 512,  // raw 模式显存预算 MB
};

let settings = { ...DEFAULTS };
const active = new WeakMap();

/* ---------- 目标分辨率 ---------- */

const UPSCALE_TARGETS = {
  "2k": 2560,
  "4k": 3840,
};

/** 计算输出尺寸。保持宽高比，不放大已经足够大的视频，受 GPU 纹理上限约束。 */
function computeOutputSize(videoW, videoH, mode, maxDim) {
  if (mode === "off" || !videoW || !videoH) {
    return { w: videoW, h: videoH };
  }

  let scale;
  if (mode === "2x") {
    scale = 2;
  } else if (mode === "4x") {
    // 4x 主要为 XLSR 服务（模型固定 4 倍）；EASU/RAVU 也能用，只是
    // 4 倍放大后画面偏软，通常还要配合较高锐化
    scale = 4;
  } else {
    const targetW = UPSCALE_TARGETS[mode];
    if (!targetW) return { w: videoW, h: videoH };
    // 已达到或超过目标宽度就不放大 —— 下采样只会损失画质
    if (videoW >= targetW) return { w: videoW, h: videoH };
    scale = targetW / videoW;
  }

  // 受纹理上限约束（M1 是 8192）
  const limit = Math.min(
    maxDim / videoW,
    maxDim / videoH,
  );
  scale = Math.min(scale, limit);
  if (scale <= 1) return { w: videoW, h: videoH };

  return {
    w: Math.round(videoW * scale),
    h: Math.round(videoH * scale),
  };
}

/* CNN 档的输入尺寸 = 目标尺寸 ÷ 4。
 *
 * XLSR 固定 4x（末端 DepthToSpace 把 48 通道重排成 4×4，4 编译在权重里），
 * 无法配置。要支持任意目标倍率，就先把画面缩放到目标的 1/4，再让 CNN
 * 放大回来。
 *
 * 例：1080p → 2K 时先缩到 640×360，CNN 输出正好 2560×1440。
 *
 * 附带的关键收益：CNN 的输入不再随源分辨率膨胀。此前 1080p 选「4 倍」
 * 会让 CNN 处理 1920×1080→7680×4320（16 倍像素量，实测卡顿）；现在
 * 无论源多大，CNN 的负担只由目标尺寸决定。
 *
 * 也更符合模型的训练分布 —— 它是在低分辨率输入上训练的。
 */
function cnnInputSize(outW, outH) {
  return {
    w: Math.max(1, Math.round(outW / CNN_SCALE)),
    h: Math.max(1, Math.round(outH / CNN_SCALE)),
  };
}

/** XLSR 的固定放大倍率 */
const CNN_SCALE = 4;

/* ---------- WebGPU 设备 ---------- */

/* WebGPU device 获取。
 *
 * 有两条路径，且不可混用：
 *   - 纯 shader 档（EASU/RAVU）：自行 requestDevice()
 *   - CNN 档（XLSR）：必须用 ORT 内部创建的 device
 *
 * 原因是 WebGPU 规定 buffer/纹理只能用于创建它的 device，而 ORT 1.27 不
 * 接受外部传入的 device（只能通过 ort.env.webgpu.device 读取它自己的）。
 * 混用会报 "[Buffer] is associated with [Device], and cannot be used with
 * [Device]"。
 *
 * 因此切换 CNN 档时整个会话必须重建 —— device 换了，所有 pipeline、纹理、
 * canvas context 都要跟着换。这个代价换来的是全程零拷贝，值得。
 */
let devicePromise = null;
let deviceIsOrt = false;
/** ORT 上下文（device + ort + 预热好的 session），仅 CNN 档下非空 */
let ortCtx = null;

function onDeviceLost(info) {
  console.warn("[VidSharp] GPU 设备丢失:", info.message);
  devicePromise = null;
  deviceIsOrt = false;
  for (const video of document.querySelectorAll("video")) {
    active.get(video)?.stop();
  }
}

/** @param wantOrt 是否需要 ORT 的 device（CNN 档） */
function getDevice(wantOrt) {
  // 需求与当前 device 来源不符时，丢弃重建
  if (devicePromise && wantOrt !== deviceIsOrt) {
    devicePromise = null;
  }
  if (!devicePromise) {
    deviceIsOrt = wantOrt;
    devicePromise = (async () => {
      if (!navigator.gpu) throw new Error("此浏览器不支持 WebGPU");

      if (wantOrt) {
        const mod = await import(chrome.runtime.getURL("src/cnn.js"));
        const ctx = await mod.getOrtDevice();
        if (ctx) {
          ctx.device.lost.then(onDeviceLost);
          ortCtx = ctx;
          return ctx.device;
        }
        // 取不到就退回自建 device，CNN 档随之不可用
        console.warn("[VidSharp] 无法使用 ORT device，CNN 档不可用");
        deviceIsOrt = false;
        ortCtx = null;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("无法获取 GPU adapter");
      const device = await adapter.requestDevice();
      device.lost.then(onDeviceLost);
      return device;
    })();
    devicePromise.catch(() => { devicePromise = null; deviceIsOrt = false; });
  }
  return devicePromise;
}

let shaderCache = null;
function getShaders() {
  if (!shaderCache) {
    shaderCache = Promise.all(
      Object.entries(SHADERS).map(([name, url]) =>
        fetch(url).then((r) => r.text()).then((code) => [name, code]),
      ),
    ).then(Object.fromEntries);
    shaderCache.catch(() => { shaderCache = null; });
  }
  return shaderCache;
}

/* RAVU 的滤波系数表。全页面共享一张纹理 —— 它是只读常量，无需按会话复制。 */
let ravuLutPromise = null;
let ravuLutDevice = null;
function getRavuLut(device) {
  // 纹理绑定在 device 上：切换 CNN 档会换 device，此时必须重建
  if (ravuLutPromise && ravuLutDevice !== device) {
    ravuLutPromise = null;
  }
  if (!ravuLutPromise) {
    ravuLutDevice = device;
    ravuLutPromise = (async () => {
      const url = chrome.runtime.getURL("src/data/ravu-lut.js");
      const { RAVU_LUT } = await import(url);
      const bin = Uint8Array.from(atob(RAVU_LUT.data), (c) => c.charCodeAt(0));
      const tex = device.createTexture({
        label: "ravu-lut",
        size: [RAVU_LUT.width, RAVU_LUT.height],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: tex }, bin,
        { bytesPerRow: RAVU_LUT.width * 8 },   // 4 通道 × fp16
        [RAVU_LUT.width, RAVU_LUT.height],
      );
      return tex;
    })();
    ravuLutPromise.catch(() => { ravuLutPromise = null; });
  }
  return ravuLutPromise;
}

/* CNN 上采样器按需创建。
 *
 * cnn.js 与 26MB 的 ORT wasm 只在用户实际选中 XLSR 档时才加载 —— 默认档
 * （EASU）完全不碰它们，因此不影响绝大多数使用场景的内存与启动时间。
 *
 * 加载或初始化失败时返回一个 ready=false 的占位对象而非抛错：调用方只需
 * 检查 ready，自然回落 EASU。
 */
async function ensureCnn(shaders, interFormat) {
  // ortCtx 由 getDevice(true) 建立。若为空说明 device 不是 ORT 的，
  // 此时创建 CnnUpscaler 只会在推理时报跨 device 错误，直接放弃。
  if (!ortCtx) {
    return { ready: false, failed: true, dispose() {} };
  }
  try {
    const mod = await import(chrome.runtime.getURL("src/cnn.js"));
    const up = new mod.CnnUpscaler(ortCtx);
    await up.init(shaders, interFormat);
    if (up.ready) notify("CNN 超分已启用（实验性）");
    return up;
  } catch (err) {
    console.warn("[VidSharp] CNN 初始化失败，使用 EASU:", err);
    notify("CNN 超分不可用，已回退 EASU");
    return { ready: false, failed: true, dispose() {} };
  }
}

/* 渲染性能统计。
 *
 * 排查卡顿必须区分三件事，只看"帧率"会误判：
 *   - 渲染帧率：我们实际画了多少帧
 *   - 视频帧率：解码器送出多少帧（rVFC 的触发次数上限）
 *   - GPU 耗时：单帧处理占了多少预算
 *
 * 渲染帧率低于视频帧率 = 我们跟不上（真卡顿）；
 * 两者接近但都低 = 视频本身帧率低或网络卡（不是我们的问题）。
 */
class PerfStats {
  constructor() {
    this.reset();
  }

  reset() {
    this.renderTimes = [];      // 最近若干帧的 GPU 提交耗时
    this.frameStamps = [];      // 最近若干帧的时间点，用于算 fps
    this.presentedFrames = 0;   // rVFC 报告的已呈现视频帧数
    this.lastPresented = 0;
    this.droppedFrames = 0;
    this.videoFps = 0;
    this.maxSamples = 60;
  }

  /** 每帧渲染结束时调用 */
  record(ms, rvfcMeta) {
    const now = performance.now();
    this.renderTimes.push(ms);
    this.frameStamps.push(now);
    if (this.renderTimes.length > this.maxSamples) this.renderTimes.shift();
    if (this.frameStamps.length > this.maxSamples) this.frameStamps.shift();

    /* rVFC 的 metadata 直接给出解码器已呈现/已丢弃的帧数，比自己数更准 ——
     * 它能区分"我们没画"和"解码器没给"。 */
    if (rvfcMeta) {
      if (typeof rvfcMeta.presentedFrames === "number") {
        const delta = rvfcMeta.presentedFrames - this.lastPresented;
        if (this.lastPresented && delta > 1) {
          // 视频帧号跳跃 = 我们漏掉了中间帧
          this.droppedFrames += delta - 1;
        }
        this.lastPresented = rvfcMeta.presentedFrames;
        this.presentedFrames = rvfcMeta.presentedFrames;
      }
    }
  }

  /** 实际渲染帧率（按最近样本的时间跨度算） */
  get fps() {
    if (this.frameStamps.length < 2) return 0;
    const span = this.frameStamps[this.frameStamps.length - 1] - this.frameStamps[0];
    if (span <= 0) return 0;
    return ((this.frameStamps.length - 1) * 1000) / span;
  }

  /** 单帧渲染耗时中位数。用中位数而非均值 —— 偶发长帧不该主导判断 */
  get medianMs() {
    if (!this.renderTimes.length) return 0;
    const sorted = [...this.renderTimes].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  get maxMs() {
    return this.renderTimes.length ? Math.max(...this.renderTimes) : 0;
  }
}

/* ---------- 处理会话 ---------- */

class Session {
  constructor(video) {
    this.video = video;
    this.canvas = null;
    this.ctx = null;
    this.device = null;
    this.pipelines = {};
    this.sampler = null;
    this.enhanceUniform = null;
    this.rcasUniform = null;
    this.intermediates = { enhanced: null, upscaled: null, sharpened: null, pong: null, cnnInput: null };
    this.rvfcHandle = null;
    this.running = false;
    this.consecutiveFailures = 0;
    this.frameCount = 0;
    this.srcSize = { w: 0, h: 0 };
    this.outSize = { w: 0, h: 0 };
    this.divider = null;
    this.compareRatio = 0.5;
    this.badgeTexture = null;
    this.badgeText = null;
    this.badgeSize = null;
    this.badgeUniform = null;
    this.grainUniform = null;
    this.downscaleUniform = null;
    this.cnnInputSize = null;
    this.ravuLut = null;   // 共享纹理，会话结束不销毁
    this.cnn = null;
    this.activeUpscaler = "EASU";  // 实际生效的放大器，供角标显示
    this.perf = new PerfStats();
    this.presr = null;
    this.presrTex = null;
    this.presrTexSize = null;
    this.warnedScale = false;
    this.rendering = false;
    this.firstFrameDone = false;
    this.renderFrame = this.renderFrame.bind(this);
    this.syncSizes = this.syncSizes.bind(this);
  }

  async start() {
    if (this.running) return;
    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
      throw new Error("此浏览器不支持 requestVideoFrameCallback");
    }

    // CNN 档需要 ORT 的 device —— 二者不可混用，见 getDevice 注释
    const wantOrt = settings.upscaler === "xlsr";
    const [device, shaders] = await Promise.all([
      getDevice(wantOrt), getShaders(),
    ]);
    this.device = device;

    this.canvas = document.createElement("canvas");
    this.canvas.className = "vidsharp-overlay";
    this.ctx = this.canvas.getContext("webgpu");
    if (!this.ctx) throw new Error("无法创建 WebGPU canvas context");

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });

    // 中间纹理用 rgba16float：多 pass 串联时避免 8bit 量化累积误差，
    // 尤其去色带的亚量化级抖动在 8bit 中间纹理里会被直接抹掉。
    this.interFormat = "rgba16float";

    this.pipelines.enhance = this.buildPipeline(
      shaders.enhance, "enhance", this.interFormat, true,
    );
    this.pipelines.easu = this.buildPipeline(
      shaders.easu, "easu", this.interFormat, false,
    );
    // 高质量降采样，用于 CNN 两段式的预缩放
    this.pipelines.downscale = this.buildPipeline(
      shaders.downscale, "downscale", this.interFormat, false,
    );
    // RAVU 固定 2x，作为 EASU 的可选替代：查表法而非解析近似，更锐
    this.pipelines.ravu = this.buildPipeline(
      shaders.ravu, "ravu", this.interFormat, false,
    );
    this.pipelines.rcas = this.buildPipeline(
      shaders.rcas, "rcas", this.format, true,
    );
    // 角标合成 pass 直接输出到 canvas，故用 canvas 格式；
    // 它的输入是 RCAS 已锐化的画面，也用同一格式以避免多余转换。
    // grain 与 badge 都在 canvas 格式上工作（管线末端）
    this.pipelines.grain = this.buildPipeline(
      shaders.grain, "grain", this.format, false,
    );
    this.pipelines.badge = this.buildPipeline(
      shaders.badge, "badge", this.format, false,
    );

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.enhanceUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.rcasUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // dstWidth, dstHeight, padding — 降采样目标尺寸
    this.downscaleUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // strength, size, frameSeed, chroma
    this.grainUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // rect(vec4) + opacity + 3 padding = 8 x f32
    this.badgeUniform = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.updateUniforms();

    // LUT 加载失败不应让整个会话挂掉 —— 只是 RAVU 档不可用，回落 EASU
    this.ravuLut = await getRavuLut(device).catch((err) => {
      console.warn("[VidSharp] RAVU LUT 加载失败，将使用 EASU:", err);
      return null;
    });

    this.shaders = shaders;   // CNN 按需初始化时要用到桥接 shader

    this.attachOverlay();

    // 预超分：拦截视频流提前处理。需硬件编码器，自检不通过会静默不启用。
    if (settings.preSuperRes) this.initPreSuperRes(shaders);

    // seek 后已解码的超分帧全部失效
    this.onSeeking = () => this.presr?.onSeek();
    this.video.addEventListener("seeking", this.onSeeking);

    this.running = true;
    this.scheduleNext();
  }

  buildPipeline(code, label, targetFormat, external) {
    const module = this.device.createShaderModule({ code, label });
    return this.device.createRenderPipeline({
      label: `${label}-pipeline`,
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: targetFormat }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  /* ---- 挂载：不遮挡播放器 UI ---- */

  attachOverlay() {
    const parent = this.video.parentElement;
    if (!parent) throw new Error("video 没有父元素，无法挂载");

    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
      this.patchedParent = parent;
    }

    // 插到 video 的紧后面：DOM 顺序上压过 video 本身，但仍在弹幕层、
    // 控制条（它们是后续兄弟节点或更高层级容器）之下。
    // 这比把 z-index 顶到最大更正确 —— 后者会盖住所有播放器 UI。
    this.video.insertAdjacentElement("afterend", this.canvas);

    this.resizeObserver = new ResizeObserver(this.syncSizes);
    this.resizeObserver.observe(this.video);
    this.syncSizes();
    this.setupCompare();
  }

  syncSizes() {
    if (!this.canvas || !this.device) return;
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh) return;

    const maxDim = this.device.limits.maxTextureDimension2D;
    const out = computeOutputSize(vw, vh, settings.upscale, maxDim);

    const srcChanged = vw !== this.srcSize.w || vh !== this.srcSize.h;
    const outChanged = out.w !== this.outSize.w || out.h !== this.outSize.h;
    if (!srcChanged && !outChanged) return;

    this.srcSize = { w: vw, h: vh };
    this.outSize = out;
    this.warnedScale = false;   // 尺寸变了，允许再次提示倍率不匹配
    this.canvas.width = out.w;
    this.canvas.height = out.h;

    this.rebuildIntermediates();
    // 分辨率变了，角标字号与位置都要重算 —— 必须重建而非复用旧纹理
    this.refreshBadge();
  }

  rebuildIntermediates() {
    const { device, interFormat } = this;
    this.intermediates.enhanced?.destroy();
    this.intermediates.upscaled?.destroy();
    this.intermediates.sharpened?.destroy();
    this.intermediates.pong?.destroy();

    const usage =
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;

    this.intermediates.enhanced = device.createTexture({
      label: "enhanced",
      size: [this.srcSize.w, this.srcSize.h],
      format: interFormat,
      usage,
    });

    // 放大后的中间纹理。若不放大则与源尺寸相同，仍需独立纹理（不能读写同一张）。
    this.intermediates.upscaled = device.createTexture({
      label: "upscaled",
      size: [this.outSize.w, this.outSize.h],
      format: interFormat,
      usage,
    });

    // 角标启用时 RCAS 的落点。格式必须与 rcas pipeline 的 target 一致
    // （即 canvas 格式），否则 render pass 校验失败。
    this.intermediates.sharpened = device.createTexture({
      label: "sharpened",
      size: [this.outSize.w, this.outSize.h],
      format: this.format,
      usage,
    });

    // 乒乓用的第二张。多个可选 pass（grain/badge）串联时需要交替读写 ——
    // 同一张纹理不能在一个 pass 里既作输入又作输出。
    this.intermediates.pong = device.createTexture({
      label: "pong",
      size: [this.outSize.w, this.outSize.h],
      format: this.format,
      usage,
    });
  }

  /* ---- 增强状态角标 ----
   *
   * 存在的意义：画面被 canvas 接管后，肉眼无法确认看到的是原始视频还是
   * 增强结果。
   *
   * 为什么画进 canvas 而不用独立 DOM 元素：
   *   独立角标与 canvas 是两个元素，会被站点的不同层级分别遮挡 ——
   *   出现"角标可见但 canvas 被挡"的情况，指示就成了误导。
   *   合成进画面后，看得见角标 ⟺ 看得见增强结果，无法解耦。
   *
   * 文字用 2D canvas 预渲染成纹理，仅在文案变化时重建；每帧只做一次
   * alpha 混合，开销可忽略。
   */

  badgeString() {
    const { w: sw, h: sh } = this.srcSize;
    const { w: ow, h: oh } = this.outSize;
    const upscaled = ow > sw;

    if (settings.badge === "detail") {
      const size = upscaled ? `${sw}×${sh} → ${ow}×${oh}` : `${ow}×${oh}`;
      const fx = [];
      // 用渲染循环记录的实际放大器，而非按设置推测 —— 倍率不匹配、
      // CNN 加载失败等情况都会静默回落，推测会给出错误信息。
      if (upscaled) fx.push(this.activeUpscaler || "EASU");
      if (settings.strength > 0) fx.push(`锐化 ${settings.strength}`);
      if (settings.deblock > 0) fx.push(`去块 ${settings.deblock}`);
      if (settings.deband > 0) fx.push(`去色带 ${settings.deband}`);
      if (settings.contrast > 0) fx.push(`对比 ${settings.contrast}`);
      return `VidSharp · ${size}${fx.length ? " · " + fx.join(" / ") : ""}`;
    }
    if (settings.badge === "perf") {
      const p = this.perf;
      const fps = p.fps;
      // 视频本身的帧率上限：rVFC 不会超过它，所以低 fps 不一定是我们的锅
      const parts = [
        `${fps.toFixed(0)}fps`,
        `${p.medianMs.toFixed(1)}ms`,
      ];
      if (p.maxMs > p.medianMs * 2) {
        // 最大值远高于中位数说明有卡顿尖峰，这才是"偶尔卡一下"的来源
        parts.push(`峰${p.maxMs.toFixed(0)}ms`);
      }
      if (p.droppedFrames > 0) parts.push(`丢${p.droppedFrames}`);
      parts.push(upscaled ? (this.activeUpscaler || "EASU") : "无放大");
      if (upscaled) parts.push(`${ow}×${oh}`);
      // 预超分的领先量是它是否真在起作用的直接证据
      if (this.presr?.enabled) {
        const us = this.video.currentTime;
        const st = this.presr.mode === "raw"
          ? this.presr.cache?.stats
          : null;
        const ahead = this.presr.mode === "raw"
          ? (this.presr.cache?.aheadSeconds(Math.round(us * 1e6)) ?? 0)
          : 0;
        parts.push(`预备${ahead.toFixed(1)}s`);
        if (st) parts.push(`${st.mb.toFixed(0)}MB`);
      }
      return `VidSharp · ${parts.join(" · ")}`;
    }

    return upscaled ? `VidSharp ${oh}p` : "VidSharp";
  }

  /* perf 模式下角标每秒刷新一次。
   *
   * 不能走 updateBadge 的"文案变了就重建纹理"路径 —— fps 每帧都在变，
   * 那样会每帧重建一次纹理（2D canvas 渲文字 + 上传 GPU），反而制造卡顿。
   * 这里限流到 1Hz，并复用同一张纹理尺寸。 */
  maybeUpdatePerfBadge() {
    if (settings.badge !== "perf") return;
    const now = performance.now();
    if (now - (this.lastPerfBadge || 0) < 1000) return;
    this.lastPerfBadge = now;
    this.badgeText = null;      // 强制重算
    this.updateBadge();
  }

  /** 把文案渲染成 GPU 纹理。文案未变则直接复用。 */
  updateBadge() {
    if (!this.device) return;

    if (settings.badge === "off") {
      this.releaseBadgeTexture();
      return;
    }
    if (!this.outSize.w) return;

    const text = this.badgeString();
    if (text === this.badgeText && this.badgeTexture) return;
    this.badgeText = text;

    // 字号跟随输出分辨率，保证不同分辨率下视觉大小一致
    const scale = Math.max(1, Math.min(3, this.outSize.h / 720));
    const fontSize = Math.round(13 * scale);
    const padX = Math.round(9 * scale);
    const padY = Math.round(5 * scale);
    const radius = Math.round(5 * scale);

    const measure = document.createElement("canvas").getContext("2d");
    const font = `500 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, ` +
                 `Consolas, "Courier New", monospace`;
    measure.font = font;
    const textW = Math.ceil(measure.measureText(text).width);

    const w = textW + padX * 2;
    const h = Math.round(fontSize * 1.5) + padY * 2;

    const c2d = document.createElement("canvas");
    c2d.width = w;
    c2d.height = h;
    const ctx = c2d.getContext("2d");

    // 背景圆角矩形
    ctx.fillStyle = "rgba(16, 16, 20, 0.68)";
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, radius);
    ctx.fill();

    ctx.font = font;
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#7ee0c0";
    ctx.fillText(text, padX, h / 2);

    this.releaseBadgeTexture();
    this.badgeTexture = this.device.createTexture({
      label: "badge",
      size: [w, h],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: c2d },
      { texture: this.badgeTexture, premultipliedAlpha: false },
      [w, h],
    );
    this.badgeSize = { w, h };
    this.writeBadgeUniform();
  }

  writeBadgeUniform() {
    if (!this.badgeUniform || !this.badgeSize || !this.outSize.w) return;
    // 右上角，边距随分辨率缩放
    const margin = Math.round(12 * Math.max(1, this.outSize.h / 720));
    const { w: bw, h: bh } = this.badgeSize;
    const x = (this.outSize.w - bw - margin) / this.outSize.w;
    const y = margin / this.outSize.h;
    this.device.queue.writeBuffer(this.badgeUniform, 0, new Float32Array([
      x, y, bw / this.outSize.w, bh / this.outSize.h,
      0.82, 0, 0, 0,
    ]));
  }

  releaseBadgeTexture() {
    this.badgeTexture?.destroy();
    this.badgeTexture = null;
    this.badgeText = null;
    this.badgeSize = null;
  }

  refreshBadge() {
    // 模式切换只需重算纹理；合成与否由 renderFrame 依 badgeTexture 判断
    this.releaseBadgeTexture();
    this.updateBadge();
  }

  /** CNN 输入纹理按需分配。尺寸未变则复用。 */
  ensureCnnInput(w, h) {
    const cur = this.intermediates.cnnInput;
    if (cur && this.cnnInputSize?.w === w && this.cnnInputSize?.h === h) return;
    cur?.destroy();
    this.intermediates.cnnInput = this.device.createTexture({
      label: "cnn-input",
      size: [w, h],
      format: this.interFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.cnnInputSize = { w, h };
  }

  /** 启动预超分。失败不影响实时管线。 */
  async initPreSuperRes(shaders) {
    try {
      const mod = await import(chrome.runtime.getURL("src/presuperres.js"));
      // 尺寸要等 videoWidth 就绪；syncSizes 已在 attachOverlay 里跑过
      if (!this.outSize.w) return;
      const p = new mod.PreSuperRes(this.device, shaders.easu);
      const ok = await p.start(
        this.outSize.w, this.outSize.h,
        settings.preSuperResMode, settings.preSuperResBudget,
      );
      if (ok) {
        this.presr = p;
        notify("预超分已启用（实验性）");
      } else {
        notify("预超分不可用：" + (p.reason || "自检未通过"));
      }
    } catch (err) {
      console.warn("[VidSharp] 预超分启用失败:", err);
    }
  }

  /** 把一个已处理好的 VideoFrame 直接画到 canvas（预超分命中时用） */
  blitFrame(frame) {
    if (!this.pipelines.blit) {
      // 懒建：只有启用预超分时才需要
      const module = this.device.createShaderModule({
        label: "blit",
        code: `
          @group(0) @binding(0) var t: texture_2d<f32>;
          @group(0) @binding(1) var s: sampler;
          struct VO { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
          @vertex fn vs_main(@builtin(vertex_index) i: u32) -> VO {
            var o: VO;
            let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
            let y = f32(i & 2u) * 2.0 - 1.0;
            o.p = vec4<f32>(x, y, 0.0, 1.0);
            o.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
            return o;
          }
          @fragment fn fs_main(v: VO) -> @location(0) vec4<f32> {
            return textureSampleLevel(t, s, v.uv, 0.0);
          }`,
      });
      this.pipelines.blit = this.device.createRenderPipeline({
        label: "blit", layout: "auto",
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint: "fs_main",
                    targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list" },
      });
    }

    // VideoFrame → 纹理需要 copyExternalImageToTexture
    // （importExternalTexture 只接受 HTMLVideoElement）
    const w = frame.displayWidth, h = frame.displayHeight;
    if (!this.presrTex || this.presrTexSize?.w !== w || this.presrTexSize?.h !== h) {
      this.presrTex?.destroy();
      this.presrTex = this.device.createTexture({
        label: "presr-frame",
        size: [w, h], format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
             | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.presrTexSize = { w, h };
    }
    this.device.queue.copyExternalImageToTexture(
      { source: frame }, { texture: this.presrTex }, [w, h],
    );

    const enc = this.device.createCommandEncoder({ label: "presr-blit" });
    this.drawPass(enc, this.pipelines.blit, [
      { binding: 0, resource: this.presrTex.createView() },
      { binding: 1, resource: this.sampler },
    ], this.ctx.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);
  }

  /* ---- 拖动对比 ---- */

  setupCompare() {
    if (!settings.compare) return;

    const divider = document.createElement("div");
    divider.className = "vidsharp-divider";
    divider.innerHTML =
      '<div class="vidsharp-divider-line"></div>' +
      '<div class="vidsharp-divider-handle">⇔</div>';
    this.divider = divider;
    this.canvas.parentElement.appendChild(divider);

    const setRatio = (clientX) => {
      const rect = this.video.getBoundingClientRect();
      if (!rect.width) return;
      this.compareRatio = Math.min(
        1, Math.max(0, (clientX - rect.left) / rect.width),
      );
      this.applyCompare();
    };

    let dragging = false;
    const onDown = (ev) => {
      dragging = true;
      divider.setPointerCapture?.(ev.pointerId);
      setRatio(ev.clientX);
      ev.preventDefault();
      ev.stopPropagation();
    };
    const onMove = (ev) => {
      if (!dragging) return;
      setRatio(ev.clientX);
      ev.preventDefault();
      ev.stopPropagation();
    };
    const onUp = (ev) => {
      dragging = false;
      divider.releasePointerCapture?.(ev.pointerId);
    };

    divider.addEventListener("pointerdown", onDown);
    divider.addEventListener("pointermove", onMove);
    divider.addEventListener("pointerup", onUp);
    divider.addEventListener("pointercancel", onUp);
    this.compareHandlers = { divider, onDown, onMove, onUp };

    this.applyCompare();
  }

  applyCompare() {
    if (!this.canvas) return;
    if (settings.compare) {
      const pct = (this.compareRatio * 100).toFixed(2);
      // 只显示左侧处理结果，右侧透出原始 video
      this.canvas.style.clipPath = `inset(0 ${(100 - pct).toFixed(2)}% 0 0)`;
      if (this.divider) this.divider.style.left = `${pct}%`;
    } else {
      this.canvas.style.clipPath = "";
    }
  }

  teardownCompare() {
    if (this.compareHandlers) {
      const { divider, onDown, onMove, onUp } = this.compareHandlers;
      divider.removeEventListener("pointerdown", onDown);
      divider.removeEventListener("pointermove", onMove);
      divider.removeEventListener("pointerup", onUp);
      divider.removeEventListener("pointercancel", onUp);
      this.compareHandlers = null;
    }
    this.divider?.remove();
    this.divider = null;
    if (this.canvas) this.canvas.style.clipPath = "";
  }

  refreshCompare() {
    this.teardownCompare();
    if (settings.compare && this.canvas?.parentElement) this.setupCompare();
  }

  /* ---- uniforms ---- */

  updateUniforms() {
    if (!this.enhanceUniform) return;
    const pct = (v) => Math.max(0, Math.min(1, v / 100));

    this.device.queue.writeBuffer(this.enhanceUniform, 0, new Float32Array([
      pct(settings.deblock),
      pct(settings.deband),
      pct(settings.contrast),
      this.frameCount % 1024,
    ]));

    this.device.queue.writeBuffer(this.rcasUniform, 0, new Float32Array([
      pct(settings.strength),
      settings.denoise ? 1 : 0,
      0,
      0,
    ]));

    this.device.queue.writeBuffer(this.grainUniform, 0, new Float32Array([
      pct(settings.grain),
      Math.max(1, Math.min(3, settings.grainSize)),
      // 颗粒图案必须每帧变化，否则固定噪点会被看成屏幕脏污
      this.frameCount % 4096,
      0.15,   // 色度颗粒比例：真实胶片彩色颗粒较弱
    ]));
  }

  /* ---- 渲染 ---- */

  scheduleNext() {
    if (!this.running) return;
    // rVFC 的第二个参数带 presentedFrames/processingDuration 等诊断信息
    this.rvfcHandle = this.video.requestVideoFrameCallback(this.renderFrame);
  }

  drawPass(encoder, pipeline, entries, targetView) {
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView,
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  async renderFrame(_now, rvfcMeta) {
    if (!this.running) return;
    const tStart = performance.now();
    // CNN 推理是异步的，慢帧可能与下一次 rVFC 回调重叠。重入会让两帧
    // 争用同一批 GPU buffer，必须丢弃而非排队 —— 视频场景下延迟比完整
    // 性更重要。
    if (this.rendering) { this.scheduleNext(); return; }
    this.rendering = true;

    try {
      this.syncSizes();
      if (!this.canvas.width || !this.canvas.height) {
        return;   // finally 会清 rendering，随后由下方 scheduleNext 续上
      }

      this.frameCount++;
      // 去色带与颗粒都依赖帧号做时域去相关，图案必须每帧变化，
      // 否则固定噪点会被眼睛识别为屏幕脏污
      if (settings.deband > 0 || settings.grain > 0) this.updateUniforms();

      /* 预超分命中时走捷径：那一帧已经超分并编码过，直接显示即可，
       * 不必再跑一遍 enhance/放大/锐化 —— 重复处理只会过锐并浪费算力。 */
      if (this.presr?.enabled) {
        const pre = this.presr.getFrame(this.video.currentTime);
        if (pre) {
          this.blitFrame(pre);
          this.consecutiveFailures = 0;
          this.canvas.classList.add("vidsharp-visible");
          this.firstFrameDone = true;
          if (this.activeUpscaler !== "预超分") {
            this.activeUpscaler = "预超分";
            this.refreshBadge();
          }
          this.perf.record(performance.now() - tStart, rvfcMeta);
          this.maybeUpdatePerfBadge();
          return;   // finally 会清 rendering 并续帧
        }
        // 未命中就回落实时管线（下面的正常流程）
      }

      const external = this.device.importExternalTexture({
        source: this.video,
      });

      // Pass 1 单独提交：CNN 档需要 enhance 的结果已经落地才能读取
      // （ORT 的 run() 是独立命令提交，插不进同一个 encoder）。
      const preEncoder = this.device.createCommandEncoder({ label: "pre" });
      this.drawPass(preEncoder, this.pipelines.enhance, [
        { binding: 0, resource: external },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.enhanceUniform } },
      ], this.intermediates.enhanced.createView());
      this.device.queue.submit([preEncoder.finish()]);

      const encoder = this.device.createCommandEncoder();

      // Pass 2: 放大
      //
      // 三种放大器的适用倍率不同：
      //   EASU  任意倍率（解析近似，运行时估边缘方向）
      //   RAVU  仅 2x —— LUT 按 2x 子像素布局训练，非 2x 会索引错位出网格伪影
      //   XLSR  任意倍率 —— 通过"先缩到目标÷4、再 CNN 放大 4 倍"实现
      // 倍率不匹配时静默回落 EASU。
      const scaleX = this.outSize.w / Math.max(1, this.srcSize.w);
      const useRavu = settings.upscaler === "ravu"
        && this.ravuLut != null
        && Math.abs(scaleX - 2) < 0.02;
      // CNN 按需初始化：24MB 的 wasm 只在用户真正选中该档时才加载。
      // 不再要求特定倍率 —— 两段式对任意目标尺寸都成立。
      if (settings.upscaler === "xlsr" && !this.cnn) {
        this.cnn = await ensureCnn(this.shaders, this.interFormat);
      }
      const useCnn = settings.upscaler === "xlsr" && this.cnn?.ready;

      let upscaled = false;
      let activeName = "EASU";
      if (useCnn) {
        // 第一段：缩放到 CNN 需要的输入尺寸（目标÷4）。
        // 缩小时走高质量区域平均以避免摩尔纹；若源本就小于该尺寸，
        // downscale shader 内部会退化为双线性放大。
        const inSize = cnnInputSize(this.outSize.w, this.outSize.h);
        this.ensureCnnInput(inSize.w, inSize.h);
        const preEnc = this.device.createCommandEncoder({ label: "cnn-downscale" });
        this.device.queue.writeBuffer(this.downscaleUniform, 0,
          new Float32Array([inSize.w, inSize.h, 0, 0]));
        this.drawPass(preEnc, this.pipelines.downscale, [
          { binding: 0, resource: this.intermediates.enhanced.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.downscaleUniform } },
        ], this.intermediates.cnnInput.createView());
        this.device.queue.submit([preEnc.finish()]);

        // 第二段：CNN 放大 4 倍
        upscaled = await this.cnn.upscale(
          this.intermediates.cnnInput,
          inSize.w, inSize.h,
          this.intermediates.upscaled,
          encoder,
        );
        // CNN 失败会置 failed，下一帧起自动走 EASU
        if (upscaled) activeName = "XLSR";
        else notify("CNN 超分不可用，已回退 EASU");
      }

      if (!upscaled) {
        if (useRavu) {
          activeName = "RAVU";
          this.drawPass(encoder, this.pipelines.ravu, [
            { binding: 0, resource: this.intermediates.enhanced.createView() },
            { binding: 1, resource: this.sampler },
            { binding: 2, resource: this.ravuLut.createView() },
          ], this.intermediates.upscaled.createView());
        } else {
          this.drawPass(encoder, this.pipelines.easu, [
            { binding: 0, resource: this.intermediates.enhanced.createView() },
            { binding: 1, resource: this.sampler },
          ], this.intermediates.upscaled.createView());
        }
      }
      // 放大器变化时刷新角标文案（badgeString 会读 activeUpscaler）
      if (activeName !== this.activeUpscaler) {
        this.activeUpscaler = activeName;
        this.refreshBadge();
      }

      /* 后续 pass 数量随设置变化（grain/角标可开可关），用乒乓纹理串联，
       * 并让最后一个 pass 直接写 canvas 以省一次全屏绘制。
       * sharpened/pong 两张纹理交替作为读写目标 —— 同一张纹理不能同时
       * 既读又写。 */
      const withGrain = settings.grain > 0;
      const withBadge = !!this.badgeTexture;
      const canvasView = this.ctx.getCurrentTexture().createView();
      const pool = [this.intermediates.sharpened, this.intermediates.pong];
      let poolIdx = 0;
      // 剩余 pass 数：RCAS 恒有，grain/badge 可选
      let remaining = 1 + (withGrain ? 1 : 0) + (withBadge ? 1 : 0);
      const nextTarget = () => {
        remaining--;
        if (remaining === 0) return { view: canvasView, tex: null };
        const tex = pool[poolIdx % 2];
        poolIdx++;
        return { view: tex.createView(), tex };
      };

      // Pass 3: RCAS 锐化
      let cur = nextTarget();
      this.drawPass(encoder, this.pipelines.rcas, [
        { binding: 0, resource: this.intermediates.upscaled.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.rcasUniform } },
      ], cur.view);
      let prev = cur.tex;

      // Pass 4: 胶片颗粒 —— 放在锐化之后，否则 RCAS 会把颗粒当细节放大
      if (withGrain) {
        cur = nextTarget();
        this.drawPass(encoder, this.pipelines.grain, [
          { binding: 0, resource: prev.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.grainUniform } },
        ], cur.view);
        prev = cur.tex;
      }

      // Pass 5: 角标合成（始终最后，避免被后续 pass 处理）
      if (withBadge) {
        cur = nextTarget();
        this.drawPass(encoder, this.pipelines.badge, [
          { binding: 0, resource: prev.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: this.badgeTexture.createView() },
          { binding: 3, resource: { buffer: this.badgeUniform } },
        ], cur.view);
      }

      this.device.queue.submit([encoder.finish()]);

      this.consecutiveFailures = 0;
      this.canvas.classList.add("vidsharp-visible");
      this.firstFrameDone = true;
      this.perf.record(performance.now() - tStart, rvfcMeta);
      this.maybeUpdatePerfBadge();
    } catch (err) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= 5) {
        console.warn(
          "[VidSharp] 连续取帧失败，可能是 DRM 保护的视频，已停止处理。", err,
        );
        this.stop();   // 置 running=false，finally 里的 scheduleNext 会自行跳过
        notify("此视频受 DRM 保护，无法处理");
      }
    } finally {
      // 必须在所有路径上清除，否则一次异常会让渲染永久卡住
      this.rendering = false;
      // 也在 finally 里续帧：try 内的早退 return 会跳过函数尾部
      this.scheduleNext();
    }
  }

  stop() {
    this.running = false;
    if (this.rvfcHandle != null && this.video.cancelVideoFrameCallback) {
      this.video.cancelVideoFrameCallback(this.rvfcHandle);
      this.rvfcHandle = null;
    }
    if (this.onSeeking) {
      this.video.removeEventListener("seeking", this.onSeeking);
      this.onSeeking = null;
    }
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.teardownCompare();
    this.releaseBadgeTexture();
    // CNN 实例按会话持有（session 与 buffer 都不可共享），需异步释放
    this.cnn?.dispose();
    this.cnn = null;
    this.presr?.stop();
    this.presr = null;
    this.presrTex?.destroy();
    this.presrTex = null;
    this.presrTexSize = null;
    this.firstFrameDone = false;
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
    this.intermediates.enhanced?.destroy();
    this.intermediates.upscaled?.destroy();
    this.intermediates.sharpened?.destroy();
    this.intermediates.pong?.destroy();
    this.intermediates.cnnInput?.destroy();
    this.cnnInputSize = null;
    this.intermediates = { enhanced: null, upscaled: null, sharpened: null, pong: null, cnnInput: null };
    this.enhanceUniform?.destroy?.();
    this.rcasUniform?.destroy?.();
    this.badgeUniform?.destroy?.();
    this.grainUniform?.destroy?.();
    this.downscaleUniform?.destroy?.();
    this.enhanceUniform = null;
    this.rcasUniform = null;
    this.badgeUniform = null;
    this.grainUniform = null;
    this.downscaleUniform = null;
    if (this.patchedParent) {
      this.patchedParent.style.position = "";
      this.patchedParent = null;
    }
    active.delete(this.video);
  }
}

/* ---------- 视频发现与开关 ---------- */

function isProcessable(video) {
  return (
    video instanceof HTMLVideoElement &&
    video.videoWidth > 0 &&
    video.videoHeight > 0
  );
}

async function enableFor(video) {
  if (active.has(video)) return;
  const session = new Session(video);
  active.set(video, session);
  try {
    await session.start();
  } catch (err) {
    console.warn("[VidSharp] 启动失败:", err);
    session.stop();
    notify(err.message || "启动失败");
  }
}

function applyToAll() {
  for (const video of document.querySelectorAll("video")) {
    if (settings.enabled && isProcessable(video)) {
      enableFor(video);
    } else if (!settings.enabled) {
      active.get(video)?.stop();
    }
  }
}

function notify(message) {
  const el = document.createElement("div");
  el.className = "vidsharp-toast";
  el.textContent = `VidSharp: ${message}`;
  document.body?.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

/* ---------- 生命周期 ---------- */

const observer = new MutationObserver(() => {
  if (settings.enabled) applyToAll();
});

function init() {
  chrome.storage.sync.get(DEFAULTS, (stored) => {
    settings = { ...DEFAULTS, ...stored };
    if (settings.enabled) applyToAll();
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  document.addEventListener("loadedmetadata", (e) => {
    if (settings.enabled && e.target instanceof HTMLVideoElement) {
      enableFor(e.target);
    }
  }, true);

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;

    let reapply = false;
    let rebuild = false;
    let resize = false;
    let compare = false;
    let badgeMode = false;
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (!(key in DEFAULTS)) continue;
      settings[key] = newValue;
      if (key === "enabled") reapply = true;
      if (key === "upscale") resize = true;
      // 切换 CNN 档意味着 device 要换（ORT 的 device 不能与自建的混用），
      // 所有 pipeline/纹理/canvas context 都失效，只能整个会话重建
      if (key === "upscaler") {
        const toCnn = newValue === "xlsr";
        if (toCnn !== deviceIsOrt) rebuild = true;
      }
      if (key === "compare") compare = true;
      if (key === "badge") badgeMode = true;
      // 预超分依赖 MAIN world 捕获脚本的注册状态，切换后需重建会话
      if (key === "preSuperRes" || key === "preSuperResMode"
          || key === "preSuperResBudget") rebuild = true;
    }

    for (const video of document.querySelectorAll("video")) {
      const session = active.get(video);
      if (!session) continue;
      session.updateUniforms();
      if (resize) session.syncSizes();
      if (compare) session.refreshCompare();
      // 角标模式切换要重建元素；其余参数变化只需刷新文案
      if (badgeMode) session.refreshBadge();
      else session.updateBadge();
    }
    if (rebuild) {
      // 先全部停掉再重建，确保旧 device 上的资源都已释放
      for (const video of document.querySelectorAll("video")) {
        active.get(video)?.stop();
      }
      ortCtx = null;
      if (settings.enabled) applyToAll();
    } else if (reapply) {
      applyToAll();
    }
  });
}

init();
