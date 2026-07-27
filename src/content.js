/* VidSharp — WebGPU 视频增强
 *
 * 管线（顺序有讲究，pass 数随设置变化）：
 *
 *   <video> --importExternalTexture-->
 *     [1] enhance  去块 + 去色带 + 局部对比度   (源分辨率)
 *     [2] easu     方向自适应放大               (→ 目标分辨率)
 *     [3] rcas     自适应锐化
 *     [4] grain    合成胶片颗粒（可选）
 *     [5] badge    状态角标合成（可选）
 *   --> <canvas>
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
  upscaler: "easu",  // easu | ravu — 放大算法
  compare: false,    // 拖动对比模式
  badge: "corner",   // off | corner | detail — 增强状态标识
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

/* ---------- WebGPU 设备 ---------- */

let devicePromise = null;

function getDevice() {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!navigator.gpu) throw new Error("此浏览器不支持 WebGPU");
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("无法获取 GPU adapter");
      const device = await adapter.requestDevice();
      device.lost.then((info) => {
        console.warn("[VidSharp] GPU 设备丢失:", info.message);
        devicePromise = null;
        for (const video of document.querySelectorAll("video")) {
          active.get(video)?.stop();
        }
      });
      return device;
    })();
    devicePromise.catch(() => { devicePromise = null; });
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
function getRavuLut(device) {
  if (!ravuLutPromise) {
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
    this.intermediates = { enhanced: null, upscaled: null, sharpened: null, pong: null };
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
    this.ravuLut = null;   // 共享纹理，会话结束不销毁
    this.firstFrameDone = false;
    this.renderFrame = this.renderFrame.bind(this);
    this.syncSizes = this.syncSizes.bind(this);
  }

  async start() {
    if (this.running) return;
    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
      throw new Error("此浏览器不支持 requestVideoFrameCallback");
    }

    const [device, shaders] = await Promise.all([getDevice(), getShaders()]);
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

    this.attachOverlay();
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
      if (upscaled) {
        const scaleX = ow / Math.max(1, sw);
        const ravuActive = settings.upscaler === "ravu"
          && Math.abs(scaleX - 2) < 0.02;
        fx.push(ravuActive ? "RAVU" : "EASU");
      }
      if (settings.strength > 0) fx.push(`锐化 ${settings.strength}`);
      if (settings.deblock > 0) fx.push(`去块 ${settings.deblock}`);
      if (settings.deband > 0) fx.push(`去色带 ${settings.deband}`);
      if (settings.contrast > 0) fx.push(`对比 ${settings.contrast}`);
      return `VidSharp · ${size}${fx.length ? " · " + fx.join(" / ") : ""}`;
    }
    return upscaled ? `VidSharp ${oh}p` : "VidSharp";
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

  renderFrame() {
    if (!this.running) return;

    try {
      this.syncSizes();
      if (!this.canvas.width || !this.canvas.height) {
        this.scheduleNext();
        return;
      }

      this.frameCount++;
      // 去色带与颗粒都依赖帧号做时域去相关，图案必须每帧变化，
      // 否则固定噪点会被眼睛识别为屏幕脏污
      if (settings.deband > 0 || settings.grain > 0) this.updateUniforms();

      const external = this.device.importExternalTexture({
        source: this.video,
      });

      const encoder = this.device.createCommandEncoder();

      // Pass 1: enhance（源分辨率）
      this.drawPass(encoder, this.pipelines.enhance, [
        { binding: 0, resource: external },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.enhanceUniform } },
      ], this.intermediates.enhanced.createView());

      // Pass 2: 放大
      //
      // RAVU 固定 2x（LUT 按 2x 子像素布局训练），因此只在目标倍率接近 2
      // 时启用；其余倍率回落 EASU。这不是妥协 —— 强行用 RAVU 做非 2x 会
      // 让子像素索引错位，画面出现网格状伪影。
      const scaleX = this.outSize.w / Math.max(1, this.srcSize.w);
      const useRavu = settings.upscaler === "ravu"
        && this.ravuLut != null
        && Math.abs(scaleX - 2) < 0.02;

      if (useRavu) {
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
    } catch (err) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= 5) {
        console.warn(
          "[VidSharp] 连续取帧失败，可能是 DRM 保护的视频，已停止处理。", err,
        );
        this.stop();
        notify("此视频受 DRM 保护，无法处理");
        return;
      }
    }

    this.scheduleNext();
  }

  stop() {
    this.running = false;
    if (this.rvfcHandle != null && this.video.cancelVideoFrameCallback) {
      this.video.cancelVideoFrameCallback(this.rvfcHandle);
      this.rvfcHandle = null;
    }
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.teardownCompare();
    this.releaseBadgeTexture();
    this.firstFrameDone = false;
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
    this.intermediates.enhanced?.destroy();
    this.intermediates.upscaled?.destroy();
    this.intermediates.sharpened?.destroy();
    this.intermediates.pong?.destroy();
    this.intermediates = { enhanced: null, upscaled: null, sharpened: null, pong: null };
    this.enhanceUniform?.destroy?.();
    this.rcasUniform?.destroy?.();
    this.badgeUniform?.destroy?.();
    this.grainUniform?.destroy?.();
    this.enhanceUniform = null;
    this.rcasUniform = null;
    this.badgeUniform = null;
    this.grainUniform = null;
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
    let resize = false;
    let compare = false;
    let badgeMode = false;
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (!(key in DEFAULTS)) continue;
      settings[key] = newValue;
      if (key === "enabled") reapply = true;
      if (key === "upscale") resize = true;
      if (key === "compare") compare = true;
      if (key === "badge") badgeMode = true;
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
    if (reapply) applyToAll();
  });
}

init();
