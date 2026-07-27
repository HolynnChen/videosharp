/* VidSharp — WebGPU 视频增强
 *
 * 三 pass 管线（顺序有讲究）：
 *
 *   <video> --importExternalTexture-->
 *     [1] enhance  去块 + 去色带 + 局部对比度   (源分辨率)
 *     [2] easu     方向自适应放大               (→ 目标分辨率)
 *     [3] rcas     自适应锐化                   (目标分辨率)
 *   --> <canvas>
 *
 * 为什么是这个顺序：
 *   - 去块必须在放大前：否则 EASU 会把块边界当真实边缘"保护"，块效应被锐化。
 *   - 锐化必须在放大后：放大前锐化会被插值重新糊掉。
 *
 * 关键约束（改代码前先读）：
 *   - external texture 单帧有效：每帧重新 import，bind group 也必须重建。
 *   - 覆盖层不能压过弹幕/控制条：插到 video 的直接后继位置，继承 video 的
 *     层级，而不是把 z-index 顶到最大（那样会盖住播放器 UI）。
 *   - DRM 视频取不到像素，靠 consecutiveFailures 检测后自动退出。
 */

const SHADERS = {
  enhance: chrome.runtime.getURL("src/enhance.wgsl"),
  easu: chrome.runtime.getURL("src/easu.wgsl"),
  rcas: chrome.runtime.getURL("src/rcas.wgsl"),
};

const DEFAULTS = {
  enabled: false,
  strength: 50,      // RCAS 锐化 0~100
  denoise: true,     // RCAS 噪声抑制
  deblock: 40,       // 去块 0~100
  deband: 30,        // 去色带 0~100
  contrast: 25,      // 局部对比度 0~100
  upscale: "2k",     // off | 2k | 4k | 2x
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
    this.intermediates = { enhanced: null, upscaled: null };
    this.rvfcHandle = null;
    this.running = false;
    this.consecutiveFailures = 0;
    this.frameCount = 0;
    this.srcSize = { w: 0, h: 0 };
    this.outSize = { w: 0, h: 0 };
    this.divider = null;
    this.compareRatio = 0.5;
    this.badge = null;
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
    this.pipelines.rcas = this.buildPipeline(
      shaders.rcas, "rcas", this.format, true,
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
    this.updateUniforms();

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
    this.setupBadge();
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
    this.updateBadge();
  }

  rebuildIntermediates() {
    const { device, interFormat } = this;
    this.intermediates.enhanced?.destroy();
    this.intermediates.upscaled?.destroy();

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
  }

  /* ---- 增强状态角标 ----
   *
   * 存在的意义：画面被 canvas 接管后，肉眼无法确认看到的是原始视频还是
   * 增强结果。角标只在首帧真正渲染成功后才出现（见 renderFrame），
   * 因此它的出现本身就是"增强确实生效"的证据 —— 不是静态装饰。
   *
   * 做成独立 DOM 而非画进 canvas：不污染画面像素，截图/录屏时也不会带上。
   */

  setupBadge() {
    if (settings.badge === "off") return;
    const badge = document.createElement("div");
    badge.className = "vidsharp-badge";
    this.badge = badge;
    this.canvas.parentElement?.appendChild(badge);
    this.updateBadge();
  }

  updateBadge() {
    if (!this.badge) return;

    const { w: sw, h: sh } = this.srcSize;
    const { w: ow, h: oh } = this.outSize;
    const upscaled = ow > sw;

    if (settings.badge === "detail") {
      const parts = [];
      parts.push(upscaled ? `${sw}×${sh} → ${ow}×${oh}` : `${ow}×${oh}`);
      const fx = [];
      if (settings.strength > 0) fx.push(`锐化 ${settings.strength}`);
      if (settings.deblock > 0) fx.push(`去块 ${settings.deblock}`);
      if (settings.deband > 0) fx.push(`去色带 ${settings.deband}`);
      if (settings.contrast > 0) fx.push(`对比 ${settings.contrast}`);
      this.badge.textContent =
        `VidSharp · ${parts.join("")}${fx.length ? " · " + fx.join(" / ") : ""}`;
    } else {
      this.badge.textContent = upscaled ? `VidSharp ${oh}p` : "VidSharp";
    }
  }

  teardownBadge() {
    this.badge?.remove();
    this.badge = null;
  }

  refreshBadge() {
    this.teardownBadge();
    if (settings.badge !== "off" && this.canvas?.parentElement) {
      this.setupBadge();
      // 已经在渲染中的会话要立刻显示，不必等下一次首帧
      if (this.firstFrameDone) this.badge?.classList.add("vidsharp-visible");
    }
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
      // 去色带抖动需要每帧变化，否则固定噪点图案会被眼睛识别为脏点
      if (settings.deband > 0) this.updateUniforms();

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

      // Pass 2: EASU 放大
      this.drawPass(encoder, this.pipelines.easu, [
        { binding: 0, resource: this.intermediates.enhanced.createView() },
        { binding: 1, resource: this.sampler },
      ], this.intermediates.upscaled.createView());

      // Pass 3: RCAS 锐化 → canvas
      this.drawPass(encoder, this.pipelines.rcas, [
        { binding: 0, resource: this.intermediates.upscaled.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.rcasUniform } },
      ], this.ctx.getCurrentTexture().createView());

      this.device.queue.submit([encoder.finish()]);

      this.consecutiveFailures = 0;
      this.canvas.classList.add("vidsharp-visible");
      // 角标在首帧成功后才显示 —— 它的出现即证明增强真的生效
      if (!this.firstFrameDone) {
        this.firstFrameDone = true;
        this.badge?.classList.add("vidsharp-visible");
      }
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
    this.teardownBadge();
    this.firstFrameDone = false;
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
    this.intermediates.enhanced?.destroy();
    this.intermediates.upscaled?.destroy();
    this.intermediates = { enhanced: null, upscaled: null };
    this.enhanceUniform?.destroy?.();
    this.rcasUniform?.destroy?.();
    this.enhanceUniform = null;
    this.rcasUniform = null;
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
