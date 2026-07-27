/* VidSharp — WebGPU RCAS 视频锐化
 *
 * 管线：<video> --importExternalTexture--> RCAS fragment shader --> <canvas>
 *
 * 关键约束（踩过的坑，改代码前先读）：
 *   - external texture 单帧有效：每帧必须重新 importExternalTexture，
 *     并且 bind group 也必须每帧重建（纹理对象变了）。
 *   - requestVideoFrameCallback 只在有新解码帧时触发，比 rAF 更省电，
 *     且天然与视频帧率对齐（30fps 视频不会白跑 60 次）。
 *   - DRM/EME 视频拿不到像素，importExternalTexture 会抛错或全黑，
 *     这里靠 consecutiveFailures 检测并自动退出，不硬扛。
 */

const SHADER_URL = chrome.runtime.getURL("src/rcas.wgsl");

const DEFAULTS = {
  enabled: false,
  // 用户面板上的 0~100，线性映射到 RCAS sharpness
  strength: 50,
  denoise: true,
  // 诊断用：只锐化左半边，便于左右对比确认生效
  splitPreview: false,
};

let settings = { ...DEFAULTS };
const active = new WeakMap(); // video -> Session

/* ---------- WebGPU 设备（全页面共享一个） ---------- */

let devicePromise = null;

function getDevice() {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!navigator.gpu) throw new Error("此浏览器不支持 WebGPU");
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("无法获取 GPU adapter");
      const device = await adapter.requestDevice();
      // 设备意外丢失（驱动重置、标签页休眠）时清空缓存，下次重新申请
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
async function getShaderCode() {
  if (!shaderCache) {
    shaderCache = fetch(SHADER_URL).then((r) => r.text());
  }
  return shaderCache;
}

/* ---------- 单个视频的处理会话 ---------- */

class Session {
  constructor(video) {
    this.video = video;
    this.canvas = null;
    this.ctx = null;
    this.device = null;
    this.pipeline = null;
    this.sampler = null;
    this.uniformBuffer = null;
    this.rvfcHandle = null;
    this.running = false;
    this.consecutiveFailures = 0;
    this.lastSize = { w: 0, h: 0 };
    this.onResize = this.onResize.bind(this);
    this.renderFrame = this.renderFrame.bind(this);
  }

  async start() {
    if (this.running) return;

    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
      throw new Error("此浏览器不支持 requestVideoFrameCallback");
    }

    const device = await getDevice();
    const code = await getShaderCode();
    this.device = device;

    this.canvas = document.createElement("canvas");
    this.canvas.className = "vidsharp-overlay";
    this.ctx = this.canvas.getContext("webgpu");
    if (!this.ctx) throw new Error("无法创建 WebGPU canvas context");

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({
      device,
      format: this.format,
      alphaMode: "opaque",
    });

    const module = device.createShaderModule({ code, label: "rcas" });
    this.pipeline = device.createRenderPipeline({
      label: "rcas-pipeline",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    // 4 x f32：sharpness, denoise, 2 x padding
    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.updateUniforms();

    this.attachOverlay();
    this.running = true;
    this.scheduleNext();
  }

  /* 选择挂载点：向上找到仍与 video 尺寸基本一致的最外层祖先。
   *
   * 为什么不能直接用 parentElement：B 站等站点的弹幕层、字幕层往往是
   * video 父元素的兄弟节点，此时无论 canvas 的 z-index 多高都会被盖住
   * （z-index 只在同一层叠上下文内可比）。挂到共同祖先才能真正盖在最上面。
   *
   * 尺寸偏差阈值取 2px，容忍边框和亚像素布局；一旦祖先明显变大就停止，
   * 避免挂到整个页面上导致覆盖层错位。
   */
  pickMountPoint() {
    const videoRect = this.video.getBoundingClientRect();
    let best = this.video.parentElement;
    let node = best;

    for (let depth = 0; node && depth < 4; depth++) {
      const rect = node.getBoundingClientRect();
      const fits =
        Math.abs(rect.width - videoRect.width) <= 2 &&
        Math.abs(rect.height - videoRect.height) <= 2;
      if (!fits) break;
      best = node;
      node = node.parentElement;
    }

    return best;
  }

  attachOverlay() {
    const parent = this.pickMountPoint();
    if (!parent) throw new Error("video 没有父元素，无法挂载");

    // 覆盖层需要一个定位上下文；static 时补成 relative
    if (getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
      this.patchedParentPosition = parent;
    }
    parent.appendChild(this.canvas);

    if (settings.splitPreview) {
      this.canvas.classList.add("vidsharp-split");
    }

    this.resizeObserver = new ResizeObserver(this.onResize);
    this.resizeObserver.observe(this.video);
    this.onResize();
  }

  applySplitPreview() {
    this.canvas?.classList.toggle("vidsharp-split", !!settings.splitPreview);
  }

  onResize() {
    if (!this.canvas) return;
    // 后备分辨率跟随视频原始分辨率，而非 CSS 尺寸 —— 锐化要在源分辨率上做才准确
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (!w || !h) return;
    if (w !== this.lastSize.w || h !== this.lastSize.h) {
      const max = this.device.limits.maxTextureDimension2D;
      this.canvas.width = Math.min(w, max);
      this.canvas.height = Math.min(h, max);
      this.lastSize = { w, h };
    }
  }

  updateUniforms() {
    if (!this.uniformBuffer) return;
    // 面板 0~100 直接线性映射到 RCAS 的 sharpness 系数 0~1。
    // 不用 FSR 原版的 exp2(-stops) 曲线：那条曲线为渲染管线的 1~2 stops
    // 微调设计，映射到 0~100 后整个区间会被压在几乎不可见的弱效果段。
    const sharpness = Math.max(0, Math.min(1, settings.strength / 100));
    const data = new Float32Array([
      sharpness,
      settings.denoise ? 1.0 : 0.0,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  scheduleNext() {
    if (!this.running) return;
    this.rvfcHandle = this.video.requestVideoFrameCallback(this.renderFrame);
  }

  renderFrame() {
    if (!this.running) return;

    try {
      this.onResize();
      if (!this.canvas.width || !this.canvas.height) {
        this.scheduleNext();
        return;
      }

      // 每帧都要重新导入：external texture 只对当前帧有效
      const externalTexture = this.device.importExternalTexture({
        source: this.video,
      });

      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: externalTexture },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.ctx.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      this.consecutiveFailures = 0;
      this.canvas.classList.add("vidsharp-visible");
    } catch (err) {
      this.consecutiveFailures++;
      // DRM 视频会稳定失败，重试无意义，直接放弃并提示
      if (this.consecutiveFailures >= 5) {
        console.warn(
          "[VidSharp] 连续取帧失败，可能是 DRM 保护的视频，已停止处理。",
          err,
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
    this.canvas?.remove();
    this.canvas = null;
    this.ctx = null;
    this.uniformBuffer?.destroy?.();
    this.uniformBuffer = null;
    if (this.patchedParentPosition) {
      this.patchedParentPosition.style.position = "";
      this.patchedParentPosition = null;
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

function disableFor(video) {
  active.get(video)?.stop();
}

function applyToAll() {
  const videos = document.querySelectorAll("video");
  for (const video of videos) {
    if (settings.enabled && isProcessable(video)) {
      enableFor(video);
    } else if (!settings.enabled) {
      disableFor(video);
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

// 新插入的 video（SPA 站点很常见）
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

  // 视频元数据加载完才知道 videoWidth，此时才能启动
  document.addEventListener(
    "loadedmetadata",
    (e) => {
      if (settings.enabled && e.target instanceof HTMLVideoElement) {
        enableFor(e.target);
      }
    },
    true,
  );

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    let needsReapply = false;
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (key in DEFAULTS) {
        settings[key] = newValue;
        if (key === "enabled") needsReapply = true;
      }
    }
    for (const video of document.querySelectorAll("video")) {
      const session = active.get(video);
      if (!session) continue;
      session.updateUniforms();
      session.applySplitPreview();
    }
    if (needsReapply) applyToAll();
  });
}

init();
