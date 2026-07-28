/* 离线超分 + 重编码
 *
 * 预超分的核心：把解码出的 VideoFrame 超分后重新编码存起来，播放时直接
 * 解压取用。这样超分不再受每帧 16~33ms 的实时约束。
 *
 * 为什么必须重编码：超分后的未压缩像素存不下 —— 2K 每帧 14MB，2 秒就
 * 0.82GB，浏览器单页显存上限通常几百 MB~2GB。编码后 2 分钟只要几十 MB。
 *
 * 两处 API 细节（实测确认，写错只在运行时暴露）：
 *   - VideoFrame → GPU：必须用 copyExternalImageToTexture。
 *     importExternalTexture 只接受 HTMLVideoElement，不接受 VideoFrame。
 *   - GPU → VideoFrame：WebGPU 不能直接产出 VideoFrame。桥是把纹理渲到
 *     配了 webgpu context 的 OffscreenCanvas，再 new VideoFrame(canvas)。
 *     全程留在 GPU 上，不回 CPU。
 *
 * 性能前提：**必须有硬件编码器**。实测软件编码 2K 约 2s/帧（比播放慢
 * 60 倍，方案完全不成立）；硬件编码在个位数 ms，才能领先播放头。
 * canEncodeInRealtime() 用于运行前自检。
 */

/** 候选编码格式，按「硬件支持普遍性 × 压缩率」排序 */
const CODEC_CANDIDATES = [
  { name: "H.264", codec: "avc1.640033" },   // 硬件支持最普遍
  { name: "VP9", codec: "vp09.00.10.08" },
  { name: "AV1", codec: "av01.0.08M.08" },
];

/**
 * 选一个本机可用的编码配置。
 * @returns {Promise<{name,codec,accel}|null>}
 */
export async function pickCodec(width, height, bitrate) {
  if (typeof VideoEncoder === "undefined") return null;
  for (const c of CODEC_CANDIDATES) {
    const res = await VideoEncoder.isConfigSupported({
      codec: c.codec, width, height, bitrate, framerate: 30,
    }).catch(() => ({ supported: false }));
    if (res.supported) {
      return {
        ...c,
        accel: res.config?.hardwareAcceleration ?? "unknown",
      };
    }
  }
  return null;
}

/**
 * 实测编码吞吐，判断能否领先播放头。
 *
 * 不能只看 isConfigSupported —— 它对软件编码也返回 supported，而软件编码
 * 慢到方案不成立。必须真编几帧看耗时。
 *
 * @returns {Promise<{ok, msPerFrame, ratio, codec, reason}>}
 */
export async function canEncodeInRealtime(width, height, fps = 30) {
  const bitrate = Math.round(width * height * 0.15);   // 约 0.15 bit/px
  const picked = await pickCodec(width, height, bitrate);
  if (!picked) {
    return { ok: false, reason: "本机无可用视频编码器" };
  }

  // 用纯色小画布试编，避免测试本身太慢
  const osc = new OffscreenCanvas(width, height);
  const ctx = osc.getContext("2d");
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, width, height);

  let count = 0;
  const encoder = new VideoEncoder({
    output: () => { count++; },
    error: () => {},
  });
  try {
    encoder.configure({
      codec: picked.codec, width, height, bitrate, framerate: fps,
      latencyMode: "quality",
    });

    const N = 8;
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      const f = new VideoFrame(osc, {
        timestamp: (i * 1e6) / fps, duration: 1e6 / fps,
      });
      encoder.encode(f, { keyFrame: i === 0 });
      f.close();
    }
    await encoder.flush();
    const msPerFrame = (performance.now() - t0) / N;

    // 处理速度 / 播放速度。留 1ms 给超分本身（EASU 实测约 0.94ms）
    const ratio = (1000 / fps) / (msPerFrame + 1.0);
    return {
      ok: ratio >= 1.2,       // 至少 1.2x 才有意义，否则攒不出余量
      msPerFrame,
      ratio,
      codec: picked,
      reason: ratio >= 1.2
        ? ""
        : `编码 ${msPerFrame.toFixed(0)}ms/帧，处理速度仅为播放的 ` +
          `${ratio.toFixed(2)}x（需硬件编码器，当前 accel=${picked.accel}）`,
    };
  } catch (err) {
    return { ok: false, reason: "编码器自检失败: " + err.message };
  } finally {
    try { if (encoder.state !== "closed") encoder.close(); } catch { /* 忽略 */ }
  }
}

/**
 * 超分 + 编码流水线。
 *
 * 用法：
 *   const p = await new SuperResEncoder(device, shaders).init(outW, outH);
 *   p.onChunk = (chunk, meta) => { ...存起来... };
 *   await p.process(videoFrame);   // 逐帧喂入
 *   await p.finish();
 */
export class SuperResEncoder {
  constructor(device, easuCode) {
    this.device = device;
    this.easuCode = easuCode;
    this.pipeline = null;
    this.blitPipeline = null;
    this.sampler = null;
    this.srcTex = null;
    this.srcSize = { w: 0, h: 0 };
    this.outTex = null;
    this.canvas = null;
    this.canvasCtx = null;
    this.encoder = null;
    this.outSize = { w: 0, h: 0 };
    this.frameIndex = 0;
    this.onChunk = null;
    this.failed = false;
  }

  async init(outW, outH, fps = 30) {
    const dev = this.device;
    this.outSize = { w: outW, h: outH };

    const bitrate = Math.round(outW * outH * 0.15);
    const picked = await pickCodec(outW, outH, bitrate);
    if (!picked) throw new Error("本机无可用视频编码器");
    this.codecInfo = picked;

    const interFormat = "rgba8unorm";
    const module = dev.createShaderModule({ code: this.easuCode, label: "sr-easu" });
    this.pipeline = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: interFormat }] },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = dev.createSampler({
      magFilter: "linear", minFilter: "linear",
      addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
    });

    this.outTex = dev.createTexture({
      label: "sr-out",
      size: [outW, outH], format: interFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // GPU → VideoFrame 的桥：webgpu context 的 OffscreenCanvas
    this.canvas = new OffscreenCanvas(outW, outH);
    this.canvasCtx = this.canvas.getContext("webgpu");
    if (!this.canvasCtx) throw new Error("OffscreenCanvas 不支持 webgpu context");
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.canvasCtx.configure({
      device: dev, format: this.canvasFormat, alphaMode: "opaque",
    });

    // blit：把处理结果拷到 canvas（格式可能不同，需要一次采样）
    const blitMod = dev.createShaderModule({
      label: "sr-blit",
      code: `
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        struct VO { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
        @vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
          var o: VO;
          let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
          let y = f32(i & 2u) * 2.0 - 1.0;
          o.p = vec4<f32>(x, y, 0.0, 1.0);
          o.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
          return o;
        }
        @fragment fn fs(v: VO) -> @location(0) vec4<f32> {
          return textureSampleLevel(t, s, v.uv, 0.0);
        }`,
    });
    this.blitPipeline = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: blitMod, entryPoint: "vs" },
      fragment: { module: blitMod, entryPoint: "fs",
                  targets: [{ format: this.canvasFormat }] },
      primitive: { topology: "triangle-list" },
    });

    this.encoder = new VideoEncoder({
      output: (chunk, meta) => {
        this.onChunk?.(chunk, meta);
      },
      error: (err) => {
        this.failed = true;
        console.warn("[VidSharp/sr-encode] 编码器错误:", err);
      },
    });
    this.encoder.configure({
      codec: picked.codec, width: outW, height: outH,
      bitrate, framerate: fps,
      // 存储用途，画质优先于延迟
      latencyMode: "quality",
    });

    return this;
  }

  /** 源尺寸变化时重建输入纹理（清晰度切换会触发） */
  ensureSrcTexture(w, h) {
    if (this.srcTex && this.srcSize.w === w && this.srcSize.h === h) return;
    this.srcTex?.destroy();
    this.srcTex = this.device.createTexture({
      label: "sr-src",
      size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.srcSize = { w, h };
  }

  /**
   * 处理一帧。调用方保留 frame 所有权（本方法不 close）。
   * @returns {boolean} 是否成功
   */
  process(frame) {
    if (this.failed || !this.encoder) return false;
    try {
      const w = frame.displayWidth;
      const h = frame.displayHeight;
      this.ensureSrcTexture(w, h);

      // VideoFrame → GPU（importExternalTexture 不接受 VideoFrame）
      this.device.queue.copyExternalImageToTexture(
        { source: frame }, { texture: this.srcTex }, [w, h],
      );

      const dev = this.device;
      const enc = dev.createCommandEncoder({ label: "sr-frame" });

      // 超分
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: this.outTex.createView(), loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store",
        }],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, dev.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.srcTex.createView() },
          { binding: 1, resource: this.sampler },
        ],
      }));
      pass.draw(3);
      pass.end();

      // blit 到 canvas，供 VideoFrame 取用
      const p2 = enc.beginRenderPass({
        colorAttachments: [{
          view: this.canvasCtx.getCurrentTexture().createView(),
          loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        }],
      });
      p2.setPipeline(this.blitPipeline);
      p2.setBindGroup(0, dev.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.outTex.createView() },
          { binding: 1, resource: this.sampler },
        ],
      }));
      p2.draw(3);
      p2.end();
      dev.queue.submit([enc.finish()]);

      // 编码。时间戳沿用源帧，播放时才能按时间检索
      const outFrame = new VideoFrame(this.canvas, {
        timestamp: frame.timestamp,
        duration: frame.duration ?? undefined,
      });
      // 每 60 帧一个关键帧，便于 seek 时快速定位
      this.encoder.encode(outFrame, { keyFrame: this.frameIndex % 60 === 0 });
      outFrame.close();
      this.frameIndex++;
      return true;
    } catch (err) {
      this.failed = true;
      console.warn("[VidSharp/sr-encode] 处理帧失败:", err);
      return false;
    }
  }

  async finish() {
    if (this.encoder?.state === "configured") {
      try { await this.encoder.flush(); } catch { /* 忽略 */ }
    }
  }

  async close() {
    this.failed = true;
    await this.finish();
    try {
      if (this.encoder && this.encoder.state !== "closed") this.encoder.close();
    } catch { /* 忽略 */ }
    this.encoder = null;
    this.srcTex?.destroy();
    this.outTex?.destroy();
    this.srcTex = null;
    this.outTex = null;
    this.canvas = null;
    this.canvasCtx = null;
  }
}
