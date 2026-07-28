/* 无编码预超分：直接缓存超分后的 VideoFrame
 *
 * 与编码模式的取舍：
 *   编码模式   省显存（2K 每帧约 50KB），但需硬件编码器；软件编码下 2K
 *              约 2s/帧，比播放慢 60 倍，完全不成立。
 *   无编码模式 零编解码开销、无二次画质损失，但吃显存。
 *
 * 所以无编码模式的关键不是"能不能做"，而是**能缓存多久**。这里按显存预算
 * 反算帧数而非写死秒数 —— 同一个预算在 720p 能存几十秒，在 4K 只够几秒，
 * 写死秒数必然在某一端出错。
 */

/** VideoFrame 的实际显存占用估算（字节）。
 *
 * 不按 RGBA 的 4 字节/像素算：VideoFrame 从 canvas 构造时浏览器通常转成
 * NV12（YUV 4:2:0），每像素 1.5 字节。取 2 字节留余量。
 */
function estimateFrameBytes(w, h) {
  return w * h * 2;
}

export class FrameCache {
  /** @param budgetMB 显存预算（MB）。超出即淘汰最旧的帧。 */
  constructor(budgetMB = 512) {
    this.budgetBytes = budgetMB * 1024 * 1024;
    /** @type {Array<{ts:number, frame:VideoFrame, bytes:number}>} */
    this.frames = [];
    this.totalBytes = 0;
    this.perFrameBytes = 0;
    this.evicted = 0;
  }

  /** 按当前帧尺寸算出预算内能存多少帧 */
  capacityFor(w, h) {
    this.perFrameBytes = estimateFrameBytes(w, h);
    return Math.max(1, Math.floor(this.budgetBytes / this.perFrameBytes));
  }

  /**
   * 存入一帧。**接管所有权** —— 调用方不要再 close。
   * @returns {boolean} 是否存入
   */
  add(frame) {
    const bytes = estimateFrameBytes(frame.displayWidth, frame.displayHeight);
    this.perFrameBytes = bytes;

    // 时间戳重复的直接替换，避免重复处理堆积
    const dup = this.frames.findIndex((f) => f.ts === frame.timestamp);
    if (dup >= 0) {
      this.totalBytes -= this.frames[dup].bytes;
      this.frames[dup].frame.close();
      this.frames.splice(dup, 1);
    }

    // 预算不够就淘汰最旧的 —— 它通常已经播过，丢掉无损体验
    while (this.totalBytes + bytes > this.budgetBytes && this.frames.length) {
      const old = this.frames.shift();
      this.totalBytes -= old.bytes;
      old.frame.close();
      this.evicted++;
    }
    if (this.totalBytes + bytes > this.budgetBytes) {
      // 单帧就超预算（预算设得过小），拒收而非清空
      frame.close();
      return false;
    }

    // 保持按时间戳有序，便于二分检索
    const idx = this.frames.findIndex((f) => f.ts > frame.timestamp);
    const entry = { ts: frame.timestamp, frame, bytes };
    if (idx < 0) this.frames.push(entry);
    else this.frames.splice(idx, 0, entry);

    this.totalBytes += bytes;
    return true;
  }

  /**
   * 取 timeUs 对应的帧。
   * @returns {VideoFrame|null} 不转移所有权，调用方不要 close
   */
  get(timeUs, toleranceUs = 16_700) {
    if (!this.frames.length) return null;

    let lo = 0, hi = this.frames.length - 1, found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.frames[mid].ts <= timeUs) { found = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    if (found < 0) return null;
    const e = this.frames[found];
    // 太旧的帧不能用 —— 说明缓存断档，用了画面会滞后
    if (timeUs - e.ts > toleranceUs + 40_000) return null;
    return e.frame;
  }

  /** 丢弃播放头之前的帧（已播过，不会再用） */
  prune(currentTimeUs, keepBehindUs = 500_000) {
    const cutoff = currentTimeUs - keepBehindUs;
    while (this.frames.length && this.frames[0].ts < cutoff) {
      const old = this.frames.shift();
      this.totalBytes -= old.bytes;
      old.frame.close();
    }
  }

  /** 领先播放头多少秒 —— 这是"预"超分的实际价值度量 */
  aheadSeconds(currentTimeUs) {
    if (!this.frames.length) return 0;
    const last = this.frames[this.frames.length - 1].ts;
    return Math.max(0, (last - currentTimeUs) / 1e6);
  }

  get stats() {
    return {
      frames: this.frames.length,
      mb: this.totalBytes / 1024 / 1024,
      budgetMb: this.budgetBytes / 1024 / 1024,
      evicted: this.evicted,
      perFrameKb: this.perFrameBytes / 1024,
    };
  }

  clear() {
    for (const e of this.frames) e.frame.close();
    this.frames.length = 0;
    this.totalBytes = 0;
  }
}

/**
 * 超分但不编码的处理器。
 *
 * 与 SuperResEncoder 的区别：末端不接 VideoEncoder，而是把 blit 结果直接
 * 构造成 VideoFrame 存进缓存，省掉编码+解码两道开销与二次画质损失。
 */
export class SuperResCache {
  constructor(device, easuCode, cache) {
    this.device = device;
    this.easuCode = easuCode;
    this.cache = cache;
    this.pipeline = null;
    this.blitPipeline = null;
    this.sampler = null;
    this.srcTex = null;
    this.srcSize = { w: 0, h: 0 };
    this.outTex = null;
    this.canvas = null;
    this.canvasCtx = null;
    this.outSize = { w: 0, h: 0 };
    this.processed = 0;
    this.rejected = 0;
    this.failed = false;
  }

  async init(outW, outH) {
    const dev = this.device;
    this.outSize = { w: outW, h: outH };
    const interFormat = "rgba8unorm";

    const module = dev.createShaderModule({
      code: this.easuCode, label: "srcache-easu",
    });
    this.pipeline = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main",
                  targets: [{ format: interFormat }] },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = dev.createSampler({
      magFilter: "linear", minFilter: "linear",
      addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
    });

    this.outTex = dev.createTexture({
      label: "srcache-out",
      size: [outW, outH], format: interFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // GPU → VideoFrame 的唯一桥梁
    this.canvas = new OffscreenCanvas(outW, outH);
    this.canvasCtx = this.canvas.getContext("webgpu");
    if (!this.canvasCtx) throw new Error("OffscreenCanvas 不支持 webgpu context");
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.canvasCtx.configure({
      device: dev, format: this.canvasFormat, alphaMode: "opaque",
    });

    const blitMod = dev.createShaderModule({
      label: "srcache-blit",
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

    return this;
  }

  ensureSrcTexture(w, h) {
    if (this.srcTex && this.srcSize.w === w && this.srcSize.h === h) return;
    this.srcTex?.destroy();
    this.srcTex = this.device.createTexture({
      label: "srcache-in",
      size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.srcSize = { w, h };
  }

  /**
   * 处理一帧并存入缓存。调用方保留 frame 所有权（本方法不 close）。
   * @returns {boolean}
   */
  process(frame) {
    if (this.failed) return false;
    try {
      const w = frame.displayWidth, h = frame.displayHeight;
      this.ensureSrcTexture(w, h);

      this.device.queue.copyExternalImageToTexture(
        { source: frame }, { texture: this.srcTex }, [w, h],
      );

      const dev = this.device;
      const enc = dev.createCommandEncoder({ label: "srcache-frame" });

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

      // 直接构造 VideoFrame 存缓存 —— 不编码，故无二次画质损失
      const outFrame = new VideoFrame(this.canvas, {
        timestamp: frame.timestamp,
        duration: frame.duration ?? undefined,
      });
      if (this.cache.add(outFrame)) this.processed++;
      else this.rejected++;   // add 内部已 close
      return true;
    } catch (err) {
      this.failed = true;
      console.warn("[VidSharp/srcache] 处理帧失败:", err);
      return false;
    }
  }

  close() {
    this.failed = true;
    this.srcTex?.destroy();
    this.outTex?.destroy();
    this.srcTex = null;
    this.outTex = null;
    this.canvas = null;
    this.canvasCtx = null;
  }
}
