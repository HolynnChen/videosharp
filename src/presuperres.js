/* 预超分协调器
 *
 * 把四个模块串成可用功能：
 *   capture-main.js  截 MSE 分片（MAIN world）
 *     → decode.js         解析 + 解码
 *     → superres-encode.js 超分 + 重编码
 *     → frame-store.js     存储 + 按时间检索
 *     → 渲染时优先取超分帧，取不到回落实时管线
 *
 * 设计取舍：
 *   - 音频与时间轴完全交给原播放器。我们只按它的 currentTime 取帧，
 *     所以 seek / 倍速 / 暂停 / 弹幕对齐全部免费沿用，规避了自建播放器
 *     最大的坑（音视频同步）。
 *   - 任何环节失败都静默回落实时管线 —— 用户至少还有实时超分可用。
 *
 * 硬前提：**必须有硬件编码器**。实测软件编码 2K 约 2s/帧，比播放慢 60 倍，
 * 预超分完全不成立。因此启用前用 canEncodeInRealtime() 自检，不达标直接
 * 不启用并说明原因。
 */

const CAPTURE_ID = "vidsharp-capture";

/** 与 capture-main.js 约定的消息标记 */
const MSG_FLAG = "__vidsharpCapture";

export class PreSuperRes {
  /**
   * @param device WebGPU device（须与实时管线同一个，纹理才能互通）
   * @param easuCode easu.wgsl 源码
   */
  constructor(device, easuCode) {
    this.device = device;
    this.easuCode = easuCode;
    this.decoder = null;
    this.encoder = null;
    this.store = null;
    this.player = null;
    this.outSize = { w: 0, h: 0 };
    this.enabled = false;
    this.failed = false;
    this.reason = "";
    this.stats = { segments: 0, decoded: 0, encoded: 0 };
    this.mode = "encode";
    this.cache = null;      // raw 模式
    this.rawProc = null;    // raw 模式
    this.onMessage = this.onMessage.bind(this);
  }

  /**
   * 启用。
   * @param mode "encode"（省显存，需硬件编码器）| "raw"（吃显存，零编解码开销）
   * @param budgetMB raw 模式的显存预算
   */
  async start(outW, outH, mode = "encode", budgetMB = 512) {
    if (this.failed) return false;
    this.mode = mode;
    try {
      const { TrackDecoder } = await import(chrome.runtime.getURL("src/decode.js"));
      this.outSize = { w: outW, h: outH };

      if (mode === "raw") {
        /* 无编码模式：直接缓存 VideoFrame。没有编码器速度这道坎，但显存
         * 决定了能领先多久 —— 按预算反算容量，而不是写死秒数。 */
        const { FrameCache, SuperResCache } =
          await import(chrome.runtime.getURL("src/frame-cache.js"));
        this.cache = new FrameCache(budgetMB);
        const cap = this.cache.capacityFor(outW, outH);
        this.rawProc = await new SuperResCache(
          this.device, this.easuCode, this.cache,
        ).init(outW, outH);
        console.log(
          `[VidSharp/presr] 无编码模式：预算 ${budgetMB}MB，` +
          `${outW}×${outH} 每帧约 ${(this.cache.perFrameBytes / 1024).toFixed(0)}KB，` +
          `可缓存约 ${cap} 帧（30fps 下 ${(cap / 30).toFixed(1)} 秒）`,
        );
      } else {
        const { canEncodeInRealtime, SuperResEncoder } =
          await import(chrome.runtime.getURL("src/superres-encode.js"));

        // 关键自检：软件编码下方案不成立，必须提前判断而非跑起来再卡
        const rt = await canEncodeInRealtime(outW, outH);
        if (!rt.ok) {
          this.failed = true;
          this.reason = rt.reason + "（可改用无编码模式）";
          console.warn("[VidSharp/presr] 不启用预超分:", this.reason);
          return false;
        }
        console.log(
          `[VidSharp/presr] 编码模式：${rt.codec.name} ` +
          `${rt.msPerFrame.toFixed(1)}ms/帧，处理速度为播放的 ${rt.ratio.toFixed(1)}x`,
        );

        const { FrameStore, FramePlayer } =
          await import(chrome.runtime.getURL("src/frame-store.js"));
        this.store = new FrameStore();
        this.player = new FramePlayer(this.store);

        this.encoder = await new SuperResEncoder(this.device, this.easuCode)
          .init(outW, outH, 30);
        this.encoder.onChunk = (chunk, meta) => {
          this.store.add(chunk, meta);
          this.stats.encoded++;
        };
      }

      this.decoder = await new TrackDecoder(
        (frame) => {
          this.stats.decoded++;
          /* 立刻处理后释放源帧 —— VideoFrame 很占显存，积压会让解码器
           * 因在途帧过多而 stall。注意 raw 模式下 process 内部会另建一个
           * 输出帧存进缓存，源帧仍需在此关闭。 */
          try {
            if (this.mode === "raw") this.rawProc?.process(frame);
            else this.encoder?.process(frame);
          } finally {
            frame.close();
          }
        },
        (err) => {
          console.warn("[VidSharp/presr] 解码失败，停用预超分:", err);
          this.stop();
        },
      ).init();

      window.addEventListener("message", this.onMessage);
      this.enabled = true;
      return true;
    } catch (err) {
      this.failed = true;
      this.reason = err.message;
      console.warn("[VidSharp/presr] 启用失败:", err);
      return false;
    }
  }

  /** 接收 MAIN world 捕获的分片 */
  onMessage(ev) {
    const d = ev.data;
    if (!d || d[MSG_FLAG] !== true || d.kind !== "segment") return;
    if (!this.enabled || this.failed) return;
    try {
      this.stats.segments++;
      this.decoder?.pushSegment(d.payload.buffer);
    } catch (err) {
      console.warn("[VidSharp/presr] 处理分片失败:", err);
    }
  }

  /**
   * 取当前播放时间对应的超分帧。
   * @param currentTimeS 原播放器的 currentTime（秒）
   * @returns {VideoFrame|null} null 表示未命中，调用方应回落实时管线
   */
  getFrame(currentTimeS) {
    if (!this.enabled || this.failed) return null;
    const us = Math.round(currentTimeS * 1e6);

    if (this.mode === "raw") {
      const frame = this.cache?.get(us) ?? null;
      // raw 模式必须勤于淘汰 —— 每帧几 MB，攒着很快就爆预算
      this.cache?.prune(us);
      return frame;
    }

    const frame = this.player?.get(us) ?? null;
    if (this.stats.encoded % 30 === 0) this.store?.prune(us);
    return frame;
  }

  /** seek 后已解码的帧全部失效 */
  onSeek() {
    this.player?.reset();
    // raw 模式下缓存的帧时间戳已失效，全部丢弃
    this.cache?.clear();
  }

  /** @param currentTimeS 用于算领先量；不传则只报缓存量 */
  info(currentTimeS = 0) {
    if (this.failed) return `预超分不可用: ${this.reason}`;
    if (!this.enabled) return "预超分未启用";

    const us = Math.round(currentTimeS * 1e6);
    if (this.mode === "raw") {
      const st = this.cache?.stats ?? { frames: 0, mb: 0, budgetMb: 0 };
      const ahead = this.cache?.aheadSeconds(us) ?? 0;
      return (
        `预超分(无编码): 领先 ${ahead.toFixed(1)}s / ` +
        `${st.frames} 帧 / ${st.mb.toFixed(0)}·${st.budgetMb.toFixed(0)}MB`
      );
    }
    const span = this.store?.span ?? { from: 0, to: 0 };
    const ahead = Math.max(0, (span.to - us) / 1e6);
    return (
      `预超分(编码): 领先 ${ahead.toFixed(1)}s / ` +
      `${this.stats.encoded} 帧 / ` +
      `${((this.store?.totalBytes ?? 0) / 1024 / 1024).toFixed(1)}MB`
    );
  }

  async stop() {
    this.enabled = false;
    window.removeEventListener("message", this.onMessage);
    this.player?.close();
    this.player = null;
    await this.encoder?.close();
    this.encoder = null;
    await this.decoder?.close();
    this.decoder = null;
    this.store?.clear();
    this.store = null;
    this.rawProc?.close();
    this.rawProc = null;
    this.cache?.clear();
    this.cache = null;
  }
}

/** 注册/注销 MAIN world 的分片捕获脚本（需 service worker 配合） */
export const CAPTURE_SCRIPT_ID = CAPTURE_ID;
