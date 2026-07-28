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
    this.onMessage = this.onMessage.bind(this);
  }

  /**
   * 启用。会先做编码器自检 —— 不达标不启用。
   * @returns {Promise<boolean>}
   */
  async start(outW, outH) {
    if (this.failed) return false;
    try {
      const { canEncodeInRealtime, SuperResEncoder } =
        await import(chrome.runtime.getURL("src/superres-encode.js"));

      // 关键自检：软件编码下方案不成立，必须提前判断而非跑起来再卡
      const rt = await canEncodeInRealtime(outW, outH);
      if (!rt.ok) {
        this.failed = true;
        this.reason = rt.reason;
        console.warn("[VidSharp/presr] 不启用预超分:", rt.reason);
        return false;
      }
      console.log(
        `[VidSharp/presr] 编码器 ${rt.codec.name} ` +
        `${rt.msPerFrame.toFixed(1)}ms/帧，处理速度为播放的 ${rt.ratio.toFixed(1)}x`,
      );

      const { TrackDecoder } = await import(chrome.runtime.getURL("src/decode.js"));
      const { FrameStore, FramePlayer } =
        await import(chrome.runtime.getURL("src/frame-store.js"));

      this.outSize = { w: outW, h: outH };
      this.store = new FrameStore();
      this.player = new FramePlayer(this.store);

      this.encoder = await new SuperResEncoder(this.device, this.easuCode)
        .init(outW, outH, 30);
      this.encoder.onChunk = (chunk, meta) => {
        this.store.add(chunk, meta);
        this.stats.encoded++;
      };

      this.decoder = await new TrackDecoder(
        (frame) => {
          this.stats.decoded++;
          // 立刻超分编码后释放 —— VideoFrame 很占显存，不能积压
          try { this.encoder?.process(frame); } finally { frame.close(); }
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
    if (!this.enabled || this.failed || !this.player) return null;
    const us = Math.round(currentTimeS * 1e6);
    const frame = this.player.get(us);
    // 顺便按播放头淘汰旧缓存，控制内存
    if (this.stats.encoded % 30 === 0) this.store?.prune(us);
    return frame;
  }

  /** seek 后已解码的帧全部失效 */
  onSeek() {
    this.player?.reset();
  }

  get info() {
    if (this.failed) return `预超分不可用: ${this.reason}`;
    if (!this.enabled) return "预超分未启用";
    const span = this.store?.span ?? { from: 0, to: 0 };
    return (
      `预超分: ${this.stats.encoded} 帧已备 / ` +
      `${((span.to - span.from) / 1e6).toFixed(1)}s / ` +
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
  }
}

/** 注册/注销 MAIN world 的分片捕获脚本（需 service worker 配合） */
export const CAPTURE_SCRIPT_ID = CAPTURE_ID;
