/* 超分帧缓存与播放
 *
 * 预超分的最后一环：把编码好的超分 chunk 存起来，播放时按 currentTime
 * 检索、解码、显示。
 *
 * 架构选择（关键）：音频与时间轴仍由原播放器负责 —— 原 <video> 继续播放
 * （视觉上被我们的 canvas 覆盖），我们只按它的 currentTime 取对应的超分帧。
 * 这样 seek、倍速、暂停、弹幕对齐全部沿用原播放器，规避了自建播放器最大
 * 的坑（音视频同步）。
 *
 * 缓存策略：
 *   - 存编码后的 chunk（不是像素）。2K 每帧未压缩 14MB，编码后约 50KB。
 *   - 按 GOP（关键帧到下一关键帧）分组。解码必须从关键帧开始，所以检索
 *     时要回退到所属 GOP 的头部。
 *   - 滑动窗口淘汰：只保留播放头附近若干秒，避免长视频撑爆内存。
 */

/** 保留播放头前后多少秒的缓存 */
const KEEP_BEHIND_S = 5;
const KEEP_AHEAD_S = 60;

/**
 * 编码后的超分帧仓库。
 *
 * chunk 按时间戳有序存放，并记录关键帧位置以支持从 GOP 头解码。
 */
export class FrameStore {
  constructor() {
    /** @type {Array<{ts:number, dur:number, type:string, data:Uint8Array}>} */
    this.chunks = [];
    this.decoderConfig = null;   // VideoDecoder 配置（含 description）
    this.totalBytes = 0;
  }

  /** 记录编码器给出的解码配置。第一个 chunk 的 metadata 里带 */
  setConfig(config) {
    if (!this.decoderConfig && config) {
      this.decoderConfig = config;
    }
  }

  add(chunk, metadata) {
    if (metadata?.decoderConfig) this.setConfig(metadata.decoderConfig);

    // chunk 的数据必须立刻拷出 —— EncodedVideoChunk 不保证长期有效
    const data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);

    this.chunks.push({
      ts: chunk.timestamp,
      dur: chunk.duration ?? 0,
      type: chunk.type,
      data,
    });
    this.totalBytes += data.byteLength;
  }

  /**
   * 找到覆盖 timeUs 的 chunk 索引，以及其所属 GOP 的起始索引。
   *
   * 解码必须从关键帧开始 —— 直接从中间 delta 帧解码会失败或出错误画面，
   * 所以要回退到 GOP 头。
   *
   * @returns {{gopStart:number, target:number}|null}
   */
  locate(timeUs) {
    if (!this.chunks.length) return null;

    // 二分找最后一个 ts <= timeUs 的 chunk
    let lo = 0, hi = this.chunks.length - 1, target = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.chunks[mid].ts <= timeUs) { target = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    if (target < 0) return null;

    // 回退到所属 GOP 的关键帧
    let gopStart = target;
    while (gopStart > 0 && this.chunks[gopStart].type !== "key") gopStart--;
    if (this.chunks[gopStart].type !== "key") return null;   // 尚无关键帧

    return { gopStart, target };
  }

  /** 淘汰播放头附近窗口之外的 chunk，控制内存 */
  prune(currentTimeUs) {
    const minTs = currentTimeUs - KEEP_BEHIND_S * 1e6;
    const maxTs = currentTimeUs + KEEP_AHEAD_S * 1e6;

    // 只从头部裁剪：尾部是未来的帧，正是我们想留的。
    // 且必须裁到关键帧边界，否则剩下的 delta 帧无法独立解码。
    let cut = 0;
    for (let i = 0; i < this.chunks.length; i++) {
      if (this.chunks[i].ts >= minTs) break;
      if (this.chunks[i].type === "key") cut = i;
    }
    if (cut > 0) {
      for (let i = 0; i < cut; i++) this.totalBytes -= this.chunks[i].data.byteLength;
      this.chunks.splice(0, cut);
    }

    // 超前太多的也丢（通常不会发生，除非用户长时间不看）
    while (this.chunks.length && this.chunks[this.chunks.length - 1].ts > maxTs) {
      this.totalBytes -= this.chunks.pop().data.byteLength;
    }
  }

  get span() {
    if (!this.chunks.length) return { from: 0, to: 0 };
    return {
      from: this.chunks[0].ts,
      to: this.chunks[this.chunks.length - 1].ts,
    };
  }

  clear() {
    this.chunks.length = 0;
    this.totalBytes = 0;
  }
}

/**
 * 按需解码超分帧的播放器。
 *
 * 只解码当前需要的 GOP，解出的帧短暂缓存以应对连续播放。
 */
export class FramePlayer {
  constructor(store) {
    this.store = store;
    this.decoder = null;
    /** @type {Map<number, VideoFrame>} ts → frame */
    this.ready = new Map();
    this.decodingGop = -1;
    this.failed = false;
    this.maxReady = 90;    // 约 3 秒 @30fps，够连续播放
  }

  async ensureDecoder() {
    if (this.decoder?.state === "configured") return true;
    const config = this.store.decoderConfig;
    if (!config) return false;

    try {
      this.decoder = new VideoDecoder({
        output: (frame) => {
          // 超出上限时丢最旧的，避免显存无限增长
          if (this.ready.size >= this.maxReady) {
            const oldest = this.ready.keys().next().value;
            this.ready.get(oldest)?.close();
            this.ready.delete(oldest);
          }
          this.ready.set(frame.timestamp, frame);
        },
        error: (err) => {
          this.failed = true;
          console.warn("[VidSharp/playback] 解码失败:", err);
        },
      });
      this.decoder.configure({ ...config, optimizeForLatency: true });
      return true;
    } catch (err) {
      this.failed = true;
      console.warn("[VidSharp/playback] 解码器配置失败:", err);
      return false;
    }
  }

  /**
   * 取 timeUs 对应的超分帧。若尚未解码则触发解码并返回 null，
   * 调用方本帧应回落到实时管线。
   *
   * @returns {VideoFrame|null} 不转移所有权，调用方不要 close
   */
  get(timeUs) {
    if (this.failed) return null;

    // 找最接近且不超过 timeUs 的已解码帧
    let best = null, bestTs = -Infinity;
    for (const [ts, frame] of this.ready) {
      if (ts <= timeUs && ts > bestTs) { bestTs = ts; best = frame; }
    }
    // 容差半帧（33ms/2），避免因浮点误差取到上一帧
    if (best && timeUs - bestTs < 50_000) return best;

    this.requestDecode(timeUs);
    return null;
  }

  /** 触发所需 GOP 的解码 */
  requestDecode(timeUs) {
    const loc = this.store.locate(timeUs);
    if (!loc) return;
    if (loc.gopStart === this.decodingGop) return;   // 已在解这个 GOP

    (async () => {
      if (!(await this.ensureDecoder())) return;
      this.decodingGop = loc.gopStart;
      try {
        // 从 GOP 头解到目标之后一小段，供连续播放
        const end = Math.min(this.store.chunks.length, loc.target + 60);
        for (let i = loc.gopStart; i < end; i++) {
          const c = this.store.chunks[i];
          this.decoder.decode(new EncodedVideoChunk({
            type: c.type,
            timestamp: c.ts,
            duration: c.dur || undefined,
            data: c.data,
          }));
        }
      } catch (err) {
        this.failed = true;
        console.warn("[VidSharp/playback] 解码请求失败:", err);
      }
    })();
  }

  /** seek 后旧帧全部失效 */
  reset() {
    for (const f of this.ready.values()) f.close();
    this.ready.clear();
    this.decodingGop = -1;
    if (this.decoder?.state === "configured") {
      try { this.decoder.reset(); } catch { /* 忽略 */ }
    }
  }

  close() {
    this.reset();
    if (this.decoder && this.decoder.state !== "closed") {
      try { this.decoder.close(); } catch { /* 忽略 */ }
    }
    this.decoder = null;
  }
}
