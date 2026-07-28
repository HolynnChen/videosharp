/* fMP4 解析 + 视频解码
 *
 * 把从 MSE 截获的分片变成可用的 VideoFrame。
 *
 * 为什么需要 mp4box.js：MSE 收到的是 fMP4 分片（init segment 含 moov/avcC，
 * media segment 含 moof+mdat）。VideoDecoder 要的是裸 sample（一个个
 * EncodedVideoChunk）+ 解码器配置（description）。中间的解封装必须自己做，
 * mp4box.js 负责这一步。
 *
 * 关键细节（踩过才知道）：
 *   - avcC/hvcC/av1C 必须从 init segment 里提取并作为 description 传给
 *     VideoDecoder，否则 H.264 的 SPS/PPS 缺失，解码直接失败。
 *   - mp4box 的 appendBuffer 要求每块带 fileStart 偏移，且必须连续递增。
 *   - VideoFrame 持有 GPU/媒体资源，必须 close()。不 close 会让解码器因
 *     在途帧过多而 stall（backpressure），表现为解码卡死。
 */

const MP4BOX_URL = chrome.runtime.getURL("vendor/mp4box/mp4box.all.mjs");

let mp4boxPromise = null;
async function loadMP4Box() {
  if (!mp4boxPromise) {
    // 新版 mp4box 是纯命名导出，没有 default —— 直接取整个 namespace
    mp4boxPromise = import(MP4BOX_URL);
    mp4boxPromise.catch(() => { mp4boxPromise = null; });
  }
  return mp4boxPromise;
}

/** 从 mp4box 的 track 信息里取出 VideoDecoder 需要的 description */
function extractDescription(MP4Box, mp4File, track) {
  const trak = mp4File.getTrackById(track.id);
  for (const entry of trak.mdia.minf.stbl.stsd.entries) {
    // 不同 codec 的配置 box 名字不同，取到哪个用哪个
    const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
    if (!box) continue;
    // box.write 需要一个 big-endian DataStream；写出后去掉 8 字节 box 头，
    // 因为 VideoDecoder 的 description 只要 box 内容不要头
    const stream = new MP4Box.DataStream(
      undefined, 0, MP4Box.Endianness.BIG_ENDIAN,
    );
    box.write(stream);
    return new Uint8Array(stream.buffer, 8);
  }
  return null;
}

/**
 * 一条视频轨的解码管线。
 *
 * 用法：先 pushSegment(initSegment)，再陆续 pushSegment(mediaSegment)，
 * 解码出的帧通过构造时传入的 onFrame 回调交出（调用方负责 close）。
 */
export class TrackDecoder {
  /**
   * @param onFrame (frame: VideoFrame, timestampUs: number) => void
   *   回调内必须在用完后 frame.close()，否则解码器会 stall。
   */
  constructor(onFrame, onError) {
    this.onFrame = onFrame;
    this.onError = onError ?? ((e) => console.warn("[VidSharp/decode]", e));
    this.mp4 = null;
    this.decoder = null;
    this.offset = 0;          // mp4box 要求 fileStart 连续递增
    this.configured = false;
    this.pendingChunks = [];  // 配置完成前到达的 sample
    this.width = 0;
    this.height = 0;
    this.decodedCount = 0;
    this.failed = false;
  }

  async init() {
    const MP4Box = await loadMP4Box();
    this.MP4Box = MP4Box;
    this.mp4 = MP4Box.createFile();

    this.mp4.onError = (e) => {
      this.failed = true;
      this.onError(new Error("mp4box: " + e));
    };

    this.mp4.onReady = (info) => {
      try {
        const track = info.videoTracks?.[0];
        if (!track) throw new Error("init segment 里没有视频轨");

        this.width = track.track_width || track.video?.width || 0;
        this.height = track.track_height || track.video?.height || 0;

        const description = extractDescription(this.MP4Box, this.mp4, track);
        if (!description) {
          throw new Error(`无法提取 ${track.codec} 的解码器配置`);
        }

        this.decoder = new VideoDecoder({
          output: (frame) => {
            this.decodedCount++;
            try {
              this.onFrame(frame, frame.timestamp);
            } catch (err) {
              // 回调出错也要 close，否则解码器 stall
              frame.close();
              this.onError(err);
            }
          },
          error: (err) => {
            this.failed = true;
            this.onError(err);
          },
        });

        this.decoder.configure({
          codec: track.codec,
          codedWidth: this.width,
          codedHeight: this.height,
          description,
          // 硬件优先，但不强制 —— 强制会在不支持时直接失败
          hardwareAcceleration: "no-preference",
          // 视频场景下延迟优先于吞吐
          optimizeForLatency: true,
        });

        this.configured = true;
        console.log(
          `[VidSharp/decode] 就绪: ${track.codec} ${this.width}×${this.height}`,
        );

        // 冲掉配置前积压的 sample
        for (const c of this.pendingChunks) this.decoder.decode(c);
        this.pendingChunks.length = 0;
      } catch (err) {
        this.failed = true;
        this.onError(err);
      }
    };

    // 必须显式请求 sample，否则 onSamples 不会触发
    this.mp4.onSamples = (id, user, samples) => {
      for (const s of samples) {
        const chunk = new EncodedVideoChunk({
          type: s.is_sync ? "key" : "delta",
          // mp4box 的 timescale 是每轨的，统一换算成微秒
          timestamp: (s.cts / s.timescale) * 1e6,
          duration: (s.duration / s.timescale) * 1e6,
          data: s.data,
        });
        if (this.configured && this.decoder?.state === "configured") {
          try {
            this.decoder.decode(chunk);
          } catch (err) {
            this.onError(err);
          }
        } else {
          this.pendingChunks.push(chunk);
        }
      }
    };

    return this;
  }

  /** 喂入一个 fMP4 分片（init 或 media 段皆可） */
  pushSegment(arrayBuffer) {
    if (this.failed || !this.mp4) return;
    try {
      // mp4box 要求带 fileStart，且必须与之前连续
      const buf = arrayBuffer;
      buf.fileStart = this.offset;
      this.offset += buf.byteLength;
      this.mp4.appendBuffer(buf);

      // onReady 之后才能设置提取；设置一次即可
      if (this.configured && !this.extractStarted) {
        this.extractStarted = true;
        const tracks = this.mp4.getInfo().videoTracks;
        if (tracks?.[0]) {
          // nbSamples 设小一点，让帧尽快吐出而非攒批
          this.mp4.setExtractionOptions(tracks[0].id, null, { nbSamples: 1 });
          this.mp4.start();
        }
      }
    } catch (err) {
      this.failed = true;
      this.onError(err);
    }
  }

  /** 等待解码队列排空 */
  async flush() {
    if (this.decoder?.state === "configured") {
      try { await this.decoder.flush(); } catch { /* 忽略 */ }
    }
  }

  async close() {
    this.failed = true;
    try { await this.flush(); } catch { /* 忽略 */ }
    if (this.decoder && this.decoder.state !== "closed") {
      try { this.decoder.close(); } catch { /* 忽略 */ }
    }
    this.decoder = null;
    try { this.mp4?.flush(); } catch { /* 忽略 */ }
    this.mp4 = null;
    this.pendingChunks.length = 0;
  }
}
