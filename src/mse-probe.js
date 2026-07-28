/* MSE hook 探针 —— 隔离世界（ISOLATED world）伴生脚本
 *
 * MAIN world 的 hook 拿不到 chrome.* API，只能 postMessage 出来。这里负责
 * 接收、汇总，并做第二步验证：**拿到的编码分片能否真的解码成帧**。
 *
 * 这一步是「预超分」可行性的真正分水岭：
 *   截到字节 ≠ 能用。还要确认
 *     1. codec 是否被 VideoDecoder 支持（AV1/HEVC 的支持不如 H.264 普遍）
 *     2. 能否从 init segment 提取出解码器配置（avcC / hvcC / av1C）
 *     3. 解码出的 VideoFrame 能否喂进现有 WebGPU 管线
 *
 * 探针只验证前两步 —— 第三步已由现有管线证明（它本来就在处理 VideoFrame
 * 等价物）。
 */
(() => {
  const TAG = "[VidSharp/mse-probe]";
  const seen = { chunks: 0, bytes: 0, mimes: new Set(), inits: [] };
  let reported = false;

  /* 从 mime 里取出 codec 串，查 VideoDecoder 是否支持。
   * mime 形如：video/mp4; codecs="avc1.640032" */
  async function checkCodec(mime) {
    const m = /codecs="?([^";]+)"?/.exec(mime || "");
    if (!m) return { codec: null, supported: null, reason: "mime 里没有 codecs" };
    const codec = m[1].trim();
    if (typeof VideoDecoder === "undefined") {
      return { codec, supported: false, reason: "此浏览器无 WebCodecs" };
    }
    try {
      // 分辨率随便给，isConfigSupported 主要看 codec 串
      const res = await VideoDecoder.isConfigSupported({
        codec, codedWidth: 1920, codedHeight: 1080,
      });
      return {
        codec,
        supported: !!res.supported,
        hardware: res.config?.hardwareAcceleration,
        reason: res.supported ? "" : "VideoDecoder 不支持此 codec",
      };
    } catch (err) {
      return { codec, supported: false, reason: err.message };
    }
  }

  /* 重编码是"预超分"的存储基础：超分后的未压缩像素无法缓存
   * （2 秒 2K 就要 0.82GB），必须编码成压缩格式才能存下整段。
   * 所以必须先确认浏览器能编码 —— 这是方案的另一个前提。 */
  async function checkEncoders() {
    if (typeof VideoEncoder === "undefined") {
      console.log(`${TAG} ✗ 第三关：无 VideoEncoder，无法重编码存储`);
      return;
    }
    // 目标分辨率按 2K 试；码率给高一些（存本地不心疼，减少二次损失）
    const candidates = [
      { name: "H.264 (avc1.640033)", codec: "avc1.640033" },
      { name: "VP9 (vp09.00.10.08)", codec: "vp09.00.10.08" },
      { name: "AV1 (av01.0.08M.08)", codec: "av01.0.08M.08" },
    ];
    for (const c of candidates) {
      try {
        const res = await VideoEncoder.isConfigSupported({
          codec: c.codec,
          width: 2560, height: 1440,
          bitrate: 20_000_000,       // 20Mbps，高码率压低二次损失
          framerate: 30,
        });
        const hw = res.config?.hardwareAcceleration;
        console.log(
          `${TAG} ${res.supported ? "✓" : "✗"} 第三关 编码 ${c.name}: ` +
          `${res.supported ? "支持" : "不支持"}${hw ? ` (${hw})` : ""}`,
        );
      } catch (err) {
        console.log(`${TAG} ✗ 第三关 编码 ${c.name}: ${err.message}`);
      }
    }
  }

  async function report() {
    if (reported) return;
    reported = true;

    console.log(`${TAG} ─── 可行性结论 ───`);
    console.log(`${TAG} 截获分片: ${seen.chunks} 个 / ` +
                `${(seen.bytes / 1024 / 1024).toFixed(2)} MB`);
    console.log(`${TAG} ✓ 第一关：能从 MSE 截到编码数据`);

    for (const mime of seen.mimes) {
      if (!/video/i.test(mime)) continue;
      const r = await checkCodec(mime);
      if (r.supported) {
        console.log(`${TAG} ✓ 第二关：codec ${r.codec} 可被 VideoDecoder 解码` +
                    (r.hardware ? ` (${r.hardware})` : ""));
      } else {
        console.log(`${TAG} ✗ 第二关：codec ${r.codec} 无法解码 — ${r.reason}`);
      }
    }

    const videoInits = seen.inits.filter((i) => /video/i.test(i.mime || ""));
    if (videoInits.length) {
      console.log(`${TAG} ✓ 抓到视频 init segment（含解码器配置）: ` +
                  `${videoInits[0].size}B boxes=[${videoInits[0].boxes}]`);
    } else {
      console.log(`${TAG} ⚠ 未抓到 init segment —— 可能挂载太晚，` +
                  `需确认 run_at: document_start 生效`);
    }

    await checkEncoders();

    console.log(`${TAG} 注：截到数据只是第一步。完整方案还需自建播放器` +
                `（音视频同步/seek/字幕），且缓存超分结果受显存限制。`);
  }

  window.addEventListener("message", (ev) => {
    const d = ev.data;
    if (!d || d.__vidsharpProbe !== true) return;

    if (d.kind === "chunk") {
      seen.chunks++;
      seen.bytes += d.payload.size || 0;
      if (d.payload.mime) seen.mimes.add(d.payload.mime);
      if (d.payload.isInit) seen.inits.push(d.payload);
      // 攒够样本再出结论，避免只看到 init segment 就下判断
      if (seen.chunks >= 6) report();
    }
  });

  console.log(`${TAG} 伴生脚本就绪，等待分片…`);
})();
