/* MSE hook 探针 —— 主世界（MAIN world）部分
 *
 * 目的只有一个：验证能否从页面的 MediaSource 管线里截到编码分片。
 * 这是「预超分」方案的第一道关卡；截不到就没有后续。
 *
 * 为什么必须在 MAIN world：
 *   content script 默认跑在 ISOLATED world，那里的 SourceBuffer.prototype
 *   与页面的是两个不同对象，改它对页面毫无影响。只有 world:"MAIN" 才共享
 *   页面的执行环境、能改到真正被使用的原型。
 *
 * 代价（Chrome 文档明确）：MAIN world 拿不到 chrome.runtime 等扩展 API，
 * 因此结果只能用 window.postMessage 传给 ISOLATED world 的伴生脚本。
 * 且页面 JS 能看到并干扰这段代码 —— 探针无所谓，产品化需考虑。
 *
 * 必须 run_at: document_start —— 页面脚本一旦先拿到 SourceBuffer 引用，
 * 后打的补丁就绕不过去了。
 */
(() => {
  const TAG = "[VidSharp/mse-probe]";

  // 只在页面里存在 MSE 时才有意义
  if (typeof SourceBuffer === "undefined" ||
      typeof MediaSource === "undefined") {
    return;
  }

  const stats = {
    sourceBuffers: [],      // 每个 SourceBuffer 的 mime 与累计信息
    totalChunks: 0,
    totalBytes: 0,
    initSegments: [],       // 抓到的初始化段（含 codec 信息）
    started: Date.now(),
  };

  function post(kind, payload) {
    window.postMessage({ __vidsharpProbe: true, kind, payload }, "*");
  }

  /* 记录 addSourceBuffer 的 mime，这是判断 codec 的最快途径 —— 决定
   * VideoDecoder 能否解（AV1/HEVC 的浏览器解码支持不如 H.264 普遍）。 */
  const origAdd = MediaSource.prototype.addSourceBuffer;
  MediaSource.prototype.addSourceBuffer = function (mime) {
    const sb = origAdd.call(this, mime);
    try {
      const entry = { mime, chunks: 0, bytes: 0, isVideo: /video/i.test(mime) };
      stats.sourceBuffers.push(entry);
      sbMeta.set(sb, entry);
      post("sourceBuffer", { mime });
      console.log(`${TAG} addSourceBuffer: ${mime}`);
    } catch (e) { /* 不能影响页面 */ }
    return sb;
  };

  const sbMeta = new WeakMap();

  /* fMP4 的第一个 append 通常是 init segment（含 moov/avcC，即解码器配置），
   * 后续是 media segment（moof+mdat）。判断依据是 box 类型。 */
  function peekBoxes(buf) {
    const u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf)
                                          : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    const boxes = [];
    let off = 0;
    // 只扫前几个 box，够判断类型即可
    while (off + 8 <= u8.length && boxes.length < 6) {
      const size = (u8[off] << 24) | (u8[off + 1] << 16) |
                   (u8[off + 2] << 8) | u8[off + 3];
      const type = String.fromCharCode(u8[off + 4], u8[off + 5],
                                       u8[off + 6], u8[off + 7]);
      boxes.push(type);
      if (size <= 0) break;      // size 0/1 需特殊处理，探针里直接停
      off += size;
    }
    return boxes;
  }

  const origAppend = SourceBuffer.prototype.appendBuffer;
  SourceBuffer.prototype.appendBuffer = function (data) {
    try {
      const len = data?.byteLength ?? 0;
      const meta = sbMeta.get(this);
      stats.totalChunks++;
      stats.totalBytes += len;
      if (meta) { meta.chunks++; meta.bytes += len; }

      const boxes = peekBoxes(data);
      const isInit = boxes.includes("ftyp") || boxes.includes("moov");

      if (isInit && stats.initSegments.length < 4) {
        stats.initSegments.push({ mime: meta?.mime, size: len, boxes });
        console.log(`${TAG} init segment: ${meta?.mime} ${len}B boxes=${boxes}`);
      }

      // 前几个分片详细打印，之后只累计，避免刷屏
      if (stats.totalChunks <= 8) {
        console.log(
          `${TAG} chunk #${stats.totalChunks} ${len}B ` +
          `boxes=[${boxes.join(",")}] mime=${meta?.mime ?? "?"}`,
        );
      }

      post("chunk", {
        n: stats.totalChunks,
        size: len,
        boxes,
        mime: meta?.mime,
        isInit,
      });
    } catch (e) {
      console.warn(`${TAG} hook 内部错误（不影响播放）:`, e);
    }
    // 无论探针是否出错，都必须把原始调用透传，否则播放会坏
    return origAppend.apply(this, arguments);
  };

  // 定期汇总，方便在控制台一眼看到全貌
  let lastReported = 0;
  setInterval(() => {
    if (!stats.totalChunks) return;
    // 分片数没变化就不重复打印 —— B 站是大块预缓冲，
    // 缓冲完成后会长时间没有新分片，每 5 秒刷一条纯属噪音
    if (stats.totalChunks === lastReported) return;
    lastReported = stats.totalChunks;
    const secs = ((Date.now() - stats.started) / 1000).toFixed(0);
    const lines = stats.sourceBuffers.map(
      (s) => `    ${s.isVideo ? "视频" : "音频"} ${s.mime}\n` +
             `      ${s.chunks} 分片 / ${(s.bytes / 1024 / 1024).toFixed(2)} MB`,
    );
    console.log(
      `${TAG} ${secs}s 汇总: ${stats.totalChunks} 分片 ` +
      `${(stats.totalBytes / 1024 / 1024).toFixed(2)} MB\n${lines.join("\n")}`,
    );
    post("summary", { stats: JSON.parse(JSON.stringify(stats)) });
  }, 5000);

  window.__vidsharpMseProbe = stats;
  console.log(`${TAG} 已挂载。播放视频后看这里，或读 window.__vidsharpMseProbe`);
})();
