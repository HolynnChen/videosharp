/* MSE 分片捕获 —— 主世界（MAIN world）
 *
 * 与 mse-probe-main.js 的区别：探针只统计，这个真的把分片数据传出去。
 *
 * 架构约束：hook 必须在 MAIN world（只有那里能改到页面的
 * SourceBuffer.prototype），但解码/超分需要的 WebGPU 与 ORT 都在
 * ISOLATED world 的 content script 里。所以分片数据必须跨世界传输。
 *
 * 传输用 postMessage + transfer list：把 ArrayBuffer 所有权转移过去，
 * 避免 300KB/片的结构化克隆开销。注意必须先拷贝一份 —— 原始 buffer 还要
 * 交给播放器，转移走会让页面播放崩掉。
 *
 * 三条不可违反的规则：
 *   1. 无条件透传原调用 —— 任何情况下都不能影响页面正常播放
 *   2. 全程 try/catch —— 我们的错误绝不能冒泡到页面
 *   3. 拷贝而非转移原 buffer —— 页面还要用它
 */
(() => {
  const TAG = "[VidSharp/capture]";

  if (typeof SourceBuffer === "undefined" ||
      typeof MediaSource === "undefined") {
    return;
  }

  // 页面里可能有多个 MediaSource（画中画、预览等），用 mime 区分轨道
  const sbMeta = new WeakMap();
  let videoTrackSeq = 0;

  function post(kind, payload, transfer) {
    try {
      window.postMessage(
        { __vidsharpCapture: true, kind, payload },
        "*",
        transfer || [],
      );
    } catch (e) { /* 页面可能覆盖了 postMessage，静默忽略 */ }
  }

  const origAdd = MediaSource.prototype.addSourceBuffer;
  MediaSource.prototype.addSourceBuffer = function (mime) {
    const sb = origAdd.call(this, mime);
    try {
      const isVideo = /video/i.test(mime);
      const meta = {
        mime,
        isVideo,
        trackId: isVideo ? ++videoTrackSeq : 0,
        appended: 0,
      };
      sbMeta.set(sb, meta);
      if (isVideo) {
        post("track", { mime, trackId: meta.trackId });
        console.log(`${TAG} 视频轨 #${meta.trackId}: ${mime}`);
      }
    } catch (e) { /* 不影响页面 */ }
    return sb;
  };

  /** 只看前两个 box 就能区分 init(ftyp/moov) 与 media(moof/mdat) 段 */
  function firstBoxType(u8) {
    if (u8.length < 8) return "";
    return String.fromCharCode(u8[4], u8[5], u8[6], u8[7]);
  }

  const origAppend = SourceBuffer.prototype.appendBuffer;
  SourceBuffer.prototype.appendBuffer = function (data) {
    try {
      const meta = sbMeta.get(this);
      // 只处理视频轨；音频仍由原播放器负责（我们不接管音频）
      if (meta?.isVideo && data?.byteLength) {
        const view = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

        // 必须拷贝：原 buffer 要交给播放器，转移走会让播放崩掉
        const copy = view.slice().buffer;
        const box = firstBoxType(view);
        const isInit = box === "ftyp" || box === "styp" || box === "moov";

        meta.appended++;
        post(
          "segment",
          {
            trackId: meta.trackId,
            mime: meta.mime,
            isInit,
            seq: meta.appended,
            buffer: copy,
          },
          [copy],   // 转移拷贝的所有权，零成本跨世界
        );
      }
    } catch (e) {
      // 捕获失败不能影响播放，只记一次
      if (!window.__vidsharpCaptureWarned) {
        window.__vidsharpCaptureWarned = true;
        console.warn(`${TAG} 捕获失败（播放不受影响）:`, e);
      }
    }
    // 无论如何都透传
    return origAppend.apply(this, arguments);
  };

  console.log(`${TAG} 已挂载`);
})();
