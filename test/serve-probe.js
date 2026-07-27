/* 本地静态服务器 —— 用于在真实 GPU 的浏览器里打开 CNN 探针。
 *
 * 为什么需要它：file:// 下 fetch .onnx / .wasm 会被 CORS 拦截。
 *
 * 用法：
 *   npm run probe
 * 然后在 Chrome 打开提示的地址。
 */
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PORT = Number(process.env.PORT) || 8777;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".wgsl": "text/plain; charset=utf-8",
  ".json": "application/json",
  ".map": "application/json",
};

const server = http.createServer((req, res) => {
  let rel = decodeURIComponent(req.url.split("?")[0]);
  if (rel === "/") rel = "/test/cnn-probe.html";
  if (rel === "/favicon.ico") { res.writeHead(204); res.end(); return; }

  const filePath = path.join(ROOT, path.normalize(rel));
  if (!filePath.startsWith(ROOT) || !fs.existsSync(filePath)
      || !fs.statSync(filePath).isFile()) {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("404 not found: " + rel);
    return;
  }

  res.writeHead(200, {
    "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream",
    // ORT 若启用多线程需要这两个头；探针用单线程，加上无害
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cache-Control": "no-cache",
  });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, () => {
  console.log(`\n  CNN 探针已启动:\n`);
  console.log(`    http://localhost:${PORT}/\n`);
  console.log(`  在 Chrome / Edge 中打开上面的地址，点「开始测试」。`);
  console.log(`  Ctrl+C 停止。\n`);
});
