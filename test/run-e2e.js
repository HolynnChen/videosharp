/* 解码链路端到端测试驱动。
 *
 * 与 run.js 分开：解码测试要 WebCodecs（不需要 GPU），shader 测试要软件
 * WebGPU（SwiftShader，很慢）。启动参数不同，混在一起会互相拖慢。
 */
const { chromium } = require("playwright-core");
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".mp4": "video/mp4",
  ".m4s": "video/iso.segment",
};

function serve() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split("?")[0]);
      const filePath = path.join(ROOT, path.normalize(rel));
      if (!filePath.startsWith(ROOT)) {
        res.writeHead(403);
        res.end();
        return;
      }
      let stat = null;
      try { stat = fs.statSync(filePath); } catch { /* 不存在 */ }
      if (!stat || !stat.isFile()) {
        res.writeHead(404);
        res.end("not found: " + rel);
        return;
      }
      res.writeHead(200, {
        "Content-Type":
          MIME[path.extname(filePath)] || "application/octet-stream",
        "Content-Length": stat.size,
      });
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

(async () => {
  const server = await serve();
  const port = server.address().port;

  const browser = await chromium.launch({
    executablePath:
      process.env.CHROMIUM_PATH ||
      "/home/ubuntu/.cache/ms-playwright/chromium-1187/chrome-linux/chrome",
    args: ["--no-sandbox","--enable-unsafe-webgpu","--enable-features=Vulkan","--use-angle=swiftshader","--use-vulkan=swiftshader","--enable-gpu"],
  });

  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

  await page.goto(`http://127.0.0.1:${port}/test/e2e.test.html`);

  let result;
  try {
    await page.waitForFunction("window.__done", { timeout: 600000 });
    result = await page.evaluate("window.__done");
  } catch {
    const partial = await page
      .evaluate("document.getElementById('result').textContent")
      .catch(() => "");
    result = { status: "timeout", log: [partial || "未在 10 分钟内完成"] };
  }

  await browser.close();
  server.close();

  console.log("=== 预超分完整链路 E2E ===");
  for (const line of result.log) console.log("  " + line);
  if (errors.length) {
    console.log("--- 页面错误 ---");
    for (const e of errors) console.log("  " + e);
  }
  console.log("状态: " + result.status);
  process.exit(result.status === "passed" ? 0 : 1);
})();
