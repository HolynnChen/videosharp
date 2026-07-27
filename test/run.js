// 用 Playwright 启动带软件 WebGPU 后端的 Chromium，跑 RCAS shader 单元测试。
// 无头环境没有真实 GPU，靠 SwiftShader/Vulkan 软件实现验证数学正确性。
const { chromium } = require("playwright-core");
const path = require("path");
const http = require("http");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..");
const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wgsl": "text/plain",
};

// file:// 下 fetch 被 CORS 拦，起个本地 server 供测试页加载 shader
function serve() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const rel = decodeURIComponent(req.url.split("?")[0]);
      if (rel === "/favicon.ico") {
        res.writeHead(204);
        res.end();
        return;
      }
      const filePath = path.join(ROOT, path.normalize(rel));
      if (!filePath.startsWith(ROOT) || !fs.existsSync(filePath)) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, {
        "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream",
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
    // headless shell 缺完整 GPU 栈，必须用完整 Chromium
    executablePath:
      process.env.CHROMIUM_PATH ||
      "/home/ubuntu/.cache/ms-playwright/chromium-1187/chrome-linux/chrome",
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-angle=swiftshader",
      "--use-vulkan=swiftshader",
      "--enable-gpu",
      "--no-sandbox",
    ],
  });

  const page = await browser.newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push("pageerror: " + e.message));

  const url = `http://127.0.0.1:${port}/test/shaders.test.html`;
  await page.goto(url);

  let result;
  try {
    await page.waitForFunction("window.__done !== undefined", { timeout: 30000 });
    result = await page.evaluate("window.__done");
  } catch {
    result = { status: "timeout", log: ["测试未在 30s 内完成"] };
  }

  await browser.close();
  server.close();

  console.log("=== VidSharp shader 测试 ===");
  for (const line of result.log) console.log("  " + line);
  if (consoleErrors.length) {
    console.log("--- 浏览器控制台错误 ---");
    for (const e of consoleErrors) console.log("  " + e);
  }
  console.log("状态: " + result.status);
  process.exit(result.status === "passed" ? 0 : 1);
})();
