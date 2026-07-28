/* manifest 一致性检查。
 *
 * 抓的是一类反复出现的低级错误：新增 shader 或模块后忘了加到
 * web_accessible_resources，扩展一跑就 "TypeError: Failed to fetch"。
 * 这种错误只在真实扩展环境暴露（本地 HTTP 测试照样能加载），
 * 所以必须靠静态核对兜住。
 *
 * 双向检查：
 *   - 代码里 getURL() 引用的资源，manifest 必须声明
 *   - manifest 声明的资源，文件必须存在
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const fail = [];
const ok = [];

const manifest = JSON.parse(
  fs.readFileSync(path.join(ROOT, "manifest.json"), "utf8"),
);

const declared = new Set();
for (const entry of manifest.web_accessible_resources ?? []) {
  for (const r of entry.resources ?? []) declared.add(r);
}

/** 通配符声明（如 vendor/ort/*）匹配 */
function isDeclared(res) {
  if (declared.has(res)) return true;
  for (const d of declared) {
    if (d.endsWith("/*") && res.startsWith(d.slice(0, -1))) return true;
  }
  return false;
}

// --- 1. 代码引用的资源都要声明 ---
const scanFiles = ["src/content.js", "src/cnn.js"];
const referenced = new Set();
for (const f of scanFiles) {
  const src = fs.readFileSync(path.join(ROOT, f), "utf8");
  for (const m of src.matchAll(/getURL\(["']([^"']+)["']\)/g)) {
    referenced.add(m[1]);
  }
}
for (const res of referenced) {
  if (!isDeclared(res)) {
    fail.push(`${res} 被代码引用但未在 manifest.web_accessible_resources 声明`);
  }
}
if (referenced.size && !fail.length) {
  ok.push(`${referenced.size} 个被引用资源均已声明`);
}

// --- 2. 声明的资源文件都要存在 ---
for (const res of declared) {
  if (res.endsWith("/*")) {
    const dir = path.join(ROOT, res.slice(0, -2));
    if (!fs.existsSync(dir)) fail.push(`${res} 声明的目录不存在`);
    else if (fs.readdirSync(dir).length === 0) fail.push(`${res} 目录为空`);
    continue;
  }
  if (!fs.existsSync(path.join(ROOT, res))) {
    fail.push(`${res} 已声明但文件不存在`);
  }
}
if (!fail.length) ok.push(`${declared.size} 个声明项文件均存在`);

// --- 3. content_scripts 与 popup 引用的文件存在 ---
for (const cs of manifest.content_scripts ?? []) {
  for (const f of [...(cs.js ?? []), ...(cs.css ?? [])]) {
    if (!fs.existsSync(path.join(ROOT, f))) {
      fail.push(`content_scripts 引用的 ${f} 不存在`);
    }
  }
}
const popup = manifest.action?.default_popup;
if (popup && !fs.existsSync(path.join(ROOT, popup))) {
  fail.push(`popup ${popup} 不存在`);
}

// --- 3b. background service worker 存在 ---
const sw = manifest.background?.service_worker;
if (sw) {
  if (!fs.existsSync(path.join(ROOT, sw))) {
    fail.push(`background.service_worker ${sw} 不存在`);
  } else {
    ok.push(`service worker ${sw} 存在`);
  }
}

/* --- 3c. 动态注册的 content script 存在 ---
 * chrome.scripting.registerContentScripts 里引用的 js 文件不在 manifest 里，
 * 静态检查很容易漏 —— 而它一旦缺失，注册会在运行时静默失败。 */
if (sw) {
  const swSrc = fs.readFileSync(path.join(ROOT, sw), "utf8");
  const dyn = new Set();
  for (const m of swSrc.matchAll(/js:\s*\[([^\]]+)\]/g)) {
    for (const q of m[1].matchAll(/["']([^"']+)["']/g)) dyn.add(q[1]);
  }
  for (const f of dyn) {
    if (!fs.existsSync(path.join(ROOT, f))) {
      fail.push(`动态注册引用的 ${f} 不存在`);
    }
  }
  if (dyn.size) ok.push(`动态注册的 ${dyn.size} 个脚本均存在`);

  // 动态注册需要 scripting 权限
  if (/registerContentScripts/.test(swSrc) &&
      !(manifest.permissions ?? []).includes("scripting")) {
    fail.push("使用了 registerContentScripts 但未申请 scripting 权限");
  }
}

// --- 4. SHADERS 表里的每个 wgsl 都要真实存在 ---
const contentSrc = fs.readFileSync(path.join(ROOT, "src/content.js"), "utf8");
const shaderBlock = contentSrc.match(/const SHADERS = \{([\s\S]*?)\};/);
if (shaderBlock) {
  const names = [...shaderBlock[1].matchAll(/getURL\(["']([^"']+)["']\)/g)]
    .map((m) => m[1]);
  for (const n of names) {
    if (!fs.existsSync(path.join(ROOT, n))) {
      fail.push(`SHADERS 表引用的 ${n} 不存在`);
    }
  }
  ok.push(`SHADERS 表 ${names.length} 个 shader 均存在`);
}

console.log("=== manifest 一致性检查 ===");
for (const m of ok) console.log("  ok: " + m);
for (const m of fail) console.log("  FAIL: " + m);
console.log("状态: " + (fail.length ? "failed" : "passed"));
process.exit(fail.length ? 1 : 0);
