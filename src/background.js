/* Service worker —— 仅用于按需注册 MSE 探针。
 *
 * 为什么需要它：MSE 探针要 hook 页面的 SourceBuffer.prototype，必须跑在
 * MAIN world。但 manifest 里的静态 content_scripts 无法按设置条件启用，
 * 而 hook 页面原型是有侵入性的操作（页面 JS 能看到、理论上会互相干扰），
 * 不该默认开启。
 *
 * 因此用 chrome.scripting.registerContentScripts 动态注册：探针开关打开时
 * 注册，关闭时注销。默认关闭，普通用户完全不受影响。
 */

const PROBE_ID = "vidsharp-mse-probe";
const CAPTURE_ID = "vidsharp-capture";

async function isRegistered() {
  const list = await chrome.scripting.getRegisteredContentScripts({
    ids: [PROBE_ID, PROBE_ID + "-main"],
  }).catch(() => []);
  return list.length > 0;
}

async function registerProbe() {
  if (await isRegistered()) return;
  await chrome.scripting.registerContentScripts([
    {
      // MAIN world：能改页面原型，但拿不到 chrome.* API
      id: PROBE_ID + "-main",
      matches: ["<all_urls>"],
      js: ["src/mse-probe-main.js"],
      world: "MAIN",
      // 必须 document_start —— 页面脚本一旦先持有 SourceBuffer 引用，
      // 后打的补丁就绕不过去
      runAt: "document_start",
      allFrames: true,
    },
    {
      // ISOLATED world：接收 postMessage、能用扩展 API
      id: PROBE_ID,
      matches: ["<all_urls>"],
      js: ["src/mse-probe.js"],
      runAt: "document_start",
      allFrames: true,
    },
  ]);
  console.log("[VidSharp] MSE 探针已注册（需刷新页面生效）");
}

async function unregisterProbe() {
  if (!(await isRegistered())) return;
  await chrome.scripting.unregisterContentScripts({
    ids: [PROBE_ID, PROBE_ID + "-main"],
  }).catch(() => {});
  console.log("[VidSharp] MSE 探针已注销");
}

/* 预超分的分片捕获脚本。与探针同样必须在 MAIN world + document_start，
 * 但它真的把数据传出去，所以只在预超分开启时注册。 */
async function registerCapture() {
  const existing = await chrome.scripting.getRegisteredContentScripts({
    ids: [CAPTURE_ID],
  }).catch(() => []);
  if (existing.length) return;
  await chrome.scripting.registerContentScripts([{
    id: CAPTURE_ID,
    matches: ["<all_urls>"],
    js: ["src/capture-main.js"],
    world: "MAIN",
    runAt: "document_start",
    allFrames: true,
  }]);
  console.log("[VidSharp] 分片捕获已注册（需刷新页面生效）");
}

async function unregisterCapture() {
  await chrome.scripting.unregisterContentScripts({ ids: [CAPTURE_ID] })
    .catch(() => {});
}

async function sync() {
  const { mseProbe, preSuperRes } = await chrome.storage.sync.get({
    mseProbe: false, preSuperRes: false,
  });
  if (mseProbe) await registerProbe();
  else await unregisterProbe();
  if (preSuperRes) await registerCapture();
  else await unregisterCapture();
}

chrome.runtime.onInstalled.addListener(sync);
chrome.runtime.onStartup.addListener(sync);
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && ("mseProbe" in changes || "preSuperRes" in changes)) {
    sync();
  }
});
