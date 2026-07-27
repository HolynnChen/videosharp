/* 下载 CNN 探针所需的 ONNX 模型。
 *
 * 模型不入 Git（约 5MB，且扩展本体不需要）。运行 `npm run probe` 前
 * 若缺文件会提示先跑这个脚本。
 */
const fs = require("fs");
const path = require("path");
const https = require("https");

const MODELS = [
  {
    name: "espcn.onnx",
    url: "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
    desc: "ESPCN / sub-pixel CNN (仅亮度通道, 224² 输入)",
  },
  {
    name: "realesr-general.onnx",
    url: "https://huggingface.co/tamnvcc/Real-ESRGAN-General-x4v3_float/resolve/main/onnx/model.onnx",
    desc: "realesr-general-x4v3 (RGB 通用, 128² 输入)",
  },
];

const DIR = path.resolve(__dirname, "..", "models");

function download(url, dest, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error("重定向次数过多"));
    https.get(url, { headers: { "User-Agent": "vidsharp-probe" } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return download(res.headers.location, dest, redirects + 1)
          .then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode}`));
      }
      const tmp = dest + ".part";
      const file = fs.createWriteStream(tmp);
      res.pipe(file);
      file.on("finish", () => {
        file.close(() => {
          fs.renameSync(tmp, dest);
          resolve();
        });
      });
      file.on("error", (err) => {
        fs.unlinkSync(tmp);
        reject(err);
      });
    }).on("error", reject);
  });
}

(async () => {
  fs.mkdirSync(DIR, { recursive: true });
  for (const m of MODELS) {
    const dest = path.join(DIR, m.name);
    if (fs.existsSync(dest)) {
      const mb = (fs.statSync(dest).size / 1024 / 1024).toFixed(2);
      console.log(`  已存在 ${m.name} (${mb} MB)`);
      continue;
    }
    process.stdout.write(`  下载 ${m.name} — ${m.desc} … `);
    try {
      await download(m.url, dest);
      const mb = (fs.statSync(dest).size / 1024 / 1024).toFixed(2);
      console.log(`完成 (${mb} MB)`);
    } catch (err) {
      console.log(`失败: ${err.message}`);
      process.exitCode = 1;
    }
  }
  console.log(`\n  模型目录: ${DIR}`);
  console.log(`  接下来运行: npm run probe\n`);
})();
