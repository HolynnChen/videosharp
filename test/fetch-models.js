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
    desc: "ESPCN / sub-pixel CNN (仅亮度, 固定 224² 输入)",
  },
  {
    name: "realesr-general.onnx",
    url: "https://huggingface.co/tamnvcc/Real-ESRGAN-General-x4v3_float/resolve/main/onnx/model.onnx",
    desc: "realesr-general-x4v3 (RGB 通用, 固定 128² 输入)",
  },
];

/* xlsr-dynamic.onnx 不在此下载：它是把高通 XLSR 的输入空间维从固定 128
 * 放宽为动态后的产物（244KB，已随仓库提交）。
 *
 * XLSR 原始权重来自：
 *   https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/
 *   models/xlsr/releases/v0.58.0/xlsr-onnx-float.zip
 *
 * 放宽方法（需 python + onnx 包）：
 *   dims = model.graph.input[0].type.tensor_type.shape.dim
 *   dims[2].dim_param = 'H'; dims[3].dim_param = 'W'   # 清掉 dim_value
 *   del model.graph.value_info[:]                       # 清中间固定 shape
 *
 * 之所以可行：XLSR 是纯卷积（Conv/Relu/Clip/Concat/DepthToSpace，无
 * Reshape），空间维本可任意，128 只是导出时的样例尺寸。放宽后能一次前向
 * 处理整帧而不分块 —— 这是能否实时的分水岭。
 */

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
