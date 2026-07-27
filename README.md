# VidSharp

用 WebGPU 对网页视频做**超分辨率 + 画质修复 + 锐化**。真人与动画内容通用，
纯 shader 实现，无模型下载。

## 处理管线

```
<video> ──importExternalTexture──▶
   [1] enhance   去块 + 去色带 + 局部对比度     (源分辨率)
   [2] EASU      方向自适应放大                 (→ 2K / 4K / 2x)
   [3] RCAS      自适应锐化                     (目标分辨率)
 ──▶ <canvas>
```

**顺序是有讲究的**：

- **去块必须在放大前** —— 否则 EASU 会把块边界当成真实边缘去「保护」，
  压缩伪影反而被强化。
- **锐化必须在放大后** —— 放大前锐化，细节会被插值重新糊掉。

中间纹理用 `rgba16float`：多 pass 串联时避免 8bit 量化误差累积，
去色带的亚量化级抖动在 8bit 中间纹理里会被直接抹掉。

## 为什么不用神经网络超分

调研 + **实测**结论：**2026 年在浏览器里用神经网络做 1080p 真人实时超分不成立**。

仓库内附带了一个可复现的探针（见下方「CNN 可行性探针」），实测数据：

| 模型 | 单块推理 (CPU/WASM 单线程) | 1080p 分块数 | 单帧总耗时 |
|---|---|---|---|
| ESPCN | 417 ms @224² | 45 | **18.8 秒** |
| realesr-general-x4v3 | 2555 ms @128² | 135 | **345 秒** |

实时预算是 **16.7ms/帧**。即便 WebGPU 相对单线程 WASM 有 10~50× 加速，
ESPCN 仍差一个数量级；而 **45 次串行推理的 GPU 调度开销本身**在 16ms 内
就已十分勉强，135 次更不可能。

根本原因是两个公开权重的**输入尺寸都是硬编码的**（ESPCN `[1,1,224,224]`、
realesr `[1,3,128,128]`），并非全卷积动态 shape，因此 1080p 必须分块，
还要额外承担块间重叠与融合的成本。

旁证：Edge 内置 VSR 用的是**原生 DirectML**（比 WebGPU 更快的路径），
RTX 2060 上 360p 仍需约 20ms/帧，且把触发条件限死在 720p 以下。微软用
更优的技术栈、专门训练的 0.1MB 模型，也只做到那个程度。

要真正实现浏览器 CNN 超分，需要的是**为此专门设计的模型**：全卷积、
单次前向、参数量 1~10 万级、且在 WebGPU 算子约束下做过针对性设计。
那是研究工作，不是集成工作。

EASU + RCAS 走的是另一条路：**不生成细节，但把已有细节重建和强化到位**。
代价是达不到 GAN 的纹理质感，收益是零依赖、全硬件覆盖、不假造纹理。

### 各算法的作用

| Pass | 算法 | 解决什么 |
|---|---|---|
| EASU | 方向自适应 12-tap 椭圆加权 | 斜边被双线性拍成阶梯 / 糊成一团 |
| RCAS | 带限幅与噪声抑制的锐化 | 画面整体偏软 |
| 去块 | 8 像素网格上的方向性平滑 | 低码率视频的方块伪影 |
| 去色带 | 渐变区三角分布抖动 | 天空、渐变面上的台阶条纹 |
| 局部对比度 | 邻域均值差非线性放大 | 暗部细节丢失、画面发闷 |

EASU / RCAS 的常量与公式已逐项对照
[ffx_fsr1.h](https://github.com/GPUOpen-Effects/FidelityFX-FSR) 原始实现核验。

## 安装

需要 Chrome / Edge 113+（WebGPU）。

1. 打开 `chrome://extensions`
2. 开启右上角「开发者模式」
3. 点「加载已解压的扩展程序」，选择本目录
4. 点工具栏图标，打开总开关

## 参数说明

| 选项 | 默认 | 说明 |
|---|---|---|
| 超分辨率 | 放大到 2K | `关闭` / `2K` / `4K` / `2倍`。已达目标分辨率的视频不会被放大 |
| 锐化强度 | 50 | 0 = 无锐化 |
| 噪声抑制 | 开 | 低码率视频建议保持开启 |
| 去块 | 40 | 低码率视频调高，高码率源可关掉 |
| 去色带 | 30 | 天空、纯色背景多的内容调高 |
| 局部对比度 | 25 | 调过头会发「HDR 味」 |
| 状态角标 | 简洁 | `简洁` / `详细参数` / `关闭`，见下 |
| 拖动对比 | 关 | 见下 |

设置改动立即生效，无需刷新页面。

## 怎么确认它在工作

### 状态角标（常驻）

画面被 canvas 接管后，肉眼无法分辨看到的是原始视频还是增强结果。
角标解决这个问题 —— 它显示在视频**左上角**：

- **简洁**：`VidSharp 1440p`（放大时显示目标高度，未放大时只显示 `VidSharp`）
- **详细参数**：`VidSharp · 1280×720 → 2560×1440 · 锐化 50 / 去块 40 / …`
- **关闭**：不显示

关键点：**角标只在首帧真正渲染成功后才出现**，所以它的存在本身就是
「增强确实生效」的证据，而不是一个静态装饰。若角标始终不出现，说明
渲染从未成功（DRM 视频、WebGPU 不可用等）。

角标是独立 DOM 元素而非画进 canvas —— 不污染画面像素，截图录屏都不会带上。
平时半透明，鼠标移到播放器上时会变清晰。

### 拖动对比（临时）

打开后画面中间出现可拖动的分割条：**左侧是处理后，右侧是原始画面**。
左右拖动即可直接比较差异。

### 控制台检查

```js
// 应返回 1（或页面视频数量）
document.querySelectorAll('.vidsharp-overlay').length

// opacity 为 0 表示从未成功渲染过一帧
getComputedStyle(document.querySelector('.vidsharp-overlay')).opacity

// 实际输出分辨率
const c = document.querySelector('.vidsharp-overlay');
console.log(c.width + 'x' + c.height);
```

## CNN 可行性探针

上面的「不用神经网络」结论是在无 GPU 的 CI 环境用 CPU/WASM 测出来的。
如果你想在**自己的真实 GPU** 上验证（比如 Apple M1、RTX 独显），可以直接跑：

```bash
npm install
npm run fetch-models   # 下载两个 ONNX 模型（约 5MB，不入 Git）
npm run probe
```

然后在 Chrome / Edge 中打开 `http://localhost:8777/`，点「开始测试」。

页面会显示本机 GPU 信息，并测量两个模型在 **WebGPU EP** 上的：

- 模型加载耗时
- warmup 耗时（含 shader 编译）
- 单块推理中位/最快耗时
- 1080p 所需分块数与单帧总耗时
- 距 60fps 预算还差多少倍

最后给出「可行 / 勉强可行 / 不可行」的判定。也可点第二个按钮同时对比
WASM EP，看 WebGPU 实际带来多少倍加速。

**注意**：模型与 `onnxruntime-web` 依赖**仅供此探针使用**，扩展本体完全不需要
它们 —— 装扩展不必 `npm install`。`models/` 已在 `.gitignore` 中，不占仓库体积。

探针踩过的两个坑（已在代码中处理，供参考）：

- `ort.env.wasm.wasmPaths` **必须是绝对 URL**。给相对路径会被拼到模块自身
  目录下，变成 `.../onnxruntime-web/node_modules/onnxruntime-web/dist/...`。
- `ort.env.wasm.numThreads = 1`。多线程依赖 blob worker，在扩展/严格 CSP
  环境下会踩 `URL.createObjectURL is not a function`。

## 已知限制

- **DRM 视频无法处理**（Netflix、Disney+ 等）。受 EME 保护的解码帧在受保护
  显存中，`importExternalTexture` 取不到像素。扩展会在连续失败 5 帧后自动
  停止并提示。
- 需要 WebGPU。不支持的浏览器会在弹窗中提示。
- 放大倍数受 GPU 纹理上限约束（Apple M1 为 8192）。超限时自动钳制。
- 覆盖层挂在 video 的紧后方，继承其层级。极少数站点若有异常的 DOM 结构，
  可能出现层级异常。

## 性能

三个 pass 都是单次全屏绘制，M1 上 1080p→2K 开销很低。若遇到掉帧：

1. 超分改为「关闭」或「2倍」（4K 输出的填充率压力最大）
2. 去色带调 0（它每帧更新 uniform 以变化抖动图案）

## 开发

```bash
npm install
npm test     # 用软件 WebGPU 后端验证三个 shader
```

测试会启动带 SwiftShader 的 Chromium，把 `texture_external` 替换成
`texture_2d` 以便注入已知像素，然后回读结果验证：

- 三个 shader 均能编译
- EASU 保留动态范围、斜边过渡单调、输出有空间变化
- enhance 全部关闭时恒等、纯色区不被破坏、去色带不污染纯色区
- RCAS `sharpness=0` 时恒等、锐化后边缘对比不降低
- 强度映射端点与单调性（回归测试，防止再次写错映射曲线）
- 输出尺寸计算 9 个用例（含纹理上限钳制）
- 角标文案生成 5 个用例（纯 JS 逻辑，shader 测试覆盖不到）

## 文件结构

```
manifest.json          MV3 配置
src/
  enhance.wgsl         去块 + 去色带 + 局部对比度
  easu.wgsl            EASU 方向自适应放大
  rcas.wgsl            RCAS 自适应锐化
  content.js           三 pass 管线 + 取帧 + 拖动对比
  content.css          覆盖层与分割条样式
  popup.html/js        控制面板
test/
  shaders.test.html    shader 单元测试
  run.js               测试驱动
  cnn-probe.html       CNN 可行性探针（浏览器内，测真实 GPU）
  serve-probe.js       探针用本地服务器
  fetch-models.js      探针用模型下载脚本
```

## 后续可加

- 按站点白名单与独立参数预设（动画/真人不同档）
- 内置 GPU 基准测试，自动推荐参数档位
- 更多画质修复手段（振铃抑制、色度上采样改进）
- 若将来出现全卷积、参数量 1~10 万级的通用轻量 SR 模型，可重新评估 CNN 档

## 许可

MIT。EASU / RCAS 算法源自 AMD FidelityFX（MIT），版权归 AMD。
