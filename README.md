# VidSharp

用 WebGPU 对网页视频做**超分辨率 + 画质修复 + 锐化**。真人与动画内容通用，
纯 shader 实现，无模型下载。

## 处理管线

```
<video> ──importExternalTexture──▶
   [1] enhance   去块 + 去色带 + 局部对比度     (源分辨率)
   [2] EASU      方向自适应放大                 (→ 2K / 4K / 2x)
   [3] RCAS      自适应锐化                     (目标分辨率)
   [4] badge     状态角标合成（可关闭）
 ──▶ <canvas>
```

**顺序是有讲究的**：

- **去块必须在放大前** —— 否则 EASU 会把块边界当成真实边缘去「保护」，
  压缩伪影反而被强化。
- **锐化必须在放大后** —— 放大前锐化，细节会被插值重新糊掉。

中间纹理用 `rgba16float`：多 pass 串联时避免 8bit 量化误差累积，
去色带的亚量化级抖动在 8bit 中间纹理里会被直接抹掉。

## 为什么不用神经网络超分

**这是实测结论，不是推测。** 在 Apple M1（Metal 3）上用仓库内的探针实测：

### 同一块 GPU，纯 shader 比 CNN 快约 450 倍

| 方案 | 1080p 单帧 | fps |
|---|---|---|
| **EASU + RCAS（本项目）** | **0.88 ms** → 2K | **1133** |
| **EASU + RCAS（本项目）** | **1.63 ms** → 4K | **614** |
| ESPCN（224² × 45 块） | 397 ms | 2.5 |
| realesr-general-x4v3（128² × 135 块） | 33485 ms | 0.03 |

EASU 处理完整 1080p→2K 只要 0.88ms；ESPCN 处理一个 224² 小块（像素量仅为
前者的 2.4%）就要 8.81ms。这不是「优化不足」，是路线本身的量级差距。

### 而且优化空间已经探到底了

批处理实测（ESPCN，M1）：

| batch | 单块均摊 | 1080p 单帧 |
|---:|---|---|
| 1 | 8.66 ms | 390 ms |
| 4 | 6.75 ms | 304 ms |
| 8 | 6.63 ms | 298 ms |
| 16 | 7.70 ms | 346 ms |

**增益仅 1.12×**，且 batch=16 反而比 8 更慢。

这一条排除了「GPU 负载低所以还有余量」的猜想 —— 瓶颈不在内核调度开销，
而在实际算力与内存带宽。**GPU 看似空闲，但已被这个模型的计算模式占满。**
ESPCN 距 60fps 差 24 倍，这 24 倍拿不到。

### 根因：模型不是全卷积

两个公开权重的输入空间尺寸都**硬编码**（ESPCN `[batch,1,224,224]`、
realesr `[1,3,128,128]`），图里的 `Reshape` 带常量 shape。调研阶段
「fully-convolutional 支持任意尺寸」的说法在这两个具体权重上**不成立** ——
必须实测才能发现。因此 1080p 只能分块，还要额外承担块间重叠与融合成本。

另外 realesr 的 WebGPU/WASM 加速比只有 3.9×（ESPCN 为 16×），提示它有算子
回退到 CPU —— 嫌疑是 `PRelu`（注意不是 `LeakyRelu`，是带可学习参数的另一个
算子）与 `Resize`。但即使修好也差 2000 倍，无救。

### 旁证

Edge 内置 VSR 用的是**原生 DirectML**（比 WebGPU 更快的路径），RTX 2060 上
360p 仍需约 20ms/帧，且把触发条件限死在 720p 以下。微软用更优的技术栈、
专门训练的 0.1MB 模型，也只做到那个程度。

### 结论

要在浏览器里做 CNN 超分，需要的是**为此专门设计的模型**：全卷积、单次前向、
参数量 1~10 万级、且在 WebGPU 算子约束下做过针对性设计。那是研究工作，
不是集成工作。

EASU + RCAS 走的是另一条路：**不生成细节，但把已有细节重建和强化到位**。
代价是达不到 GAN 的纹理质感，收益是零依赖、全硬件覆盖、不假造纹理，
以及上面那个 450 倍的性能差。

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
角标解决这个问题 —— 它显示在视频**右上角**：

- **简洁**：`VidSharp 1440p`（放大时显示目标高度，未放大时只显示 `VidSharp`）
- **详细参数**：`VidSharp · 1280×720 → 2560×1440 · 锐化 50 / 去块 40 / …`
- **关闭**：不显示

**角标是画进 canvas 的，不是独立 DOM 元素。** 这是刻意的设计：

独立 DOM 角标与 canvas 是两个元素，会被站点的不同层级分别遮挡，可能出现
「角标可见但 canvas 被挡住」的情况 —— 那时角标就成了误导。合成进画面后，
**看得见角标 ⟺ 看得见增强结果**，两者物理上无法解耦。

代价是截图/录屏会带上角标。需要干净画面时把它切到「关闭」。

文字用 2D canvas 预渲染成纹理，仅在文案或分辨率变化时重建；每帧只多一次
alpha 混合。角标关闭时管线少一个 pass，RCAS 直接输出到 canvas。

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

上面的数据是在 Apple M1 上实测的。若想在**你自己的 GPU** 上复核：

```bash
npm install
npm run fetch-models   # 下载两个 ONNX 模型（约 5MB，不入 Git）
npm run probe
```

然后在 Chrome / Edge 中打开 `http://localhost:8777/`。三个按钮：

| 按钮 | 内容 |
|---|---|
| **基础测试** | 两个模型在 WebGPU 上的加载/warmup/单块耗时、1080p 分块数与单帧估算 |
| **完整诊断** | 基础 + 算子回退检测 + 批处理 + 并行提交 + **纯 shader 参照** |
| **加测 WASM** | 同时跑 WASM EP 作对照 |

「完整诊断」里最有价值的是**纯 shader 参照** —— 它用本项目实际的
`easu.wgsl` 跑 1080p→2K/4K，给出同一块 GPU 在 shader 负载下的能力基线。
CNN 与它的差距即为模型本身的代价，与 GPU 强弱无关。

长流程可随时点「中止」，已跑出的结果会保留。

**注意**：模型与 `onnxruntime-web` 依赖**仅供此探针使用**，扩展本体完全不
需要它们 —— 装扩展不必 `npm install`。`models/` 已在 `.gitignore` 中。

### 探针踩过的坑（已在代码中处理）

- `ort.env.wasm.wasmPaths` **必须是绝对 URL**。相对路径会被拼到模块自身
  目录下，变成 `.../onnxruntime-web/node_modules/onnxruntime-web/dist/...`。
- `ort.env.wasm.numThreads = 1`。多线程依赖 blob worker，在扩展/严格 CSP
  环境下会踩 `URL.createObjectURL is not a function`。
- **同一个 session 不能并发调用 `run()`** —— 会死锁（不是串行化）。并发测试
  必须为每路建独立 session。
- **`session.release()` 返回 Promise，必须 await** —— 否则 WASM 堆还没回收
  就创建下一个 session，反复几轮后 `memory access out of bounds`。
- ORT Web 在多 session 场景下有状态泄漏，会出现
  `Cannot read properties of undefined (reading 'getBindGroupLayout')`。
  重型模型的诊断因此限制了迭代次数与 batch 上限。

## 已知限制

- **DRM 视频无法处理**（Netflix、Disney+ 等）。受 EME 保护的解码帧在受保护
  显存中，`importExternalTexture` 取不到像素。扩展会在连续失败 5 帧后自动
  停止并提示。
- 需要 WebGPU。不支持的浏览器会在弹窗中提示。
- 放大倍数受 GPU 纹理上限约束（Apple M1 为 8192）。超限时自动钳制。
- 覆盖层挂在 video 的紧后方，继承其层级。极少数站点若有异常的 DOM 结构，
  可能出现层级异常。

## 性能

Apple M1 实测（EASU pass，`npm run probe` → 完整诊断）：

| 输出 | 单帧 | 理论 fps |
|---|---|---|
| 1080p → 2K | 0.88 ms | 1133 |
| 1080p → 4K | 1.63 ms | 614 |

即便 60fps 视频每帧预算 16.7ms，这个开销也只占约 5%~10%。加上 enhance
与 RCAS 两个 pass 后仍有大量余量。

若遇到掉帧：

1. 超分改为「关闭」或「2倍」（4K 输出的填充率压力最大）
2. 去色带调 0（它每帧更新 uniform 以变化抖动图案）
3. 角标切「关闭」（可省一个 pass）

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
- badge 矩形内确实合成、**矩形外逐位不变**、opacity=0 时恒等
- 强度映射端点与单调性（回归测试，防止再次写错映射曲线）
- 输出尺寸计算 9 个用例（含纹理上限钳制）
- 角标文案生成 5 个用例（纯 JS 逻辑，shader 测试覆盖不到）

共 18 项。

## 文件结构

```
manifest.json          MV3 配置
src/
  enhance.wgsl         去块 + 去色带 + 局部对比度
  easu.wgsl            EASU 方向自适应放大
  rcas.wgsl            RCAS 自适应锐化
  badge.wgsl           状态角标合成
  content.js           多 pass 管线 + 取帧 + 角标 + 拖动对比
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
