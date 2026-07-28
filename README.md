# VidSharp

用 WebGPU 对网页视频做**超分辨率 + 画质修复 + 锐化**。真人与动画内容通用，
纯 shader 实现，无模型下载。

## 处理管线

```
<video> ──importExternalTexture──▶
   [1] enhance      去块 + 去色带 + 局部对比度   (源分辨率)
   [2] 放大         EASU / RAVU / XLSR           (→ 2K / 4K / 2x / 4x)
   [3] RCAS         自适应锐化
   [4] grain        合成胶片颗粒（可选）
   [5] badge        状态角标合成（可选）
 ──▶ <canvas>
```

**顺序是有讲究的**：

- **去块必须在放大前** —— 否则 EASU 会把块边界当成真实边缘去「保护」，
  压缩伪影反而被强化。
- **锐化必须在放大后** —— 放大前锐化，细节会被插值重新糊掉。
- **颗粒必须在锐化后** —— 否则 RCAS 会把颗粒当细节放大成噪点。

中间纹理用 `rgba16float`：多 pass 串联时避免 8bit 量化误差累积，
去色带的亚量化级抖动在 8bit 中间纹理里会被直接抹掉。

## 关于神经网络超分

**结论有前提，且被实测修正过两次。** Apple M1（Metal 3）实测：

### 分水岭不是参数量，而是能否一次前向

| 模型 | 方式 | 1080p 单帧 | fps |
|---|---|---|---|
| **EASU（纯 shader）** | 一次 pass | **0.94 ms** | 1065 |
| **XLSR（28K 参数）** | **全卷积，一次前向** | **28.3 ms** | **35.4** |
| ESPCN（固定 224² 输入） | 分 45 块 | 372 ms | 2.7 |
| realesr-general-x4v3（固定 128²） | 分 135 块 | 34369 ms | 0.03 |

XLSR 与 ESPCN 同为 CNN，**差 13 倍** —— 差别只在「一次前向」vs「分 45 块」。
批处理增益仅 **1.05×** 反证了瓶颈是分块调度而非算力。

所以准确的结论是：**分块的 CNN 不可行；全卷积单次前向的 CNN 能做到 30fps。**
本项目据此加入了 `XLSR` 实验性档位（见下）。

但也要看清代价：XLSR 比 EASU **慢约 30 倍**，28.3ms 已占满 60fps 的
16.7ms 预算（169%），只够 30fps 内容。默认档仍是 EASU。

### 原始对比数据

同一块 GPU，纯 shader 比分块 CNN 快约 450 倍

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
| RAVU | 结构分类 + 预训练系数查表 | 同上，但插值核来自真实数据拟合，更锐 |
| RCAS | 带限幅与噪声抑制的锐化 | 画面整体偏软 |
| 去块 | 8 像素网格上的方向性平滑 | 低码率视频的方块伪影 |
| 去色带 | 渐变区三角分布抖动 | 天空、渐变面上的台阶条纹 |
| 局部对比度 | 邻域均值差非线性放大 | 暗部细节丢失、画面发闷 |
| **胶片颗粒** | 亮度自适应合成噪声 | **唯一真正「增加」细节的手段**，见下 |

EASU / RCAS 的常量与公式已逐项对照
[ffx_fsr1.h](https://github.com/GPUOpen-Effects/FidelityFX-FSR) 原始实现核验；
RAVU 移植自 [bjin/mpv-prescalers](https://github.com/bjin/mpv-prescalers)。

### 重建 vs 生成：一个重要区别

EASU / RAVU / RCAS / Anime4K 全都是**重建**：让已有边缘更准更锐，但**不增加
任何原本不存在的信息**。压缩视频丢掉的高频纹理（皮肤、织物、树叶的细微起伏）
它们变不回来 —— 数学上不可能。

电影工业的做法反过来：既然真实纹理找不回，就注入结构合理的合成噪声，让眼睛
重新感知到「表面有质感」。AV1 规范把这件事标准化为 film grain synthesis
（编码时去噪省码率、解码时按参数合成回来）。

`grain.wgsl` 就是这个思路，三条约束区别于朴素「加噪点」：

1. **亮度自适应** —— 0.35 附近达峰，纯黑与高光几乎不加（否则显脏）
2. **平坦区优先** —— 已有丰富纹理处衰减到 0.35 倍，避免与真实细节打架
3. **时域去相关** —— 每帧图案变化，否则固定噪点会被看成屏幕脏污

幅度基准 3/255（约 3 个量化级），实测中间调 ±3、纯黑与高光偏差 0。

### RAVU 与 EASU 的区别

EASU 在**运行时推断**该怎么插值（估边缘方向 → 旋转采样核）。
RAVU 把这件事搬到**离线**：训练阶段对大量图像块做分类回归，把
(方向 24 档 × 强度 4 档 × 相干性 3 档) = 288 类各自的最优滤波系数预先算好
存进 LUT。运行时只做「算结构 → 查表 → 加权求和」。

代价是多一张 39KB 的 LUT 与 25 次采样；收益是插值核来自真实数据拟合而非
解析近似，锐度更高 —— 相当于用查表换掉了 CNN 的卷积，训练成本已在离线付掉。

**限制**：RAVU-Lite 的 LUT 按 2x 子像素布局训练，因此**仅在放大倍率恰为 2 时
生效**，其余倍率自动回落 EASU。强行用于非 2x 会让子像素索引错位，产生网格伪影。

### XLSR CNN 档（实验性）

高通的 28K 参数超分模型，是三个放大器中**唯一带学习先验、能"生成"细节**的。

启用只需把「放大算法」选 `XLSR CNN`，**任意目标倍率都支持**。

角标「详细参数」显示的是**实际生效**的放大器（`EASU` / `RAVU` / `XLSR`），
不是设置值 —— 想确认有没有生效看这里最准。

### 两段式：如何支持任意倍率

XLSR 的末端是 `DepthToSpace`，把 48 个通道重排成 4×4 空间像素 —— **4 是编译
进权重的**，模型只能做 4 倍。RAVU（LUT 按 2x 布局）、Anime4K 也都是固定倍率，
这类模型都是「为特定倍率训练」的。

解法是两段式：**先把画面缩放到目标尺寸 ÷ 4，再让 CNN 放大 4 倍。**

```
1080p ──downscale──▶ 640×360 ──CNN 4x──▶ 2560×1440   (正好 2K)
```

关键收益是 **CNN 的负担不再随源分辨率膨胀**：

| 目标 | CNN 输入 | 相对开销 |
|---|---|---|
| 1080p → 2K（两段式） | 640×360 | **1×** |
| 1080p 直接 4 倍（旧行为） | 1920×1080 | **16×** ← 会卡顿 |

同样得到高分辨率输出，开销差 16 倍。而且先缩小更符合模型的训练分布 ——
它本就是在低分辨率输入上训练的。

预缩放用专门的 `downscale.wgsl` 而非双线性：缩放比超过 2 倍时双线性只取
2×2 邻域会漏采样，产生摩尔纹，而视频上每帧走样位置不同会形成闪烁。
它按缩放比动态决定采样半径做 Lanczos 加权区域平均（实测把逐像素棋盘图
平均成 127~127 的均匀中灰，走样完全抑制）。

### 何时值得用 XLSR

CNN 的价值在于用学习先验「生成」纹理，所以**源画质越差收益越大**：

| 源 | 评价 |
|---|---|
| 360p / 480p 老片 | ✅ 最适合，先验有发挥空间 |
| 低码率 1080p | ⚠️ 有一定改善，但代价高 |
| 高码率 1080p | ❌ 信息本就完整，先验帮不上忙 |

对**码率不足**的 1080p（如 B 站普通画质），更有效的往往是去块 / 去色带 /
局部对比度 —— 那才是对症的手段。此时「超分辨率」甚至可以选 `关闭`，
把算力全用在修复上。

代价：
- **28.3ms/帧**（M1），只够 30fps 内容，60fps 会掉帧
- 首次启用需加载 26MB 的 ORT wasm（已打包进扩展，离线可用）
- 仅在选中该档时才加载 —— 默认档完全不碰它

技术要点：模型原始 ONNX 的输入是硬编码 `[1,3,128,128]`，但它是纯卷积
（`Conv/Relu/Clip/Concat/DepthToSpace`，**无 `Reshape`**），空间维本可任意 ——
128 只是导出时的样例尺寸。把输入维改为 `dim_param` 后即可一次前向处理整帧，
这正是它比 ESPCN 快 13 倍的原因。

管线上 CNN 需要 NCHW 平面布局的 GPUBuffer，而 WebGPU 是 RGBA 交错纹理，
故有 `to-tensor.wgsl` / `from-tensor.wgsl` 两个桥接 shader。这一步必须留在
GPU 上 —— 1080p 经 CPU 往返仅双向拷贝就有 24MB，会把 28ms 推到 100ms+。

**device 必须共享。** WebGPU 规定 buffer/纹理只能用于创建它的 device，而
ORT 1.27 会自行 `requestDevice()` 且不接受外部传入。混用会报：

```
[Buffer "cnn-in"] is associated with [Device], and cannot be used with [Device]
```

解决办法是反过来 —— 先创建一次 session 让 ORT 初始化好 device，再通过
`ort.env.webgpu.device` 取出来给整条渲染管线用。代价是**切换 CNN 档时整个
会话要重建**（device 换了，所有 pipeline / 纹理 / canvas context 都失效），
换来的是全程零拷贝。

## 安装

需要 Chrome / Edge 113+（WebGPU）。

1. 打开 `chrome://extensions`
2. 开启右上角「开发者模式」
3. 点「加载已解压的扩展程序」，选择本目录
4. 点工具栏图标，打开总开关

## 参数说明

| 选项 | 默认 | 说明 |
|---|---|---|
| 超分辨率 | 放大到 2K | `关闭` / `2K` / `4K` / `2倍` / `4倍`。已达目标分辨率的视频不会被放大 |
| 放大算法 | EASU | `EASU`（通用）/ `RAVU`（更锐，仅 2x）/ `XLSR CNN`（实验性，仅 4x） |
| 锐化强度 | 50 | 0 = 无锐化 |
| 噪声抑制 | 开 | 低码率视频建议保持开启 |
| 去块 | 40 | 低码率视频调高，高码率源可关掉 |
| 去色带 | 30 | 天空、纯色背景多的内容调高 |
| 局部对比度 | 25 | 调过头会发「HDR 味」 |
| 胶片颗粒 | 0 | 唯一能增加细节感的选项。真人建议 20~30，动画慎用 |
| 颗粒粗细 | 1 | 1 = 逐像素，越大颗粒越粗 |
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

## MSE 流探针（开发者选项，默认关闭）

用于评估「预超分」方案的可行性 —— 即**提前**拦截视频流处理好、播放时直接用，
以摆脱每帧 16~33ms 的实时约束。

popup 底部「开发者 → MSE 流探针」打开，刷新页面，播放视频后看控制台。
它会输出截获的分片数、codec、以及两道关卡的结论。

### 为什么需要探针而不直接做

调研已确认三条路彻底堵死：

| 路径 | 判定 | 根因 |
|---|---|---|
| 网络层拦截改流 | ❌ | MV3 的 `webRequest`/`declarativeNetRequest` **不能读写响应体** |
| 读 `<video>` 已缓冲帧 | ❌ | `importExternalTexture` / `createImageBitmap` / rVFC **只给当前帧**；浏览器缓冲的是压缩比特流，不是像素 |
| 双 video 副本 | ❌ | MSE 的 `blob:` URL **绑定到创建它的 MediaSource**，无法共享 |

唯一剩下的是 **hook `SourceBuffer.prototype.appendBuffer`**。探针验证的就是这条。

### 即便截到数据，仍有两道硬约束

**1. 必须自己接管播放。** 超分结果不回填（回填要重编码，画质二次损失且开销
可能超过超分本身），所以得自建播放器：fMP4 解析、音视频同步、seek、缓冲管理、
弹幕层对齐。调研评估工作量以**人月**计，且要持续对抗站点改版。

**2. 显存撑不住缓存。** 超分后是未压缩像素：

| 缓存时长 | 2K | 4K |
|---|---|---|
| 2 秒 | 0.82 GB | 1.85 GB |
| 5 秒 | 2.06 GB | 4.63 GB |

浏览器单页显存上限通常几百 MB ~ 2GB。2 秒已在边缘，而 2 秒的提前量**不足以
换用更重的模型** —— 仍需维持「平均处理速度 ≥ 播放速度」，只能吸收瞬时抖动。

### 一个很强的旁证

GitHub 上**没有任何项目做过浏览器端预超分**。所有方案（Anime4K-WebExtension
204★、YouTube-VSR、VSR-Bench）都是实时抓当前帧。NVIDIA RTX VSR 与 Edge VSR
也是实时逐帧 —— 靠专用硬件把单帧压进帧预算，而非预处理。

技术细节：探针需两个脚本。MAIN world 的那个才能改到页面的
`SourceBuffer.prototype`（ISOLATED world 里改的是另一个对象，对页面无效），
但它拿不到 `chrome.*` API，只能 `postMessage` 给 ISOLATED world 的伴生脚本。
且必须 `run_at: document_start` —— 页面脚本一旦先持有引用，补丁就绕不过去。

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
npm test     # manifest 一致性检查 + shader 验证
npm run check # 只跑 manifest 检查（快，不需要浏览器）
```

`check-manifest.js` 抓的是一类反复出现的低级错误：新增 shader 后忘了加到
`web_accessible_resources`，扩展一跑就 `TypeError: Failed to fetch`。这种错误
**只在真实扩展环境暴露**（本地 HTTP 测试照样能加载），所以必须靠静态核对。
它做双向检查：代码 `getURL()` 引用的资源必须已声明，声明的资源文件必须存在。

测试会启动带 SwiftShader 的 Chromium，把 `texture_external` 替换成
`texture_2d` 以便注入已知像素，然后回读结果验证：

- 八个 shader 均能编译
- EASU 保留动态范围、斜边过渡单调、输出有空间变化
- RAVU LUT 尺寸正确、保留动态范围、**平坦区逐位保真**、无偏色
- enhance 全部关闭时恒等、纯色区不被破坏、去色带不污染纯色区
- RCAS `sharpness=0` 时恒等、锐化后边缘对比不降低
- grain `strength=0` 恒等、中间调确有颗粒、幅度 ≤6 量化级、纯黑与高光不加
- badge 矩形内确实合成、**矩形外逐位不变**、opacity=0 时恒等
- 强度映射端点与单调性（回归测试，防止再次写错映射曲线）
- 输出尺寸计算 9 个用例（含纹理上限钳制）
- downscale 抑制走样（棋盘图平均为 127~127）、等比时保留原对比
- **桥接 shader 往返恒等**（texture→NCHW→texture，误差 0）—— CNN 档的正确性
  完全依赖它，布局算错会导致通道错位或水平撕裂，而这类错误编译期不报错
- 角标文案生成 7 个用例（纯 JS 逻辑，shader 测试覆盖不到）
- 输出尺寸计算 11 个用例

共 40 项。

其中「RAVU 平坦区逐位保真」抓到过一个真实 bug：原版 anti-ringing 用
`(0.1+v)^33 / (0.1+v)^32` 逼近软极值，但 v≈0.157 时 `0.257^32 ≈ 1e-19`
在 fp32 下严重下溢，导致钳制区间反转、输出变成 -0.049。现改用直接的硬极值
（只在正权重覆盖的采样点取 min/max），语义相同但数值稳定，且更快。

## 文件结构

```
manifest.json          MV3 配置
src/
  background.js        service worker（按需注册 MSE 探针）
  mse-probe-main.js    MSE hook（MAIN world）
  mse-probe.js         探针伴生脚本（ISOLATED world）
  enhance.wgsl         去块 + 去色带 + 局部对比度
  easu.wgsl            EASU 方向自适应放大
  ravu.wgsl            RAVU 查表法放大（更锐，仅 2x）
  data/ravu-lut.js     RAVU 预训练系数表（39KB base64）
  downscale.wgsl       高质量降采样（CNN 两段式的预缩放）
  to-tensor.wgsl       纹理 → NCHW 张量（CNN 输入桥接）
  from-tensor.wgsl     NCHW 张量 → 纹理（CNN 输出桥接）
  cnn.js               XLSR 推理封装（ORT session 管理与降级）
  rcas.wgsl            RCAS 自适应锐化
  grain.wgsl           合成胶片颗粒
  badge.wgsl           状态角标合成
  content.js           多 pass 管线 + 取帧 + 角标 + 拖动对比
  content.css          覆盖层与分割条样式
  popup.html/js        控制面板
vendor/ort/            ORT WebGPU 运行时（26MB，仅 CNN 档需要）
models/
  xlsr-dynamic.onnx    XLSR 权重，输入空间维已放宽为动态
test/
  check-manifest.js    manifest 与文件一致性检查
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
