# VidSharp

用 WebGPU 对网页视频做 **RCAS 自适应锐化**。真人与动画内容通用，单 pass，开销极低。

## 为什么是 RCAS 而不是神经网络超分

调研后的结论：**2026 年在浏览器里用神经网络做 1080p 真人实时超分不成立**。

- Edge 内置 VSR 用原生 DirectML，RTX 2060 上 360p 就要约 20ms/帧，且把触发条件限死在 <720p
- 浏览器 WebGPU 只会更紧
- NVIDIA / Intel / 微软都因此把方案放在了驱动层，且都不开放 API 给第三方
- 现有浏览器扩展基本都以 Anime4K 为核心，对真人内容效果有限

RCAS 走的是另一条路：**不生成细节，只把已有细节的对比推到位**。代价是达不到 GAN 的纹理质感，收益是：

- 覆盖 100% 支持 WebGPU 的硬件，无模型下载
- 单 pass fragment shader，开销可忽略
- **对真人内容比轻量 CNN 更安全** —— 不假造纹理，不会把皮肤推成塑料感

## RCAS 相比朴素锐化（unsharp mask）好在哪

移植自 AMD FidelityFX 的 `FsrRcasF`，两个机制是关键：

1. **限幅（lobe limiter）** —— 依据 3x3 十字邻域的极值反推每个通道允许的最大锐化量，从原理上避免 ringing 和过冲，不会把边缘推出色域。
2. **噪声抑制（denoise）** —— 用局部动态范围归一化后的偏差判断「这里是噪点还是真实结构」，噪点区自动减弱。压缩视频的块噪声因此不会被放大。

常量与公式已逐项对照 [ffx_fsr1.h](https://github.com/GPUOpen-Effects/FidelityFX-FSR) 原始实现核验。

## 安装

需要 Chrome / Edge 113+（WebGPU）。

1. 打开 `chrome://extensions`
2. 开启右上角「开发者模式」
3. 点「加载已解压的扩展程序」，选择本目录
4. 点工具栏图标，打开「启用锐化」

## 使用

| 选项 | 说明 |
|---|---|
| 启用锐化 | 全局开关，对所有站点生效 |
| 强度 | 0~100，映射到 FSR sharpness stops（2.0 → 0.0） |
| 噪声抑制 | 建议保持开启，尤其是低码率视频 |

设置改动立即生效，无需刷新页面。

## 已知限制

- **DRM 视频无法处理**（Netflix、Disney+ 等）。受 EME 保护的视频解码帧在受保护显存里，`importExternalTexture` 取不到像素。扩展会在连续失败 5 帧后自动停止并提示。
- 需要 WebGPU。不支持的浏览器会在弹窗中提示。
- 覆盖层会挂在 video 的父元素下。极少数站点若有异常的定位/层级结构，可能出现遮挡。

## 开发

```bash
npm install
npm test     # 用软件 WebGPU 后端验证 shader 数学正确性
```

测试会启动带 SwiftShader 的 Chromium，把 `rcas.wgsl` 里的 `texture_external`
替换成 `texture_2d` 以便注入已知像素，然后验证：

- sharpness=0 时恒等（RCAS 特性：lobe=0 → 输出=输入）
- 锐化后边缘对比不降低
- 平坦区不被改动（纯色区 range≈0 的除法兜底不产生 NaN）
- 噪声抑制开启后锐化量不高于关闭时

## 文件结构

```
manifest.json          MV3 配置
src/
  rcas.wgsl            RCAS shader（核心算法）
  content.js           取帧渲染管线
  content.css          覆盖层样式
  popup.html/js        控制面板
test/
  rcas.test.html       shader 单元测试
  run.js               测试驱动（起 HTTP server + 软件 WebGPU）
```

## 后续可加

- 轻度去块滤波（针对低码率源的块效应）
- 局部对比度增强
- 按站点白名单
- 轻量 CNN 作为可选高质量档（ESPCN / eSR / SPAN-tiny 级别，参数量对齐 Edge VSR 的 0.1~1MB）

## 许可

RCAS 算法源自 AMD FidelityFX（MIT），版权归 AMD。
