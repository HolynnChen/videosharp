// RCAS (Robust Contrast Adaptive Sharpening) — WGSL 移植
//
// 源自 AMD FidelityFX Super Resolution 的 ffx_fsr1.h / FsrRcasF。
// 原始实现为 MIT 许可 (Copyright AMD)。
//
// 与朴素 unsharp mask 的区别，也是它对真人视频更安全的原因：
//   1. 限幅 (lobe limiter) —— 依据 3x3 十字邻域的极值反推最大允许锐化量，
//      从原理上避免 ringing 和过冲，不会把边缘推出色域。
//   2. 噪声抑制 (denoise) —— 用局部亮度对比度归一化后的偏差判断"这里是噪点
//      还是真实结构"，噪点区自动减弱锐化。压缩视频的块噪声因此不会被放大。
//
// 采样布局（十字，不取角点）：
//        b
//      d e f
//        h

// FSR_RCAS_LIMIT = 0.25 - (1/16)，锐化强度的硬上限
const RCAS_LIMIT: f32 = 0.1875;

struct Params {
    // 锐化强度，已在 JS 侧转换为线性值 exp2(-stops)
    sharpness: f32,
    // 是否启用噪声抑制：1.0 开 / 0.0 关
    denoise: f32,
    _pad0: f32,
    _pad1: f32,
};

// 输入是 EASU pass 的输出纹理（已放大到目标分辨率）。
// 锐化必须在放大之后做 —— 放大前锐化会被插值重新糊掉。
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// 全屏三角形，无需顶点缓冲
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
    var out: VertexOut;
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(idx & 2u) * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // NDC → UV，y 轴翻转
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(srcTex, srcSampler, uv, 0.0).rgb;
}

// 近似亮度（FSR 原式：B*0.5 + (R*0.5 + G)，相当于 2x 亮度，比例关系足够）
fn luma(c: vec3<f32>) -> f32 {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(srcTex));
    let texel = 1.0 / dims;

    let e = tap(in.uv);
    let b = tap(in.uv + vec2<f32>(0.0, -texel.y));
    let d = tap(in.uv + vec2<f32>(-texel.x, 0.0));
    let f = tap(in.uv + vec2<f32>(texel.x, 0.0));
    let h = tap(in.uv + vec2<f32>(0.0, texel.y));

    // ---- 噪声抑制 ----
    // nz 衡量中心点相对四邻域均值的偏差，再用局部动态范围归一化。
    // 偏差大但动态范围小 → 判定为噪点 → 削弱锐化。
    let bL = luma(b);
    let dL = luma(d);
    let eL = luma(e);
    let fL = luma(f);
    let hL = luma(h);

    var nz = 0.25 * (bL + dL + fL + hL) - eL;
    let range = max(max(max(bL, dL), max(fL, hL)), eL)
              - min(min(min(bL, dL), min(fL, hL)), eL);
    // range 为 0 时（纯色区）除法会出 inf，用 1e-5 兜底
    nz = clamp(abs(nz) / max(range, 1e-5), 0.0, 1.0);
    nz = -0.5 * nz + 1.0;

    // ---- 逐通道限幅 ----
    // 只看四邻域（不含中心）的极值，据此算出各通道允许的锐化 lobe。
    let mn4 = min(min(b, d), min(f, h));
    let mx4 = max(max(b, d), max(f, h));

    // peakC = (1.0, -4.0)：以 [0,1] 为目标色域上限反推余量
    let hitMin = min(mn4, e) / (4.0 * mx4);
    let hitMax = (vec3<f32>(1.0) - max(mx4, e)) / (4.0 * mn4 - 4.0);
    let lobeRGB = max(-hitMin, hitMax);

    // 取三通道中最严格的那个，保证不会有单通道越界导致偏色
    var lobe = max(-RCAS_LIMIT, min(max(max(lobeRGB.r, lobeRGB.g), lobeRGB.b), 0.0))
             * params.sharpness;

    if (params.denoise > 0.5) {
        lobe = lobe * nz;
    }

    // ---- 合成 ----
    let rcpL = 1.0 / (4.0 * lobe + 1.0);
    let outColor = ((b + d + f + h) * lobe + e) * rcpL;

    return vec4<f32>(clamp(outColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
