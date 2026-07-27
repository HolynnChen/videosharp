// 预处理增强：去块 + 去色带 + 局部对比度
//
// 这一 pass 跑在 EASU 放大之前 —— 顺序很重要：压缩伪影必须在放大前压掉，
// 否则 EASU 会把块边界当成真实边缘去"保护"，反而让块效应更明显锐利。
//
// 三个效果都在源分辨率上做，共用一次 3x3 邻域采样。

struct Params {
    // 去块强度 0~1，0 = 关闭
    deblock: f32,
    // 去色带强度 0~1，0 = 关闭
    deband: f32,
    // 局部对比度强度 0~1，0 = 关闭
    contrast: f32,
    // 帧序号，用于让去色带的抖动随时间变化，避免固定噪点图案
    frameSeed: f32,
};

@group(0) @binding(0) var srcTex: texture_external;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
    var out: VertexOut;
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(idx & 2u) * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleBaseClampToEdge(srcTex, srcSampler, uv).rgb;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// 无状态伪随机，用于去色带抖动
fn hash(p: vec2<f32>) -> f32 {
    var v = fract(p * vec2<f32>(0.1031, 0.1030));
    v += dot(v, v.yx + 33.33);
    return fract((v.x + v.y) * v.x);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(srcTex));
    let texel = 1.0 / dims;
    let pixelPos = in.uv * dims;

    var color = tap(in.uv);

    // 3x3 邻域，去块与对比度共用
    let n  = tap(in.uv + vec2<f32>( 0.0, -texel.y));
    let s  = tap(in.uv + vec2<f32>( 0.0,  texel.y));
    let w  = tap(in.uv + vec2<f32>(-texel.x, 0.0));
    let e  = tap(in.uv + vec2<f32>( texel.x, 0.0));

    // ================= 去块 =================
    // 思路：H.264/HEVC 的块边界固定落在 8 的倍数像素上。只在这些位置做
    // 方向性平滑，且仅当跳变"小到不像真实边缘、大到不像噪声"时才处理 ——
    // 真实边缘的跳变远大于块效应，用上限阈值把它们排除在外。
    if (params.deblock > 0.001) {
        let onGridX = abs(fract(pixelPos.x / 8.0)) < 0.15;
        let onGridY = abs(fract(pixelPos.y / 8.0)) < 0.15;

        let lc = luma(color);
        var smoothed = color;
        var applied = false;

        if (onGridX) {
            let dw = abs(lc - luma(w));
            let de = abs(lc - luma(e));
            let jump = max(dw, de);
            // 0.06 以下认为是块效应，0.25 以上认为是真实边缘
            if (jump > 0.004 && jump < 0.25) {
                smoothed = mix(smoothed, (w + color + e) / 3.0, 0.65);
                applied = true;
            }
        }
        if (onGridY) {
            let dn = abs(lc - luma(n));
            let ds = abs(lc - luma(s));
            let jump = max(dn, ds);
            if (jump > 0.004 && jump < 0.25) {
                smoothed = mix(smoothed, (n + color + s) / 3.0, 0.65);
                applied = true;
            }
        }
        if (applied) {
            color = mix(color, smoothed, params.deblock);
        }
    }

    // ================= 去色带 =================
    // 色带出现在平滑渐变区（8bit 量化台阶）。判据：局部动态范围极小
    // 但确实存在非零梯度 —— 纯色区不需要处理，纹理区不能处理。
    // 处理方式是加入亚量化级的三角分布抖动，把硬台阶打散成视觉噪声。
    if (params.deband > 0.001) {
        let lc = luma(color);
        let ln = luma(n); let ls = luma(s);
        let lw = luma(w); let le = luma(e);
        let localMax = max(max(ln, ls), max(lw, le));
        let localMin = min(min(ln, ls), min(lw, le));
        let range = localMax - localMin;

        // 1/255 ≈ 0.0039，色带台阶通常在 1~3 个量化级
        let isFlat = range < 0.05 && range > 0.0;
        if (isFlat) {
            // 三角分布抖动（两个均匀分布之差），比单个均匀分布更接近理想抖动
            let seed = pixelPos + vec2<f32>(params.frameSeed * 7.13);
            let r1 = hash(seed);
            let r2 = hash(seed + vec2<f32>(37.7, 17.3));
            let dither = (r1 - r2) * (1.0 / 255.0) * 1.5;
            // 渐变越平滑，抖动给得越足
            let strength = params.deband * (1.0 - range / 0.05);
            color += vec3<f32>(dither * strength);
        }
    }

    // ================= 局部对比度 =================
    // 用中心与邻域均值之差作为局部细节分量，非线性放大。
    // 与锐化的区别：作用尺度更大（关注区域明暗关系而非边缘），
    // 且用 smoothstep 抑制强边缘处的增益，避免和后续 RCAS 叠加过冲。
    if (params.contrast > 0.001) {
        let localMean = (n + s + w + e) * 0.25;
        let detail = color - localMean;
        let detailLuma = abs(luma(detail));
        // 强边缘（detailLuma 大）处衰减，平坦区全额增益
        let falloff = 1.0 - smoothstep(0.08, 0.3, detailLuma);
        color += detail * params.contrast * 0.6 * falloff;
    }

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
