// RAVU-Lite-AR (r3) — WGSL 移植
//
// 移植自 bjin/mpv-prescalers 的 ravu-lite-ar-r3.hook（LGPL-3.0）。
// RAVU 源于 Google 的 RAISR 思路。
//
// 与 EASU 的本质区别：
//   EASU 在运行时"推断"该怎么插值（估边缘方向 → 旋转采样核）。
//   RAVU 把这件事搬到离线：训练阶段对大量图像块做分类回归，把
//   (方向 24 档 × 强度 4 档 × 相干性 3 档) = 288 类各自的最优滤波系数
//   预先算好存进 LUT。运行时只做"算结构 → 查表 → 加权求和"。
//
//   代价是多一张 39KB 的 LUT 纹理与 25 次采样；收益是插值核来自真实数据
//   拟合而非解析近似，锐度与保真度都更高 —— 相当于用查表换掉了 CNN 的
//   卷积，训练成本已在离线阶段付掉。
//
// 固定 2x 放大：每个源像素产出 4 个子像素。LUT 的 RGBA 四通道正好对应
// 左上/左下/右上/右下，故单 pass 即可完成（原版分两步是受 mpv 钩子模型
// 限制，这里无此约束）。
//
// AR = anti-ringing：用局部软最大/最小对结果做钳制，抑制过冲。

// 5x5 邻域，利用中心对称只需存一半 + 中心 = 13 组系数
const LUT_W: f32 = 13.0;
const LUT_H: f32 = 288.0;

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var lutTex: texture_2d<f32>;

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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// 保持 sampler 被实际使用，否则 layout:"auto" 会把它从 bind group layout
// 里优化掉，导致后续 binding 编号错位（曾表现为 textureDimensions 返回 0）。
fn keepSampler() -> f32 {
    return textureSampleLevel(srcTex, srcSampler, vec2<f32>(0.5), 0.0).a * 0.0;
}

// LUT 用 NEAREST 语义取样：行 = 结构类别，列 = 采样位置
fn lutFetch(col: i32, row: i32) -> vec4<f32> {
    return textureLoad(lutTex, vec2<i32>(col, row), 0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let srcSize = vec2<f32>(textureDimensions(srcTex));

    // 输出像素映射回源像素网格。RAVU-Lite 固定 2x。
    // 原版分两步：step1 对每个源像素算出 4 个子像素存进 RGBA；step2 用
    //   dir = fract(pos * size) - 0.5;  idx = (dir.x>0)*2 + (dir.y>0)
    // 从 -dir 偏移处取对应通道。等价的单 pass 写法是：先找到当前输出像素
    // 所属的源像素中心，再由输出像素在该 2x2 内的位置选通道。
    //
    // 2x 放大下，输出像素 (ox,oy) 属于源像素 floor(ox/2)，
    // 子像素位置由 ox&1 / oy&1 决定。
    let outPos = in.uv * srcSize * 2.0;
    let srcIdx = floor(outPos * 0.5);
    let sub = outPos - srcIdx * 2.0;      // 各分量落在 [0,2)
    // 通道顺序与原版一致：x 大贡献 2，y 大贡献 1
    let subIdx = i32(sub.x >= 1.0) * 2 + i32(sub.y >= 1.0);
    // 源像素中心（用于取 5x5 邻域）
    let center = srcIdx + 0.5;

    // ---- 取 5x5 邻域亮度 ----
    // 必须用 textureLoad 精确取整数网格像素：RAVU 的 LUT 系数是按"离散
    // 5x5 像素邻域"训练的，若用带 linear 过滤的 textureSampleLevel，
    // 采样点落在像素边界会被插值，等于喂进了模糊后的邻域，结果整体偏暗/发灰。
    // 这个 bug 表现为平坦区偏离输入达 40 个灰阶。
    let maxXY = vec2<i32>(srcSize) - vec2<i32>(1);
    var L: array<f32, 25>;
    for (var j = 0; j < 5; j++) {
        for (var i = 0; i < 5; i++) {
            let c = clamp(
                vec2<i32>(srcIdx) + vec2<i32>(i - 2, j - 2),
                vec2<i32>(0), maxXY,
            );
            L[i * 5 + j] = luma(textureLoad(srcTex, c, 0).rgb);
        }
    }

    // ---- 结构张量（3x3 内部窗口，高斯加权）----
    // abd = [Σw·gx², Σw·gx·gy, Σw·gy²]，梯度用中心差分
    var abd = vec3<f32>(0.0);
    let W = array<f32, 9>(
        0.1018680644198163,  0.11543163961422666, 0.1018680644198163,
        0.11543163961422666, 0.13080118386382833, 0.11543163961422666,
        0.1018680644198163,  0.11543163961422666, 0.1018680644198163,
    );
    for (var j = 1; j < 4; j++) {
        for (var i = 1; i < 4; i++) {
            let gx = (L[(i + 1) * 5 + j] - L[(i - 1) * 5 + j]) / 2.0;
            let gy = (L[i * 5 + (j + 1)] - L[i * 5 + (j - 1)]) / 2.0;
            let w = W[(i - 1) * 3 + (j - 1)];
            abd += vec3<f32>(gx * gx, gx * gy, gy * gy) * w;
        }
    }

    // ---- 特征值分解 → (方向, 强度, 相干性) ----
    let a = abd.x; let b = abd.y; let d = abd.z;
    let T = a + d;
    let D = a * d - b * b;
    let delta = sqrt(max(T * T / 4.0 - D, 0.0));
    let L1 = T / 2.0 + delta;
    let L2 = T / 2.0 - delta;
    let sqrtL1 = sqrt(max(L1, 0.0));
    let sqrtL2 = sqrt(max(L2, 0.0));

    let PI = 3.141592653589793;
    // b 接近 0 时 atan 不稳定，退化为 0（与原版的 mix 判据一致）。
    // atan2 值域 (-PI, PI]，加 PI 后落在 (0, 2PI]，再对 PI 取模归入 [0, PI)。
    // 不用 % 运算符：WGSL 的 % 对负操作数会保留符号，这里先加 PI 保证非负
    // 再用 fract 更稳妥。
    var theta = 0.0;
    if (abs(b) >= 1.192092896e-7) {
        let raw = atan2(L1 - a, b) + PI;
        theta = fract(raw / PI) * PI;
    }
    let lambda = sqrtL1;
    var mu = 0.0;
    if (sqrtL1 + sqrtL2 >= 1.192092896e-7) {
        mu = (sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2);
    }

    // 量化到 24 / 4 / 3 档
    let angle = floor(theta * 24.0 / PI);
    var strength = 0.0;
    if (lambda >= 0.05)       { strength = 3.0; }
    else if (lambda >= 0.016) { strength = 2.0; }
    else if (lambda >= 0.004) { strength = 1.0; }
    var coherence = 0.0;
    if (mu >= 0.5)       { coherence = 2.0; }
    else if (mu >= 0.25) { coherence = 1.0; }

    let row = i32((angle * 4.0 + strength) * 3.0 + coherence);

    // ---- 查表加权求和 ----
    // 13 组系数覆盖 25 个采样点：前 12 组各配对两个对称位置
    // （w 与 w.wzyx 互换通道以复用），第 13 组为中心点。
    var res = vec4<f32>(0.0);
    // anti-ringing 的邻域极值。
    //
    // 原版用 (0.1+v)^33 / (0.1+v)^32 这种"软极值"技巧逼近 max/min。
    // 移植时踩了坑：v≈0.157 时 0.257^32 ≈ 1e-19，fp32 下严重下溢，
    // hi 变成 ~1.7e-19，hiV = hi2/hi - 0.1 退化成 -0.1，于是
    // clamp(res, 0.157, -0.1) 区间反转，输出垃圾（实测 -0.049）。
    //
    // 这里改用直接的硬极值：只在正权重覆盖的采样点上取 min/max。
    // 语义与软极值一致（都是"限制在邻域值域内"），但数值完全稳定，
    // 且少了 4 组 vec4 累加器与 10 次自乘，更快。
    var nbMin = 1.0;
    var nbMax = 0.0;

    for (var k = 0; k < 12; k++) {
        let w = lutFetch(k, row);
        let v1 = L[k];
        let v2 = L[24 - k];
        res += v1 * w + v2 * vec4<f32>(w.w, w.z, w.y, w.x);

        // 只有正权重的采样点才参与极值 —— 负权重是锐化的"减项"，
        // 把它们计入会让范围过宽，失去抑制过冲的作用。
        if (any(w > vec4<f32>(0.0))) {
            nbMin = min(nbMin, min(v1, v2));
            nbMax = max(nbMax, max(v1, v2));
        }
    }

    // 中心点（第 13 组，索引 12 对应 L[12]）
    {
        let w = lutFetch(12, row);
        let v = L[12];
        res += v * w;
        if (any(w > vec4<f32>(0.0))) {
            nbMin = min(nbMin, v);
            nbMax = max(nbMax, v);
        }
    }

    // ---- anti-ringing 钳制 ----
    // 0.8 的混合系数沿用原版：完全钳制会削弱锐度，完全不钳制会有振铃。
    res = mix(res, clamp(res, vec4<f32>(nbMin), vec4<f32>(nbMax)), 0.8);

    // 取当前子像素对应的通道
    var outLuma = res.x;
    if (subIdx == 1) { outLuma = res.y; }
    else if (subIdx == 2) { outLuma = res.z; }
    else if (subIdx == 3) { outLuma = res.w; }
    outLuma += keepSampler();

    // RAVU 只处理亮度。色度取所属源像素的原色，按亮度变化量平移 RGB，
    // 这样保留原色相而只提升亮度通道的锐度。
    // 用 textureLoad 与邻域取样保持一致（避免过滤引入的偏差）。
    let orig = textureLoad(srcTex, clamp(vec2<i32>(srcIdx), vec2<i32>(0), maxXY), 0).rgb;
    let origLuma = luma(orig);
    let result = orig + vec3<f32>(outLuma - origLuma);

    return vec4<f32>(clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
