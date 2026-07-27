// Film grain synthesis —— 合成胶片颗粒
//
// 这是本管线中唯一真正"增加"细节的步骤。
//
// EASU / RCAS / RAVU / Anime4K 全都是"重建"：让已有边缘更准更锐，但不增加
// 任何原本不存在的信息。压缩视频丢掉的高频纹理（皮肤、织物、树叶的细微
// 起伏）它们变不回来 —— 数学上不可能。
//
// 电影工业的做法反过来：既然真实纹理找不回，就注入结构合理的合成噪声，
// 让眼睛重新感知到"表面有质感"。AV1 规范把这件事标准化为 film grain
// synthesis（编码时去噪省码率、解码时按参数合成回来）。
//
// 与"加噪点"的区别在三处约束：
//   1. 亮度自适应 —— 暗部颗粒重、亮部轻（模仿胶片银盐颗粒的响应曲线），
//      纯黑与纯白几乎不加，否则会显脏。
//   2. 平坦区优先 —— 已有丰富纹理的区域少加，避免与真实细节打架。
//   3. 时域去相关 —— 每帧图案必须变，否则固定噪点会被识别为屏幕脏污。

struct Params {
    // 颗粒强度 0~1
    strength: f32,
    // 颗粒尺寸：1.0 = 逐像素，越大颗粒越粗
    size: f32,
    // 帧序号，用于时域去相关
    frameSeed: f32,
    // 色度颗粒占亮度颗粒的比例（0 = 只加亮度颗粒，通常更自然）
    chroma: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// 整数哈希，比 fract(sin(...)) 更均匀且无平台差异
fn hash3(p: vec3<u32>) -> f32 {
    var h = p.x * 374761393u + p.y * 668265263u + p.z * 2147483647u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h) * (1.0 / 4294967296.0);
}

// 三角分布噪声（两个均匀分布之差）。相比单个均匀分布，它的能量更集中在
// 小幅度，视觉上更接近真实胶片颗粒，也更不容易产生色块感。
fn triangularNoise(cell: vec3<u32>) -> f32 {
    let a = hash3(cell);
    let b = hash3(cell + vec3<u32>(17u, 31u, 7u));
    return a - b;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(srcTex));
    let color = textureSampleLevel(srcTex, srcSampler, in.uv, 0.0).rgb;

    if (params.strength < 0.001) {
        return vec4<f32>(color, 1.0);
    }

    // 颗粒网格：size>1 时多个像素共享同一颗粒，模拟更粗的胶片
    let cellSize = max(1.0, params.size);
    let cell = floor(in.uv * dims / cellSize);
    let seed = vec3<u32>(
        u32(cell.x) & 0xFFFFu,
        u32(cell.y) & 0xFFFFu,
        u32(params.frameSeed) & 0xFFFFu,
    );

    let n = triangularNoise(seed);

    // ---- 亮度自适应 ----
    // 胶片颗粒在中间调最明显，暗部次之，高光几乎不可见。
    // 用一条在 0.15 附近达峰、向两端衰减的曲线近似这个响应。
    let l = luma(color);
    let mid = 1.0 - abs(l - 0.35) / 0.65;          // 0.35 处为 1，向两端降
    let shadowGuard = smoothstep(0.0, 0.06, l);     // 纯黑不加，否则显脏
    let highlightGuard = 1.0 - smoothstep(0.82, 1.0, l); // 高光不加
    let lumaWeight = clamp(mid, 0.0, 1.0) * shadowGuard * highlightGuard;

    // ---- 平坦区优先 ----
    // 已有丰富纹理处少加，避免与真实细节竞争。用 3x3 十字的局部方差近似
    // "纹理繁忙度"。这一步共用 4 次采样，开销很小。
    let texel = 1.0 / dims;
    let ln = luma(textureSampleLevel(srcTex, srcSampler, in.uv + vec2<f32>(0.0, -texel.y), 0.0).rgb);
    let ls = luma(textureSampleLevel(srcTex, srcSampler, in.uv + vec2<f32>(0.0,  texel.y), 0.0).rgb);
    let lw = luma(textureSampleLevel(srcTex, srcSampler, in.uv + vec2<f32>(-texel.x, 0.0), 0.0).rgb);
    let le = luma(textureSampleLevel(srcTex, srcSampler, in.uv + vec2<f32>( texel.x, 0.0), 0.0).rgb);
    let localRange = max(max(ln, ls), max(lw, le)) - min(min(ln, ls), min(lw, le));
    // range 超过 0.25 认为是强边缘/繁忙纹理，衰减到 0.35 倍
    let flatWeight = mix(1.0, 0.35, smoothstep(0.05, 0.25, localRange));

    // 基础幅度取 3/255 —— 大约是 8bit 下 3 个量化级，足以被感知为质感
    // 而不至于看成噪点。乘上强度与两个权重。
    let amp = (3.0 / 255.0) * params.strength * lumaWeight * flatWeight;

    var result = color + vec3<f32>(n * amp);

    // 色度颗粒：真实胶片的彩色颗粒较弱，默认几乎不加
    if (params.chroma > 0.001) {
        let nc1 = triangularNoise(seed + vec3<u32>(101u, 0u, 0u));
        let nc2 = triangularNoise(seed + vec3<u32>(0u, 211u, 0u));
        let camp = amp * params.chroma;
        result += vec3<f32>(nc1 * camp, 0.0, nc2 * camp);
    }

    return vec4<f32>(clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
