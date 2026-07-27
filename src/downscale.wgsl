// 高质量降采样 —— 用于 CNN 两段式管线的预缩放
//
// 为什么需要专门的 shader：
//   CNN 模型固定 4x（XLSR 末端的 DepthToSpace 把 48 通道重排成 4×4 空间
//   像素，4 是编译进权重的）。要支持任意目标倍率，就得先把源缩放到
//   「目标尺寸 ÷ 4」，再交给 CNN 放大 4 倍。
//
//   例：1080p → 2K 时，先缩到 640×360，CNN 输出正好 2560×1440。
//   附带的好处是 CNN 只需处理 1/9 的像素量，开销降到 1/16 左右。
//
// 为什么不直接用双线性或 EASU：
//   缩小时双线性只取 2×2 邻域，当缩放比超过 2 倍就会漏采样，产生摩尔纹与
//   闪烁（视频上尤其明显，因为每帧走样位置不同）。EASU 的 12-tap 核是为
//   放大设计的，缩小时同样采样不足。
//
//   这里按实际缩放比动态决定采样半径，做加权区域平均 —— 相当于先低通
//   再重采样，符合 Nyquist 要求。权重用 Lanczos2 近似，比箱式平均保留
//   更多细节。

struct Params {
    // 目标尺寸（缩放比由它与源尺寸推出）
    dstWidth: f32,
    dstHeight: f32,
    _pad0: f32,
    _pad1: f32,
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

// Lanczos2 的无三角函数近似：在 |x|<2 内近似 sinc(x)·sinc(x/2)
fn lanczosWeight(x: f32) -> f32 {
    let ax = abs(x);
    if (ax >= 2.0) { return 0.0; }
    // (1 - (x/2)^2) 型窗，配合中心峰，形状接近 Lanczos2 且无除零风险
    let t = ax * 0.5;
    let w = 1.0 - t * t;
    return w * w * (1.0 - 0.5 * ax);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let srcSize = vec2<f32>(textureDimensions(srcTex));
    let dstSize = vec2<f32>(params.dstWidth, params.dstHeight);

    // 每个输出像素覆盖多少个源像素。>1 表示在缩小。
    let ratio = srcSize / max(dstSize, vec2<f32>(1.0));

    // 放大或等比时无需区域平均，直接双线性即可（此 shader 只为缩小设计）
    if (ratio.x <= 1.0 && ratio.y <= 1.0) {
        return vec4<f32>(
            textureSampleLevel(srcTex, srcSampler, in.uv, 0.0).rgb, 1.0);
    }

    // 采样半径随缩放比增长，保证覆盖整个源像素足迹。
    // 上限 4（即最多 9×9=81 次采样）—— 再大收益递减而开销陡增；
    // 缩放比超过 8 倍的场景本项目不会出现（最多 1080p→270p 即 4 倍）。
    let radius = clamp(ceil(ratio), vec2<f32>(1.0), vec2<f32>(4.0));
    let center = in.uv * srcSize;

    var acc = vec3<f32>(0.0);
    var wsum = 0.0;

    let rx = i32(radius.x);
    let ry = i32(radius.y);
    for (var dy = -ry; dy <= ry; dy++) {
        for (var dx = -rx; dx <= rx; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let samplePos = floor(center + offset) + 0.5;
            // 权重按「归一化到输出像素足迹」的距离计算，
            // 这样缩放比越大、低通越强，正好抑制走样
            let d = (samplePos - center) / ratio;
            let w = lanczosWeight(d.x) * lanczosWeight(d.y);
            if (w > 0.0) {
                let uv = clamp(samplePos, vec2<f32>(0.5), srcSize - 0.5) / srcSize;
                acc += textureSampleLevel(srcTex, srcSampler, uv, 0.0).rgb * w;
                wsum += w;
            }
        }
    }

    // wsum 理论上不会为 0（中心样本权重恒正），兜底防除零
    let result = acc / max(wsum, 1e-5);
    return vec4<f32>(clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
