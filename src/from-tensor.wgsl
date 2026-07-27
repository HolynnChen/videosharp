// NCHW 张量 → 纹理（CNN 输出桥接）
//
// 与 to-tensor.wgsl 相反：把 ORT 输出的 GPUBuffer（NCHW 平面布局）写回
// RGBA 纹理，供后续 RCAS / grain / badge pass 使用。
//
// 用 render pass 而非 compute + storage texture：storage texture 需要
// 声明具体格式且不支持 rgba16float 的 read_write（Chrome stable 尚未支持），
// 走 fragment 输出更兼容。
//
// 注意 CNN 输出可能超出 [0,1]（XLSR 末端有 Clip，但其他模型未必），
// 这里统一钳制，避免后续 pass 收到非法值。

struct Dims {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<uniform> dims: Dims;

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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // dims 是 CNN 的输出尺寸；渲染目标可能略大（两段式下
    // round(target/4)*4 与 target 最多差 2 像素，如 480p→2K 的 1442 vs 1440）。
    // 越界时钳到最后一行/列 —— 差异仅出现在边缘 1~2 像素，视觉不可见。
    let px = vec2<u32>(in.uv * vec2<f32>(f32(dims.width), f32(dims.height)));
    let x = min(px.x, dims.width - 1u);
    let y = min(px.y, dims.height - 1u);

    let hw = dims.width * dims.height;
    let idx = y * dims.width + x;

    let rgb = vec3<f32>(
        src[idx],
        src[idx + hw],
        src[idx + hw * 2u],
    );

    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
