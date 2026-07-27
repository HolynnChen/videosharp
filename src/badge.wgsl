// 角标合成 pass —— 把预渲染好的文字纹理叠加到画面右上角
//
// 为什么角标要画进 canvas 而不是用独立 DOM：
//   独立 DOM 角标与 canvas 是两个元素，各自可能被站点的不同层级遮挡。
//   角标可见不代表 canvas 可见，那它作为"增强已生效"的指示就失效了。
//   画进 canvas 后，看得见角标 ⟺ 看得见增强画面，物理上无法解耦。
//
// 文字纹理由 JS 侧用 2D canvas 预渲染（仅在文案变化时重建），这里只做
// alpha 混合，每帧开销可忽略。

struct Params {
    // 角标在画面中的归一化位置与尺寸 (x, y, w, h)，原点左上
    rect: vec4<f32>,
    // 整体不透明度
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var badgeTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

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
    let base = textureSampleLevel(srcTex, srcSampler, in.uv, 0.0).rgb;

    let r = params.rect;
    // 当前像素在角标矩形内的局部坐标
    let local = (in.uv - r.xy) / r.zw;

    // 矩形外直接返回原图。分支在 GPU 上会整个 warp 一起走，
    // 角标只占画面极小一块，绝大多数 warp 走的是这条便宜路径。
    if (local.x < 0.0 || local.x > 1.0 || local.y < 0.0 || local.y > 1.0) {
        return vec4<f32>(base, 1.0);
    }

    // 文字纹理已含背景圆角矩形与文字，直接 premultiplied 混合
    let badge = textureSampleLevel(badgeTex, srcSampler, local, 0.0);
    let a = badge.a * params.opacity;
    return vec4<f32>(mix(base, badge.rgb, a), 1.0);
}
