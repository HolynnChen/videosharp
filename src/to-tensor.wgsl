// 纹理 → NCHW 张量（CNN 输入桥接）
//
// 为什么需要这一步：ORT Web 的 Tensor.fromGpuBuffer() 只接受 GPUBuffer，
// 且要求 NCHW 平面布局（先整个 R 平面、再整个 G 平面、再 B 平面）。
// 而 WebGPU 纹理是 RGBA 交错的。两者内存布局完全不同，必须显式重排。
//
// 关键是这一步必须留在 GPU 上。1080p 有 200 万像素，若经 CPU 往返，
// 光是 24MB 的双向拷贝就会把 XLSR 的 28ms 推到 100ms 以上。
//
// 输出布局（batch=1，3 通道）：
//   [0 .. HW)        R 平面
//   [HW .. 2HW)      G 平面
//   [2HW .. 3HW)     B 平面

struct Dims {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dims.width || gid.y >= dims.height) {
        return;
    }

    let c = textureLoad(srcTex, vec2<i32>(gid.xy), 0);

    let hw = dims.width * dims.height;
    let idx = gid.y * dims.width + gid.x;

    // XLSR 的训练输入是 [0,1] 归一化的 RGB，与纹理值域一致，无需额外缩放
    dst[idx]          = c.r;
    dst[idx + hw]     = c.g;
    dst[idx + hw * 2] = c.b;
}
