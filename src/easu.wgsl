// EASU (Edge Adaptive Spatial Upsampling) — WGSL 移植
//
// 源自 AMD FidelityFX Super Resolution 的 ffx_fsr1.h / FsrEasuF。
// 原始实现为 MIT 许可 (Copyright AMD)。
//
// 为什么用 EASU 而不是浏览器默认的双线性放大：
//   双线性只按距离加权，斜边会被拍成阶梯或糊成一团。EASU 先估计每个输出
//   像素处的边缘方向与强度，再把 12-tap 采样核沿边缘方向拉长、垂直方向压扁，
//   相当于"沿着边缘插值、不跨过边缘插值"。斜线因此保持锐利连续。
//
// 与原版的差异（不影响数学）：
//   - 原版用 4 次 gather4 取 12 个 texel；WGSL 的 textureGather 对 external
//     texture 不可用，这里改为逐点采样。采样数相同，只是访存模式不同。
//   - 原版用 APrxLoRcpF1/APrxLoRsqF1 等快速近似倒数；这里直接用精确除法，
//     现代 GPU 上开销差异可忽略，且避免了近似误差。
//
// 12-tap 布局（相对左上角的 f 点）：
//      b c
//    e f g h
//    i j k l
//      n o

// 输入是上一 pass 的中间纹理（enhance 的输出），不是 external texture。
// 管线固定为 enhance → easu → rcas，所以这里只需处理 texture_2d。
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

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

fn tapAt(coord: vec2<f32>, srcSize: vec2<f32>) -> vec3<f32> {
    // 采样点取像素中心，clamp 到有效范围避免边界越界
    let clamped = clamp(coord, vec2<f32>(0.5), srcSize - vec2<f32>(0.5));
    return textureSampleLevel(srcTex, srcSampler, clamped / srcSize, 0.0).rgb;
}

// 与 RCAS 一致的近似亮度（2x 亮度，比例关系足够用于方向估计）
fn luma(c: vec3<f32>) -> f32 {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// FsrEasuSetF：从十字形 5 点亮度累积边缘方向与强度
// w 是该象限的双线性权重
fn easuSet(
    dir: ptr<function, vec2<f32>>,
    len: ptr<function, f32>,
    lA: f32, lB: f32, lC: f32, lD: f32, lE: f32,
    w: f32,
) {
    // X 轴：以 lC 为中心，lB 在左、lD 在右
    let dc = lD - lC;
    let cb = lC - lB;
    var lenX = max(abs(dc), abs(cb));
    lenX = select(1.0 / lenX, 0.0, lenX == 0.0);
    let dirX = lD - lB;
    (*dir).x += dirX * w;
    var sx = clamp(abs(dirX) * lenX, 0.0, 1.0);
    sx = sx * sx;
    *len += sx * w;

    // Y 轴：lA 在上、lE 在下
    let ec = lE - lC;
    let ca = lC - lA;
    var lenY = max(abs(ec), abs(ca));
    lenY = select(1.0 / lenY, 0.0, lenY == 0.0);
    let dirY = lE - lA;
    (*dir).y += dirY * w;
    var sy = clamp(abs(dirY) * lenY, 0.0, 1.0);
    sy = sy * sy;
    *len += sy * w;
}

// FsrEasuTapF：把偏移旋转到边缘坐标系，做各向异性缩放，算 lanczos2 近似权重
fn easuTap(
    aC: ptr<function, vec3<f32>>,
    aW: ptr<function, f32>,
    off: vec2<f32>,
    dir: vec2<f32>,
    len2: vec2<f32>,
    lob: f32,
    clp: f32,
    color: vec3<f32>,
) {
    // 旋转到 (沿边缘, 垂直边缘) 坐标系
    var v: vec2<f32>;
    v.x = off.x * dir.x + off.y * dir.y;
    v.y = off.x * -dir.y + off.y * dir.x;
    // 各向异性：沿边缘方向拉长(len2.x)、垂直方向压扁(len2.y)
    v = v * len2;

    var d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);

    // lanczos2 的无 sin/sqrt 近似：
    //   (25/16 * (2/5*x^2 - 1)^2 - (25/16 - 1)) * (lob*x^2 - 1)^2
    var wB = (2.0 / 5.0) * d2 - 1.0;
    var wA = lob * d2 - 1.0;
    wB = wB * wB;
    wA = wA * wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    let weight = wB * wA;

    *aC += color * weight;
    *aW += weight;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let srcSize = vec2<f32>(textureDimensions(srcTex));

    // 输出像素中心映射回输入像素空间
    let pp = in.uv * srcSize - vec2<f32>(0.5);
    let fp = floor(pp);
    let frac = pp - fp;

    // ---- 取 12 个 tap ----
    // 以 fp 为 f 点（左上），坐标 +0.5 取像素中心
    let base = fp + vec2<f32>(0.5);
    let b = tapAt(base + vec2<f32>( 0.0, -1.0), srcSize);
    let c = tapAt(base + vec2<f32>( 1.0, -1.0), srcSize);
    let e = tapAt(base + vec2<f32>(-1.0,  0.0), srcSize);
    let f = tapAt(base + vec2<f32>( 0.0,  0.0), srcSize);
    let g = tapAt(base + vec2<f32>( 1.0,  0.0), srcSize);
    let h = tapAt(base + vec2<f32>( 2.0,  0.0), srcSize);
    let i = tapAt(base + vec2<f32>(-1.0,  1.0), srcSize);
    let j = tapAt(base + vec2<f32>( 0.0,  1.0), srcSize);
    let k = tapAt(base + vec2<f32>( 1.0,  1.0), srcSize);
    let l = tapAt(base + vec2<f32>( 2.0,  1.0), srcSize);
    let n = tapAt(base + vec2<f32>( 0.0,  2.0), srcSize);
    let o = tapAt(base + vec2<f32>( 1.0,  2.0), srcSize);

    let lb = luma(b); let lc = luma(c);
    let le = luma(e); let lf = luma(f); let lg = luma(g); let lh = luma(h);
    let li = luma(i); let lj = luma(j); let lk = luma(k); let ll = luma(l);
    let ln = luma(n); let lo = luma(o);

    // ---- 四象限双线性权重 ----
    let biS = (1.0 - frac.x) * (1.0 - frac.y);
    let biT = frac.x * (1.0 - frac.y);
    let biU = (1.0 - frac.x) * frac.y;
    let biV = frac.x * frac.y;

    var dir = vec2<f32>(0.0);
    var len = 0.0;
    // 每个象限以其最近的 texel 为中心取十字：上 左 中 右 下
    easuSet(&dir, &len, lb, le, lf, lg, lj, biS); // 中心 f
    easuSet(&dir, &len, lc, lf, lg, lh, lk, biT); // 中心 g
    easuSet(&dir, &len, lf, li, lj, lk, ln, biU); // 中心 j
    easuSet(&dir, &len, lg, lj, lk, ll, lo, biV); // 中心 k

    // ---- 归一化方向、推导核形状 ----
    let dir2 = dir * dir;
    var dirR = dir2.x + dir2.y;
    let zro = dirR < (1.0 / 32768.0);
    // 无明显方向时退化为水平，权重由 len≈0 保证接近各向同性
    dirR = select(inverseSqrt(dirR), 1.0, zro);
    dir.x = select(dir.x, 1.0, zro);
    dir = dir * dirR;

    // len 从 {0..2} 映射到 {0..1} 再平方
    len = len * 0.5;
    len = len * len;

    // 对角线方向的拉伸系数
    let stretch = (dir.x * dir.x + dir.y * dir.y)
                / max(abs(dir.x), abs(dir.y));
    let len2 = vec2<f32>(
        1.0 + (stretch - 1.0) * len,
        1.0 - 0.5 * len,
    );

    // 负 lobe 位置与裁剪半径
    let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    let clp = 1.0 / lob;

    // ---- 累积 12 tap ----
    var aC = vec3<f32>(0.0);
    var aW = 0.0;
    easuTap(&aC, &aW, vec2<f32>( 0.0, -1.0) - frac, dir, len2, lob, clp, b);
    easuTap(&aC, &aW, vec2<f32>( 1.0, -1.0) - frac, dir, len2, lob, clp, c);
    easuTap(&aC, &aW, vec2<f32>(-1.0,  0.0) - frac, dir, len2, lob, clp, e);
    easuTap(&aC, &aW, vec2<f32>( 0.0,  0.0) - frac, dir, len2, lob, clp, f);
    easuTap(&aC, &aW, vec2<f32>( 1.0,  0.0) - frac, dir, len2, lob, clp, g);
    easuTap(&aC, &aW, vec2<f32>( 2.0,  0.0) - frac, dir, len2, lob, clp, h);
    easuTap(&aC, &aW, vec2<f32>(-1.0,  1.0) - frac, dir, len2, lob, clp, i);
    easuTap(&aC, &aW, vec2<f32>( 0.0,  1.0) - frac, dir, len2, lob, clp, j);
    easuTap(&aC, &aW, vec2<f32>( 1.0,  1.0) - frac, dir, len2, lob, clp, k);
    easuTap(&aC, &aW, vec2<f32>( 2.0,  1.0) - frac, dir, len2, lob, clp, l);
    easuTap(&aC, &aW, vec2<f32>( 0.0,  2.0) - frac, dir, len2, lob, clp, n);
    easuTap(&aC, &aW, vec2<f32>( 1.0,  2.0) - frac, dir, len2, lob, clp, o);

    // ---- 去 ringing：钳制到最近 4 个 texel 的范围内 ----
    let min4 = min(min(f, g), min(j, k));
    let max4 = max(max(f, g), max(j, k));

    // aW 理论上不会为 0（中心 tap 权重恒正），兜底防除零
    let result = aC / max(aW, 1e-5);
    return vec4<f32>(clamp(result, min4, max4), 1.0);
}
