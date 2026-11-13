// Per splat: accumulates dL/dparams over the pixels of its 3-sigma box (no float atomics needed).
// u.p0 = (count, _, _, _).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var cw: texture_2d<f32>;
@group(0) @binding(2) var grad: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> grads: array<Splat>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = linear_id(wg, li);
    if i >= u32(u.p0.x) { return; }
    let s = splats[i];
    let size = vec2<i32>(textureDimensions(cw));
    let e = half_extent(s, 1.0);
    let lo = clamp(vec2<i32>(floor(s.mu - e)), vec2(0), size - 1);
    let hi = clamp(vec2<i32>(ceil(s.mu + e)), vec2(0), size - 1);
    let sc = exp(s.log_scale);
    let inv2 = 1.0 / (sc * sc);
    let c = cos(s.theta);
    let si = sin(s.theta);
    let sig = sigmoid(s.opacity);

    var g_mu = vec2(0.0);
    var g_ls = vec2(0.0);
    var g_theta = 0.0;
    var g_o = 0.0;
    var g_c = vec3(0.0);
    var contribution = 0.0;
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let l = local(s, vec2<f32>(f32(x), f32(y)) + 0.5, 1.0);
            if l.z > EXTENT * EXTENT { continue; }
            let w = sig * exp(-0.5 * l.z);
            let px = textureLoad(cw, vec2(x, y), 0);
            let g = textureLoad(grad, vec2(x, y), 0).rgb;
            let inv_w = 1.0 / (px.w + BETA);
            contribution += w * inv_w;
            g_c += g * (w * inv_w);
            let dl_dw = dot(g, s.color.rgb - px.rgb) * inv_w;
            let dl_dq = dl_dw * (-0.5 * w);
            let uv = l.xy;
            // dq/dmu = -2 R (u / s^2), back to image axes.
            let a = uv * inv2;
            g_mu += dl_dq * -2.0 * vec2(c * a.x - si * a.y, si * a.x + c * a.y);
            g_ls += dl_dq * -2.0 * uv * uv * inv2;
            g_theta += dl_dq * 2.0 * uv.x * uv.y * (inv2.x - inv2.y);
            g_o += dl_dw * w * (1.0 - sig);
        }
    }
    var out: Splat;
    out.mu = g_mu;
    out.log_scale = g_ls;
    out.theta = g_theta;
    out.opacity = g_o;
    out.color = vec4(g_c, 0.0);
    // Not a gradient: how many pixels' worth of the image this splat makes up (for respawning).
    out._pad = vec2(contribution, 0.0);
    grads[i] = out;
}
