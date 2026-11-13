// Adam step, one invocation per (parameter slot k, splat i). Slots follow the Splat layout:
// 0-1 mu, 2-3 log_scale, 4 theta, 5 opacity, 6-7 pad, 8-10 color, 11 pad.
// u.p0 = (count, step t, lr_pos, lr_scale), u.p1 = (lr_rot, lr_opacity, lr_color, max_log_scale),
// u.p2 = (width, height).

struct U { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32> }
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> moment1: array<f32>;
@group(0) @binding(3) var<storage, read_write> moment2: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

const B1: f32 = 0.9;
const B2: f32 = 0.999;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    let i = id.y;
    if k >= 12u || i >= u32(u.p0.x) { return; }
    var lr = 0.0;
    switch k {
        case 0u, 1u: { lr = u.p0.z; }
        case 2u, 3u: { lr = u.p0.w; }
        case 4u: { lr = u.p1.x; }
        case 5u: { lr = u.p1.y; }
        case 8u, 9u, 10u: { lr = u.p1.z; }
        default: { return; }
    }
    let j = i * 12u + k;
    let g = grads[j];
    let m = B1 * moment1[j] + (1.0 - B1) * g;
    let v = B2 * moment2[j] + (1.0 - B2) * g * g;
    moment1[j] = m;
    moment2[j] = v;
    let t = u.p0.y;
    let mh = m / (1.0 - pow(B1, t));
    let vh = v / (1.0 - pow(B2, t));
    var x = params[j] - lr * mh / (sqrt(vh) + 1e-8);
    if k == 2u || k == 3u { x = clamp(x, log(0.3), u.p1.w); }
    if k == 5u { x = clamp(x, -6.0, 6.0); }
    if k < 2u { x = clamp(x, 0.0, u.p2[k]); }
    params[j] = x;
}
