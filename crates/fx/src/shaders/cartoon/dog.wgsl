// 1D difference of Gaussians of Oklab L across the edge (along the gradient).
// u.p0 = (sigma_e, tau, _, _). Output r = G_sigma - tau * G_{1.6 sigma}.

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var tfm: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<f32>(textureDimensions(src));
    if any(vec2<f32>(id.xy) >= size) { return; }
    let s1 = u.p0.x;
    let s2 = 1.6 * s1;
    let t = textureLoad(tfm, id.xy, 0).xy;
    let n = vec2(-t.y, t.x);
    let uv = (vec2<f32>(id.xy) + 0.5) / size;
    let r = i32(ceil(2.5 * s2));
    var a = vec2(0.0);
    var b = vec2(0.0);
    for (var i = -r; i <= r; i++) {
        let l = textureSampleLevel(src, smp, uv + f32(i) * n / size, 0.0).x;
        let d2 = f32(i * i);
        let w1 = exp(-d2 / (2.0 * s1 * s1));
        let w2 = exp(-d2 / (2.0 * s2 * s2));
        a += vec2(w1 * l, w1);
        b += vec2(w2 * l, w2);
    }
    textureStore(dst, id.xy, vec4(a.x / a.y - u.p0.y * b.x / b.y, 0.0, 0.0, 1.0));
}
