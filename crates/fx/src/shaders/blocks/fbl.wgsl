// One pass of the separable flow-based bilateral filter in Oklab.
// u.p0 = (sigma_d, sigma_r, along_gradient (0|1), _).

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
    let sigma_d = u.p0.x;
    let sigma_r = u.p0.y;
    let t = textureLoad(tfm, id.xy, 0).xy;
    let dir = select(t, vec2(-t.y, t.x), u.p0.z > 0.5);
    let uv = (vec2<f32>(id.xy) + 0.5) / size;
    let c0 = textureLoad(src, id.xy, 0);
    let r = i32(ceil(2.0 * sigma_d));
    var sum = vec3(0.0);
    var wsum = 0.0;
    for (var i = -r; i <= r; i++) {
        let c = textureSampleLevel(src, smp, uv + f32(i) * dir / size, 0.0).rgb;
        let d = c - c0.rgb;
        let w = exp(-f32(i * i) / (2.0 * sigma_d * sigma_d) - dot(d, d) / (2.0 * sigma_r * sigma_r));
        sum += w * c;
        wsum += w;
    }
    textureStore(dst, id.xy, vec4(sum / wsum, c0.a));
}
