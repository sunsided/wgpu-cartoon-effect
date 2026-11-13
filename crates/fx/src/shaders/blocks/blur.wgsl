// One separable Gaussian pass. u.p0 = (dir.x, dir.y, sigma, _).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let dir = vec2<i32>(u.p0.xy);
    let sigma = u.p0.z;
    let r = i32(ceil(2.5 * sigma));
    var sum = vec4(0.0);
    var wsum = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
        sum += w * fetch(src, vec2<i32>(id.xy) + i * dir);
        wsum += w;
    }
    textureStore(dst, id.xy, sum / wsum);
}
