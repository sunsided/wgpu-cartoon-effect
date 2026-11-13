// Places splat i at the best of a few random candidates, favoring high gradient magnitude
// (detail needs more, smaller splats). u.p0 = (count, seed, init_scale, edge_bias).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var target_lab: texture_2d<f32>;
@group(0) @binding(1) var tfm: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(3) var<storage, read_write> moment1: array<f32>;
@group(0) @binding(4) var<storage, read_write> moment2: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = linear_id(wg, li);
    if i >= u32(u.p0.x) { return; }
    let size = vec2<f32>(textureDimensions(target_lab));
    var h = hash_u32(i ^ hash_u32(u32(u.p0.y)));
    var best = -1.0;
    var pos = vec2(0.0);
    for (var k = 0; k < 4; k++) {
        let p = vec2(hash01(h), hash01(h ^ 0x9e3779b9u)) * size;
        h = hash_u32(h);
        let g = textureLoad(tfm, vec2<u32>(p), 0).w;
        let score = mix(1.0, g * 20.0, u.p0.w) * hash01(h ^ 0x85ebca6bu);
        if score > best { best = score; pos = p; }
    }
    var s: Splat;
    s.mu = pos;
    s.log_scale = vec2(log(u.p0.z));
    s.theta = hash01(h ^ 0xc2b2ae35u) * 3.14159265;
    s.opacity = 2.0;
    s.color = vec4(textureLoad(target_lab, vec2<u32>(pos), 0).rgb, 0.0);
    splats[i] = s;
    for (var k = 0u; k < 12u; k++) {
        moment1[i * 12u + k] = 0.0;
        moment2[i * 12u + k] = 0.0;
    }
}
