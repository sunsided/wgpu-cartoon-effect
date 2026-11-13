// Smooths the DoG response along the tangent flow (line integral convolution), then
// applies the soft XDoG threshold. u.p0 = (sigma_m, epsilon, phi, _). Output r = 1 (paper) .. 0 (ink).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var dog: texture_2d<f32>;
@group(0) @binding(1) var tfm: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<f32>(textureDimensions(dog));
    if any(vec2<f32>(id.xy) >= size) { return; }
    let sigma = u.p0.x;
    let steps = i32(ceil(2.0 * sigma));
    let p0 = vec2<f32>(id.xy) + 0.5;
    var sum = textureLoad(dog, id.xy, 0).x;
    var wsum = 1.0;
    for (var side = -1.0; side <= 1.0; side += 2.0) {
        var p = p0;
        var t = side * textureLoad(tfm, id.xy, 0).xy;
        for (var i = 1; i <= steps; i++) {
            p += t;
            var tn = textureSampleLevel(tfm, smp, p / size, 0.0).xy;
            // The tangent field has no inherent sign; keep walking the same way.
            if dot(tn, t) < 0.0 { tn = -tn; }
            t = tn;
            let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
            sum += w * textureSampleLevel(dog, smp, p / size, 0.0).x;
            wsum += w;
        }
    }
    let d = sum / wsum;
    let e = select(1.0 + tanh(u.p0.z * (d - u.p0.y)), 1.0, d >= u.p0.y);
    textureStore(dst, id.xy, vec4(e, 0.0, 0.0, 1.0));
}
