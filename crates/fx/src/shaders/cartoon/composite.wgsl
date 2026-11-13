// Soft luminance quantization (Winnemöller 2006) + ink lines, Oklab in, linear RGB out.
// u.p0 = (levels, sharpness, line_strength, _).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var lab: texture_2d<f32>;
@group(0) @binding(1) var edges: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(lab)) { return; }
    var c = textureLoad(lab, id.xy, 0);
    let dq = 1.0 / u.p0.x;
    // Nearest bin boundary; tanh pushes values toward the bin centers on either side of it.
    // sharpness -> 0 is the identity, large sharpness is hard posterization.
    let qb = round(c.x / dq) * dq;
    let t = (c.x - qb) / (0.5 * dq);
    c.x = qb + 0.5 * dq * tanh(u.p0.y * t) / tanh(u.p0.y);
    let ink = mix(1.0, textureLoad(edges, id.xy, 0).x, u.p0.z);
    textureStore(dst, id.xy, vec4(oklab_to_linear(c.xyz) * ink, c.a));
}
