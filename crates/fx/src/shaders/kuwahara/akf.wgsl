// Anisotropic Kuwahara filter with polynomial sector weights
// (Kyprianidis et al. 2009, 2010). Oklab in, Oklab out.
// u.p0 = (radius, q (sharpness), alpha (anisotropy tuning), variance scale).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var tfm: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> u: U;

const N: i32 = 8;
const ZERO_CROSSING: f32 = 0.58;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let radius = u.p0.x;
    let q = u.p0.y;
    let alpha = u.p0.z;
    let var_scale = u.p0.w;

    let t = textureLoad(tfm, id.xy, 0);
    let aniso = t.z;
    // Ellipse stretched along the edge tangent, by how anisotropic the neighborhood is.
    let a = radius * clamp((alpha + aniso) / alpha, 0.1, 2.0);
    let b = radius * clamp(alpha / (alpha + aniso), 0.1, 2.0);
    let cos_phi = t.x;
    let sin_phi = t.y;
    // Maps offsets into the unit-diameter disc of the ellipse frame.
    let sr0 = vec2(cos_phi, sin_phi) * (0.5 / a);
    let sr1 = vec2(-sin_phi, cos_phi) * (0.5 / b);
    let max_x = i32(ceil(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi)));
    let max_y = i32(ceil(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi)));

    let zeta = 2.0 / radius;
    let sin_zc = sin(ZERO_CROSSING);
    let eta = (zeta + cos(ZERO_CROSSING)) / (sin_zc * sin_zc);

    var m: array<vec4<f32>, 8>;
    var s: array<vec3<f32>, 8>;
    let p = vec2<i32>(id.xy);
    for (var j = -max_y; j <= max_y; j++) {
        for (var i = -max_x; i <= max_x; i++) {
            let o = vec2<f32>(f32(i), f32(j));
            var v = vec2(dot(sr0, o), dot(sr1, o));
            if dot(v, v) > 0.25 { continue; }
            let c = fetch(src, p + vec2(i, j)).rgb;
            var w: array<f32, 8>;
            var sum = 0.0;
            for (var k = 0; k < 2; k++) {
                let vxx = zeta - eta * v.x * v.x;
                let vyy = zeta - eta * v.y * v.y;
                var z = max(0.0, v.y + vxx); w[k] = z * z;
                z = max(0.0, -v.x + vyy); w[k + 2] = z * z;
                z = max(0.0, -v.y + vxx); w[k + 4] = z * z;
                z = max(0.0, v.x + vyy); w[k + 6] = z * z;
                sum += w[k] + w[k + 2] + w[k + 4] + w[k + 6];
                // Second set of four sectors, rotated by 45 degrees.
                v = 0.70710678 * vec2(v.x - v.y, v.x + v.y);
            }
            let g = exp(-3.125 * dot(v, v)) / max(sum, 1e-8);
            for (var k = 0; k < N; k++) {
                let wk = w[k] * g;
                m[k] += vec4(c * wk, wk);
                s[k] += c * c * wk;
            }
        }
    }

    var o = vec4(0.0);
    for (var k = 0; k < N; k++) {
        let mean = m[k].rgb / max(m[k].w, 1e-8);
        let variance = abs(s[k] / max(m[k].w, 1e-8) - mean * mean);
        let wk = 1.0 / (1.0 + pow(var_scale * (variance.x + variance.y + variance.z), 0.5 * q));
        o += vec4(mean * wk, wk);
    }
    textureStore(dst, id.xy, vec4(o.rgb / o.w, 1.0));
}
