// Turns each fitted splat into a straight brush stroke along its major axis.
// u.p0 = (count, length_mul, width_mul, _).

struct Stroke {
    color: vec4<f32>,  // Oklab + opacity
    n: u32,
    radius: f32,
    seed: u32,
    _pad: u32,
    pts: array<vec2<f32>, 16>,
}

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read_write> strokes: array<Stroke>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = linear_id(wg, li);
    if i >= u32(u.p0.x) { return; }
    let s = splats[i];
    let sc = exp(s.log_scale);
    let axis = vec2(cos(s.theta), sin(s.theta));
    // Major axis direction and sigmas.
    let major_x = sc.x >= sc.y;
    let dir = select(vec2(-axis.y, axis.x), axis, major_x);
    let major = max(sc.x, sc.y);
    let minor = min(sc.x, sc.y);

    var st: Stroke;
    st.color = vec4(s.color.rgb, sigmoid(s.opacity));
    // A stroke 2 sigma wide covers the splat's core; round caps add one radius at each end.
    st.radius = max(minor * u.p0.z, 0.5);
    let half = max(2.0 * major * u.p0.y - st.radius, 0.0);
    st.seed = hash_u32(i * 0x9e3779b9u + 1u);
    let n = clamp(u32(ceil(2.0 * half / st.radius)) + 1u, 2u, 16u);
    for (var k = 0u; k < n; k++) {
        st.pts[k] = s.mu + dir * mix(-half, half, f32(k) / f32(n - 1u));
    }
    st.n = n;
    strokes[i] = st;
}
