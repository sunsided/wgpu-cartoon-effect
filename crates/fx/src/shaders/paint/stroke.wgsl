// One brush layer of Hertzmann's painterly rendering (1998): one invocation per grid cell.
// Where the canvas differs enough from the reference, trace a curved stroke along the
// tangent flow of the reference, starting at the cell's point of largest error.

const MAX_POINTS: u32 = 16u;

struct Stroke {
    color: vec4<f32>,  // Oklab + opacity
    n: u32,            // number of points, 0 = no stroke
    radius: f32,
    seed: u32,
    _pad: u32,
    pts: array<vec2<f32>, 16>,
}

// p0 = (radius, threshold, max_len, min_len), p1 = (curvature, first_layer, cols, seed)
struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var reference: texture_2d<f32>;
@group(0) @binding(1) var canvas: texture_2d<f32>;
@group(0) @binding(2) var tfm: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var<storage, read_write> strokes: array<Stroke>;
@group(0) @binding(5) var<uniform> u: U;

fn at(t: texture_2d<f32>, p: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t, smp, p / vec2<f32>(textureDimensions(t)), 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<f32>(textureDimensions(reference));
    let r = u.p0.x;
    let cols = u32(u.p1.z);
    let rows = u32(ceil(size.y / r));
    if id.x >= cols || id.y >= rows { return; }
    let idx = id.y * cols + id.x;
    let seed = hash_u32(idx ^ hash_u32(u32(u.p1.w) + u32(r) * 7919u));
    let first = u.p1.y > 0.5;

    // Area error over a 4x4 sampling of the cell; remember where it is largest.
    let origin = vec2<f32>(id.xy) * r;
    var err = 0.0;
    var best = -1.0;
    var start = origin + 0.5 * r;
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let p = min(origin + (vec2(f32(i), f32(j)) + 0.5) * 0.25 * r, size - 0.5);
            let e = distance(at(reference, p).rgb, at(canvas, p).rgb);
            err += e;
            if e > best { best = e; start = p; }
        }
    }
    if first {
        // Blank canvas: cover everything, jittered so the grid doesn't show.
        start = min(origin + vec2(hash01(seed), hash01(seed ^ 0x9e3779b9u)) * r, size - 0.5);
    } else if err / 16.0 < u.p0.y {
        strokes[idx].n = 0u;
        return;
    }

    let color = at(reference, start);
    var s: Stroke;
    s.color = vec4(color.rgb, 1.0);
    s.radius = r;
    s.seed = seed;
    s.pts[0] = start;
    var n = 1u;
    var d_prev = vec2(0.0);
    let max_len = min(u32(u.p0.z), MAX_POINTS - 1u);
    for (var i = 1u; i <= max_len; i++) {
        let cur = s.pts[n - 1u];
        if f32(i) > u.p0.w {
            let ref_c = at(reference, cur).rgb;
            // Stop once the canvas there is already closer to the reference than our paint.
            if distance(ref_c, at(canvas, cur).rgb) < distance(ref_c, color.rgb) { break; }
        }
        let f = at(tfm, cur);
        if f.w < 1e-3 && i > 1u { break; }
        var d = normalize(f.xy);
        if i == 1u && hash01(seed ^ 0x5bd1e995u) < 0.5 { d = -d; }
        if dot(d, d_prev) < 0.0 { d = -d; }
        if i > 1u { d = normalize(mix(d, d_prev, u.p1.x)); }
        let next = cur + r * d;
        if any(next < vec2(0.0)) || any(next >= size) { break; }
        s.pts[n] = next;
        n++;
        d_prev = d;
    }
    if n == 1u {
        // Single dab: give it a direction so it renders as a short oriented stroke.
        s.pts[1] = start + 0.5 * r * normalize(at(tfm, start).xy + vec2(1e-6, 0.0));
        n = 2u;
    }
    s.n = n;
    strokes[idx] = s;
}
