// Draws strokes as instanced triangle strips with a procedural bristle brush.
// Instances are visited in a seeded pseudo-random permutation (Hertzmann paints in random order).

const MAX_POINTS: u32 = 16u;

struct Stroke {
    color: vec4<f32>,
    n: u32,
    radius: f32,
    seed: u32,
    _pad: u32,
    pts: array<vec2<f32>, 16>,
}

// p0 = (opacity, bristles, dryness, value_jitter), p1 = (count, bits, seed, chroma_jitter), p2 = (width, height)
struct U { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32> }
@group(0) @binding(0) var<storage, read> strokes: array<Stroke>;
@group(0) @binding(1) var<uniform> u: U;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // x: distance along the stroke in px, y: across in [-1, 1], z: stroke length, w: radius
    @location(1) coord: vec4<f32>,
    @location(2) @interpolate(flat) seed: u32,
}

// Bijection on [0, 2^bits), cycle-walked into [0, count).
fn permute(i: u32, count: u32, bits: u32, seed: u32) -> u32 {
    let mask = (1u << bits) - 1u;
    var x = i;
    // count > 2^(bits-1), so each step lands in range with p > 1/2; 64 misses won't happen.
    for (var k = 0; k < 64; k++) {
        x = (x ^ seed) & mask;
        x = (x * 0x2c1b3c6du) & mask;
        x ^= x >> max(bits / 2u, 1u);
        x = (x * 0x297a2d39u) & mask;
        x ^= x >> max(bits / 3u, 1u);
        if x < count { break; }
    }
    return min(x, count - 1u);
}

fn pt(si: u32, j: u32) -> vec2<f32> {
    return strokes[si].pts[j];
}

// Direction of segment k -> k+1, clamped to the stroke's segments.
fn seg_dir(si: u32, n: u32, k: u32) -> vec2<f32> {
    let j = min(k, n - 2u);
    let d = pt(si, j + 1u) - pt(si, j);
    return select(vec2(1.0, 0.0), normalize(d), dot(d, d) > 1e-8);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var out: VsOut;
    let si = permute(ii, u32(u.p1.x), u32(u.p1.y), u32(u.p1.z));
    let n = strokes[si].n;
    let r = strokes[si].radius;
    let seed = strokes[si].seed;
    if n < 2u {
        out.pos = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    // Strip vertices: pairs along [start cap, pts[0..n], end cap]; extras collapse onto the end cap.
    let k = min(vi / 2u, n + 1u);
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);
    var len = 0.0;
    var along = 0.0;
    for (var i = 1u; i < n; i++) {
        let d = distance(pt(si, i), pt(si, i - 1u));
        len += d;
        // Body vertex k sits on point k - 1: sum the segments before it.
        if i < k { along += d; }
    }
    var p: vec2<f32>;
    var t: vec2<f32>;
    if k == 0u {
        t = seg_dir(si, n, 0u);
        p = pt(si, 0u) - t * r;
        along = -r;
    } else if k == n + 1u {
        t = seg_dir(si, n, n - 2u);
        p = pt(si, n - 1u) + t * r;
        along = len + r;
    } else {
        let j = k - 1u;
        // Average adjacent segment directions for a smooth joint.
        t = normalize(seg_dir(si, n, max(j, 1u) - 1u) + seg_dir(si, n, j) + vec2(1e-6, 0.0));
        p = pt(si, j);
    }
    p += vec2(-t.y, t.x) * side * r;
    out.pos = vec4(p / u.p2.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);

    // Per-stroke color jitter in Oklab.
    var c = strokes[si].color.rgb;
    c.x += (hash01(seed ^ 0x68e31da4u) - 0.5) * u.p0.w;
    c.y += (hash01(seed ^ 0xb5297a4du) - 0.5) * u.p1.w;
    c.z += (hash01(seed ^ 0x1b56c4e9u) - 0.5) * u.p1.w;
    out.color = vec4(c, 1.0);
    out.coord = vec4(along, side, len, r);
    out.seed = seed;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let along = in.coord.x;
    let v = in.coord.y;
    let len = in.coord.z;
    let r = in.coord.w;
    // Distance from the stroke's spine in radii, with round caps.
    var d = abs(v);
    if along < 0.0 { d = length(vec2(along / r, v)); }
    if along > len { d = length(vec2((along - len) / r, v)); }
    let coverage = smoothstep(1.0, 0.8, d);

    // Bristles: noise across the brush, stretched along it.
    let bristle_count = clamp(r * 0.8, 4.0, 40.0);
    let b = noise1(v * bristle_count, in.seed);
    let streak = noise1(v * bristle_count * 2.7 + along * 0.02, in.seed ^ 0x27d4eb2fu);
    let s = clamp(along / max(len + r, 1.0), 0.0, 1.0);
    let bristles = u.p0.y;
    var alpha = coverage * u.p0.x * mix(1.0, 0.35 + 0.65 * b, bristles);
    // Dry brush: paint runs out toward the end of the stroke, first between the bristles.
    alpha *= 1.0 - u.p0.z * smoothstep(0.25, 1.0, s) * (1.0 - streak);

    var c = in.color.rgb;
    c.x += (b - 0.5) * 0.06 * bristles;
    return vec4(c, alpha);
}
