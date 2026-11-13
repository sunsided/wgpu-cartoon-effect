// Shared by the splat kernels. Gaussian i has weight
//   w = sigmoid(o) * exp(-q / 2),  q = (u.x / sx)^2 + (u.y / sy)^2,  u = R(-theta) (p - mu),
// and the image is the normalized sum C = (sum(w c) + BETA bg) / (sum(w) + BETA) (order independent).
// The faint background (a heavily blurred image) fills gaps no splat reaches, which would
// otherwise turn black and, receiving no gradient, never get fixed.

struct Splat {
    mu: vec2<f32>,
    log_scale: vec2<f32>,
    theta: f32,
    opacity: f32,  // logit
    _pad: vec2<f32>,
    color: vec4<f32>,  // Oklab
}

const TILE: u32 = 16u;
const TILE_CAP: u32 = 512u;
const BETA: f32 = 1e-3;
/// Cut-off in standard deviations.
const EXTENT: f32 = 3.0;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn linear_id(wg: vec3<u32>, li: u32) -> u32 {
    return wg.x * 64u + li;
}

/// Pixel-space bounding half-extent of a splat.
fn half_extent(s: Splat, scale_mul: f32) -> vec2<f32> {
    let sc = exp(s.log_scale) * scale_mul;
    let c = abs(cos(s.theta));
    let si = abs(sin(s.theta));
    return EXTENT * vec2(sqrt(sc.x * sc.x * c * c + sc.y * sc.y * si * si), sqrt(sc.x * sc.x * si * si + sc.y * sc.y * c * c));
}

/// Local frame offset u and q for pixel center p.
fn local(s: Splat, p: vec2<f32>, scale_mul: f32) -> vec3<f32> {
    let d = p - s.mu;
    let c = cos(s.theta);
    let si = sin(s.theta);
    let u = vec2(c * d.x + si * d.y, -si * d.x + c * d.y);
    let sc = exp(s.log_scale) * scale_mul;
    let v = u / sc;
    return vec3(u, dot(v, v));
}
