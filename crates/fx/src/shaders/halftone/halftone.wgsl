// Halftone print: up to four inks, each a screen of round dots on its own rotated grid.
// Appended to `rotated_grid::WGSL` (`RotatedGrid`, `rotated_grid_*`). Linear RGB in and out.
// `src` is pre-blurred so each dot takes the average tone of its cell.
// u.grid = the screen (image size, cell spacing; its angle is replaced per ink),
// u.angles = screen angle of ink slots 0..3 in radians,
// u.extra = (mode: 0 mono, 1 duotone, 2 CMYK, duotone spot hue in radians, _, _).

struct U { grid: RotatedGrid, angles: vec4<f32>, extra: vec4<f32> }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u: U;

const PI: f32 = 3.14159265;
const MONO: i32 = 0;
const DUOTONE: i32 = 1;

// Ink amounts per slot (cyan or spot, magenta, yellow, black) for a linear RGB color.
fn separate(mode: i32, rgb: vec3<f32>) -> vec4<f32> {
    let dark = 1.0 - clamp(linear_to_oklab(rgb).x, 0.0, 1.0);
    if mode == MONO {
        return vec4(0.0, 0.0, 0.0, dark);
    }
    if mode == DUOTONE {
        // The spot ink carries the tones, black only deepens the shadows.
        return vec4(dark, 0.0, 0.0, clamp(2.0 * dark - 1.0, 0.0, 1.0));
    }
    // Naive CMYK with full black replacement, on gamma-encoded values like print separations.
    let c = linear_to_srgb(rgb);
    let k = 1.0 - max(c.r, max(c.g, c.b));
    if k >= 1.0 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return vec4((1.0 - c - k) / (1.0 - k), k);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let mode = i32(u.extra.x);
    let hue = u.extra.y;
    let spot = clamp(oklab_to_linear(vec3(0.6, 0.15 * cos(hue), 0.15 * sin(hue))), vec3(0.0), vec3(1.0));
    let inks = array<vec3<f32>, 4>(
        select(srgb_to_linear(vec3(0.0, 0.68, 0.94)), spot, mode == DUOTONE),
        srgb_to_linear(vec3(0.93, 0.0, 0.55)),
        srgb_to_linear(vec3(1.0, 0.95, 0.0)),
        vec3(0.01),
    );

    // Pixel centers, as the grid's image space has its origin at the top-left corner.
    let p = vec2<f32>(id.xy) + 0.5;
    var paper = vec3(1.0);
    for (var ink = 0; ink < 4; ink++) {
        var grid = u.grid;
        grid.angle = u.angles[ink];
        // Dots in dark areas grow past half the spacing, so check all four cell corners.
        // Unlike the CPU iterator, keep dots just outside the image so the edges get ink too.
        let dots = rotated_grid_neighbors(grid, p);
        for (var i = 0; i < 4; i++) {
            let dot = dots[i];
            let amount = separate(mode, fetch(src, vec2<i32>(floor(dot))).rgb)[ink];
            if amount <= 0.0 { continue; }
            // The dot covers the same share of its cell as the ink amount.
            let radius = grid.spacing.x * sqrt(amount / PI);
            let coverage = clamp(radius - distance(p, dot) + 0.5, 0.0, 1.0);
            paper *= 1.0 - coverage * (1.0 - inks[ink]);
        }
    }
    textureStore(dst, id.xy, vec4(paper, 1.0));
}
