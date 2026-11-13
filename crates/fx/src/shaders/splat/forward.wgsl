// Renders the normalized splat sum per pixel. Writes (C, W) and the loss gradient dL/dC = C - target.
// u.p0 = (tiles_x, scale_mul, _, _).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read> tile_count: array<u32>;
@group(0) @binding(2) var<storage, read> tile_list: array<u32>;
@group(0) @binding(3) var target_lab: texture_2d<f32>;
@group(0) @binding(4) var background: texture_2d<f32>;
@group(0) @binding(5) var out_cw: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_grad: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(target_lab)) { return; }
    let p = vec2<f32>(id.xy) + 0.5;
    let t = (id.y / TILE) * u32(u.p0.x) + id.x / TILE;
    let n = min(tile_count[t], TILE_CAP);
    var acc = vec3(0.0);
    var wsum = 0.0;
    for (var k = 0u; k < n; k++) {
        let s = splats[tile_list[t * TILE_CAP + k]];
        let q = local(s, p, u.p0.y).z;
        if q > EXTENT * EXTENT { continue; }
        let w = sigmoid(s.opacity) * exp(-0.5 * q);
        acc += w * s.color.rgb;
        wsum += w;
    }
    let c = (acc + BETA * textureLoad(background, id.xy, 0).rgb) / (wsum + BETA);
    textureStore(out_cw, id.xy, vec4(c, wsum));
    textureStore(out_grad, id.xy, vec4(c - textureLoad(target_lab, id.xy, 0).rgb, 0.0));
}
