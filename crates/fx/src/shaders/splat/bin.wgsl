// Appends each splat to every 16x16 tile its 3-sigma box touches. u.p0 = (count, tiles_x, tiles_y, scale_mul).

struct U { p0: vec4<f32>, p1: vec4<f32> }
@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<storage, read_write> tile_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> tile_list: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = linear_id(wg, li);
    if i >= u32(u.p0.x) { return; }
    let s = splats[i];
    let e = half_extent(s, u.p0.w);
    let tiles = vec2<i32>(i32(u.p0.y), i32(u.p0.z));
    let lo = clamp(vec2<i32>(floor((s.mu - e) / f32(TILE))), vec2(0), tiles - 1);
    let hi = clamp(vec2<i32>(floor((s.mu + e) / f32(TILE))), vec2(0), tiles - 1);
    for (var ty = lo.y; ty <= hi.y; ty++) {
        for (var tx = lo.x; tx <= hi.x; tx++) {
            let t = u32(ty * tiles.x + tx);
            let slot = atomicAdd(&tile_count[t], 1u);
            // shortcut: overflowing tiles drop splats; raise TILE_CAP if dense regions look holey.
            if slot < TILE_CAP { tile_list[t * TILE_CAP + slot] = i; }
        }
    }
}
