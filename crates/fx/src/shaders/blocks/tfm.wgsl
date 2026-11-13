// Smoothed structure tensor -> tangent flow map: (tangent.xy, anisotropy, sqrt(major eigenvalue)).

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let g = textureLoad(src, id.xy, 0).xyz;
    let e = g.x;
    let f = g.y;
    let gg = g.z;
    let root = sqrt((e - gg) * (e - gg) + 4.0 * f * f);
    let l1 = 0.5 * (e + gg + root);
    let l2 = 0.5 * (e + gg - root);
    // Eigenvector of the minor eigenvalue = direction of least change = edge tangent.
    var t = vec2(l1 - e, -f);
    if dot(t, t) < 1e-12 { t = vec2(0.0, 1.0); }
    t = normalize(t);
    let a = select(0.0, (l1 - l2) / (l1 + l2), l1 + l2 > 1e-8);
    textureStore(dst, id.xy, vec4(t, a, sqrt(max(l1, 0.0))));
}
