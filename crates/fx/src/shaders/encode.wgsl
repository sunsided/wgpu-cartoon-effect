@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

// Writes sRGB-encoded bytes into a plain Rgba8Unorm texture (storage can't be *Srgb).
@compute @workgroup_size(8, 8)
fn encode(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let c = textureLoad(src, id.xy, 0);
    textureStore(dst, id.xy, vec4(linear_to_srgb(c.rgb), 1.0));
}
