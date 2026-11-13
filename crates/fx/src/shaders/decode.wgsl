// Uploaded 8-bit sRGB image -> linear working format.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst_work: texture_storage_2d<rgba16float, write>;

// Source is an Rgba8UnormSrgb texture, so textureLoad already returns linear values.
@compute @workgroup_size(8, 8)
fn decode(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    textureStore(dst_work, id.xy, textureLoad(src, id.xy, 0));
}
