@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let c = textureLoad(src, id.xy, 0);
    textureStore(dst, id.xy, vec4(linear_to_oklab(c.rgb), c.a));
}
