// Structure tensor (E, F, G) = (gx·gx, gx·gy, gy·gy) summed over Oklab channels, Sobel gradients.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(src)) { return; }
    let p = vec2<i32>(id.xy);
    let a = fetch(src, p + vec2(-1, -1)).rgb;
    let b = fetch(src, p + vec2(0, -1)).rgb;
    let c = fetch(src, p + vec2(1, -1)).rgb;
    let d = fetch(src, p + vec2(-1, 0)).rgb;
    let f = fetch(src, p + vec2(1, 0)).rgb;
    let g = fetch(src, p + vec2(-1, 1)).rgb;
    let h = fetch(src, p + vec2(0, 1)).rgb;
    let i = fetch(src, p + vec2(1, 1)).rgb;
    let gx = (c + 2.0 * f + i - a - 2.0 * d - g) * 0.25;
    let gy = (g + 2.0 * h + i - a - 2.0 * b - c) * 0.25;
    textureStore(dst, id.xy, vec4(dot(gx, gx), dot(gx, gy), dot(gy, gy), 0.0));
}
