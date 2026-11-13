# wgpu-cartoon-effect

Image stylization on the GPU with [wgpu](https://wgpu.rs): native CLI, a bare Trunk dev page,
and a SvelteKit UI, all sharing one Rust core (`crates/fx`) of WGSL compute passes.
Successor of [opencv-cartoon-effect](https://github.com/sunsided/opencv-cartoon-effect).

| Effect    | Technique                                                                                          |
|-----------|----------------------------------------------------------------------------------------------------|
| `cartoon` | Flow-based bilateral filter in Oklab, soft luminance quantization, FDoG/XDoG lines along the tangent flow |
| `kuwahara` | Anisotropic Kuwahara filter with polynomial sector weights (Kyprianidis et al.)                    |
| `paint`   | Coarse-to-fine curved brush strokes (Hertzmann 1998): placed by a compute pass, drawn as instanced bristle-textured strips |
| `splat`   | N anisotropic 2D Gaussians fitted to the image with Adam; few splats or a shrunken render scale abstract it |

## Layout

- `crates/fx`: effects library (no IO, no windowing)
- `crates/fx-cli`: `fx <effect> <input> -o out.png [-s key=value]...`, `fx --list`
- `crates/fx-wasm`: wasm-bindgen API (`Renderer.create(canvas)`, `setImage`, `render`), built with wasm-pack
- `crates/fx-playground`: Trunk page, `?effect=cartoon&levels=4`
- `web/`: SvelteKit app (Yarn via Corepack) with sliders generated from each effect's parameter list

## Running

Uses [Task](https://taskfile.dev):

```sh
task test                # headless GPU tests (WGPU_FORCE_FALLBACK_ADAPTER=1 for software rendering)
task samples             # every effect on the sample image into out/
task trunk               # http://127.0.0.1:8180
task web                 # builds the wasm package, then the Vite dev server
```

The browser targets need WebGPU (compute shaders, no WebGL fallback).
