//! Image approximated by N anisotropic 2D Gaussians fitted with Adam (in the spirit of
//! Image-GS / 2D Gaussian splatting): few splats give soft, painterly abstraction.
//! The normalized weighted sum makes the result independent of splat order, so the
//! backward pass can run per splat without float atomics.

use std::cell::RefCell;

use super::p;
use super::paint::{Paint, STROKE_SIZE, StrokeStyle};
use crate::blocks::Blocks;
use crate::kernel::{Frame, Kernel, Res, view};
use crate::{EffectInfo, ParamDesc, Params};

const COUNT: ParamDesc = p("count", "Splats", 100.0, 30000.0, 100.0, 1500.0);
const ITERATIONS: ParamDesc = p("iterations", "Iterations", 0.0, 1000.0, 10.0, 200.0);
const LR: ParamDesc = p("lr", "Learning rate", 0.1, 5.0, 0.05, 1.0);
const EDGE_BIAS: ParamDesc = p("edge_bias", "Detail bias", 0.0, 1.0, 0.01, 0.7);
const MAX_SIGMA: ParamDesc = p("max_sigma", "Max splat size", 2.0, 64.0, 1.0, 24.0);
const BLOB_SCALE: ParamDesc = p("blob_scale", "Render scale", 0.3, 2.0, 0.01, 0.6);
const SEED: ParamDesc = p("seed", "Seed", 0.0, 100.0, 1.0, 0.0);
const STYLE: ParamDesc = p("style", "Render as strokes", 0.0, 1.0, 1.0, 0.0);
const STROKE_LENGTH: ParamDesc = p("stroke_length", "Stroke length", 0.2, 3.0, 0.05, 0.8);
const STROKE_WIDTH: ParamDesc = p("stroke_width", "Stroke width", 0.5, 4.0, 0.05, 3.0);
const BRISTLES: ParamDesc = p("bristles", "Bristles", 0.0, 1.0, 0.01, 0.6);
const DRY: ParamDesc = p("dry", "Dry brush", 0.0, 1.0, 0.01, 0.3);

pub const INFO: EffectInfo = EffectInfo {
    key: "splat",
    label: "Gaussian splats",
    description: "Fits soft 2D Gaussians to the image, then renders them as blobs or as brush strokes. \
                  Changing a fit setting re-runs the optimizer, which takes a moment.",
    params: &[
        COUNT,
        ITERATIONS,
        LR,
        EDGE_BIAS,
        MAX_SIGMA,
        BLOB_SCALE,
        SEED,
        STYLE,
        STROKE_LENGTH,
        STROKE_WIDTH,
        BRISTLES,
        DRY,
    ],
};

/// Bytes per `Splat` (12 floats).
const SPLAT_SIZE: u64 = 48;
/// Must match `TILE` and `TILE_CAP` in shaders/splat/common.wgsl.
const TILE: u32 = 16;
const TILE_CAP: u64 = 512;

/// Respawn starved splats every this many iterations, during the first 3/4 of the fit.
const RESPAWN_EVERY: u32 = 50;
/// Splats making up less than this many pixels' worth of the image get respawned.
const MIN_CONTRIBUTION: f32 = 1.0;

const LR_POS: f32 = 0.3;
const LR_SCALE: f32 = 0.03;
const LR_ROT: f32 = 0.03;
const LR_OPACITY: f32 = 0.05;
const LR_COLOR: f32 = 0.01;

pub struct Splat {
    init: Kernel,
    bin: Kernel,
    forward: Kernel,
    backward: Kernel,
    adam: Kernel,
    respawn: Kernel,
    to_strokes: Kernel,
    /// Last fitted splats, so changing only the render scale doesn't re-run the fit.
    cache: RefCell<Option<(FitKey, wgpu::Buffer)>>,
}

macro_rules! splat_kernel {
    ($device:expr, $name:literal) => {
        Kernel::new(
            $device,
            concat!("splat/", $name),
            &format!(
                "{}\n{}",
                include_str!("../shaders/splat/common.wgsl"),
                include_str!(concat!("../shaders/splat/", $name, ".wgsl"))
            ),
            "main",
        )
    };
}

impl Splat {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            init: splat_kernel!(device, "init"),
            bin: splat_kernel!(device, "bin"),
            forward: splat_kernel!(device, "forward"),
            backward: splat_kernel!(device, "backward"),
            adam: splat_kernel!(device, "adam"),
            respawn: splat_kernel!(device, "respawn"),
            to_strokes: splat_kernel!(device, "to_strokes"),
            cache: RefCell::new(None),
        }
    }

    pub fn run(
        &self,
        b: &Blocks,
        paint: &Paint,
        f: &mut Frame,
        input: &wgpu::Texture,
        p: &Params,
    ) -> wgpu::Texture {
        let (w, h) = f.size;
        let n = COUNT.get(p) as u32;
        let seed = SEED.get(p);
        let target = b.linear_to_oklab(f, input);
        let key = FitKey {
            image: f.image_id,
            params: [COUNT, ITERATIONS, LR, EDGE_BIAS, MAX_SIGMA, SEED].map(|d| d.get(p).to_bits()),
        };
        let cached = self
            .cache
            .borrow()
            .as_ref()
            .filter(|(k, _)| *k == key)
            .map(|(_, b)| b.clone());
        let background = b.blur(f, &target, w.max(h) as f32 / 16.0);

        let buffer = |size: u64, usage: wgpu::BufferUsages| {
            f.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let fitted = cached.is_some();
        let splats = cached.unwrap_or_else(|| buffer(u64::from(n) * SPLAT_SIZE, storage));
        let tiles = (w.div_ceil(TILE), h.div_ceil(TILE));
        let tile_count = buffer(
            u64::from(tiles.0 * tiles.1) * 4,
            storage | wgpu::BufferUsages::COPY_DST,
        );
        let tile_list = buffer(u64::from(tiles.0 * tiles.1) * TILE_CAP * 4, storage);
        let cw = f.work_texture();
        let grad = f.work_texture();

        let forward = |f: &mut Frame, scale_mul: f32| {
            f.encoder.clear_buffer(&tile_count, 0, None);
            let u = f.uniform(&[n as f32, tiles.0 as f32, tiles.1 as f32, scale_mul]);
            self.bin.dispatch_linear(
                f,
                &[
                    Res::Buf(&splats),
                    Res::Buf(&tile_count),
                    Res::Buf(&tile_list),
                    Res::Buf(&u),
                ],
                n,
            );
            let u = f.uniform(&[tiles.0 as f32, scale_mul]);
            self.forward.dispatch(
                f,
                &[
                    Res::Buf(&splats),
                    Res::Buf(&tile_count),
                    Res::Buf(&tile_list),
                    Res::Tex(&target),
                    Res::Tex(&background),
                    Res::Tex(&cw),
                    Res::Tex(&grad),
                    Res::Buf(&u),
                ],
            );
            cw.clone()
        };

        if fitted {
            return self.finish(b, paint, f, p, &splats, &background, forward);
        }

        let tfm = b.flow(f, &target, 1.0);
        let grads = buffer(u64::from(n) * SPLAT_SIZE, storage);
        let m1 = buffer(u64::from(n) * SPLAT_SIZE, storage);
        let m2 = buffer(u64::from(n) * SPLAT_SIZE, storage);
        // Spacing of a regular grid with n points, halved: neighbors overlap a little.
        let init_scale = 0.5 * ((w * h) as f32 / n as f32).sqrt();
        let u = f.uniform(&[n as f32, seed + 1.0, init_scale, EDGE_BIAS.get(p)]);
        self.init.dispatch_linear(
            f,
            &[
                Res::Tex(&target),
                Res::Tex(&tfm),
                Res::Buf(&splats),
                Res::Buf(&m1),
                Res::Buf(&m2),
                Res::Buf(&u),
            ],
            n,
        );

        let lr = LR.get(p);
        let count_u = f.uniform(&[n as f32]);
        let iterations = ITERATIONS.get(p) as u32;
        for step in 1..=iterations {
            forward(f, 1.0);
            self.backward.dispatch_linear(
                f,
                &[
                    Res::Buf(&splats),
                    Res::Tex(&cw),
                    Res::Tex(&grad),
                    Res::Buf(&grads),
                    Res::Buf(&count_u),
                ],
                n,
            );
            let u = f.uniform(&[
                n as f32,
                step as f32,
                lr * LR_POS,
                lr * LR_SCALE,
                lr * LR_ROT,
                lr * LR_OPACITY,
                lr * LR_COLOR,
                MAX_SIGMA.get(p).ln(),
                w as f32,
                h as f32,
            ]);
            self.adam.dispatch_size(
                f,
                &[
                    Res::Buf(&splats),
                    Res::Buf(&grads),
                    Res::Buf(&m1),
                    Res::Buf(&m2),
                    Res::Buf(&u),
                ],
                (12, n),
            );
            // After the Adam step, so respawned splats don't receive their predecessor's update.
            if step % RESPAWN_EVERY == 0 && step * 4 < iterations * 3 {
                let u = f.uniform(&[
                    n as f32,
                    seed * 7919.0 + step as f32,
                    init_scale,
                    MIN_CONTRIBUTION,
                ]);
                self.respawn.dispatch_linear(
                    f,
                    &[
                        Res::Tex(&target),
                        Res::Tex(&grad),
                        Res::Buf(&grads),
                        Res::Buf(&splats),
                        Res::Buf(&m1),
                        Res::Buf(&m2),
                        Res::Buf(&u),
                    ],
                    n,
                );
            }
        }
        *self.cache.borrow_mut() = Some((key, splats.clone()));
        self.finish(b, paint, f, p, &splats, &background, forward)
    }

    /// Renders the fitted splats, as Gaussians or as brush strokes.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        b: &Blocks,
        paint: &Paint,
        f: &mut Frame,
        p: &Params,
        splats: &wgpu::Buffer,
        background: &wgpu::Texture,
        forward: impl Fn(&mut Frame, f32) -> wgpu::Texture,
    ) -> wgpu::Texture {
        if STYLE.get(p) < 0.5 {
            let cw = forward(f, BLOB_SCALE.get(p));
            return b.oklab_to_linear(f, &cw);
        }
        let n = COUNT.get(p) as u32;
        let strokes = f.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat strokes"),
            size: u64::from(n) * STROKE_SIZE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let u = f.uniform(&[n as f32, STROKE_LENGTH.get(p), STROKE_WIDTH.get(p)]);
        self.to_strokes.dispatch_linear(
            f,
            &[Res::Buf(splats), Res::Buf(&strokes), Res::Buf(&u)],
            n,
        );

        // Underpaint with the blurred background, then strokes from broad to fine so detail stays on top.
        let canvas = f.work_texture();
        f.encoder.copy_texture_to_texture(
            background.as_image_copy(),
            canvas.as_image_copy(),
            background.size(),
        );
        let canvas_view = view(&canvas);
        let style = StrokeStyle {
            opacity: 1.0,
            bristles: BRISTLES.get(p),
            dry: DRY.get(p),
            jitter_l: 0.02,
            jitter_ab: 0.005,
            shuffle: SEED.get(p) * 7919.0 + 1.0,
        };
        for (i, radii) in [
            (16.0, f32::INFINITY),
            (8.0, 16.0),
            (4.0, 8.0),
            (2.0, 4.0),
            (0.0, 2.0),
        ]
        .into_iter()
        .enumerate()
        {
            let style = StrokeStyle {
                shuffle: style.shuffle + i as f32 * 104729.0,
                ..style
            };
            paint.draw(
                f,
                &canvas_view,
                &strokes,
                n,
                &style,
                radii,
                wgpu::LoadOp::Load,
            );
        }
        b.oklab_to_linear(f, &canvas)
    }
}

/// Identifies a fit: the image and every parameter that affects the optimization.
#[derive(PartialEq)]
struct FitKey {
    image: u64,
    params: [u32; 6],
}
