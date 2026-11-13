//! Image approximated by N anisotropic 2D Gaussians fitted with Adam (in the spirit of
//! Image-GS / 2D Gaussian splatting): few splats give soft, painterly abstraction.
//! The normalized weighted sum makes the result independent of splat order, so the
//! backward pass can run per splat without float atomics.

use std::cell::RefCell;

use super::p;
use crate::blocks::Blocks;
use crate::kernel::{Frame, Kernel, Res};
use crate::{EffectInfo, ParamDesc, Params};

const COUNT: ParamDesc = p("count", "Splats", 100.0, 30000.0, 100.0, 1500.0);
const ITERATIONS: ParamDesc = p("iterations", "Iterations", 0.0, 1000.0, 10.0, 200.0);
const LR: ParamDesc = p("lr", "Learning rate", 0.1, 5.0, 0.05, 1.0);
const EDGE_BIAS: ParamDesc = p("edge_bias", "Detail bias", 0.0, 1.0, 0.01, 0.7);
const MAX_SIGMA: ParamDesc = p("max_sigma", "Max splat size", 2.0, 64.0, 1.0, 24.0);
const BLOB_SCALE: ParamDesc = p("blob_scale", "Render scale", 0.3, 2.0, 0.01, 0.6);
const SEED: ParamDesc = p("seed", "Seed", 0.0, 100.0, 1.0, 0.0);

pub const INFO: EffectInfo = EffectInfo {
    key: "splat",
    label: "Gaussian splats",
    params: &[
        COUNT, ITERATIONS, LR, EDGE_BIAS, MAX_SIGMA, BLOB_SCALE, SEED,
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
            cache: RefCell::new(None),
        }
    }

    pub fn run(
        &self,
        b: &Blocks,
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
        };

        if fitted {
            forward(f, BLOB_SCALE.get(p));
            return b.oklab_to_linear(f, &cw);
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
        forward(f, BLOB_SCALE.get(p));
        *self.cache.borrow_mut() = Some((key, splats.clone()));
        b.oklab_to_linear(f, &cw)
    }
}

/// Identifies a fit: the image and every parameter that affects the optimization.
#[derive(PartialEq)]
struct FitKey {
    image: u64,
    params: [u32; 6],
}
