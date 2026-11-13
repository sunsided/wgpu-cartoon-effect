//! Building blocks shared between effects.

use crate::kernel::{Frame, Kernel, Res};

macro_rules! kernel {
    ($device:expr, $name:literal) => {
        Kernel::new($device, $name, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/", $name, ".wgsl")), "main")
    };
}
pub(crate) use kernel;

pub struct Blocks {
    to_oklab: Kernel,
    sst: Kernel,
    blur: Kernel,
    tfm: Kernel,
    fbl: Kernel,
    /// Bilinear, clamp-to-edge.
    pub linear: wgpu::Sampler,
}

impl Blocks {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            to_oklab: kernel!(device, "blocks/to_oklab"),
            sst: kernel!(device, "blocks/sst"),
            blur: kernel!(device, "blocks/blur"),
            tfm: kernel!(device, "blocks/tfm"),
            fbl: kernel!(device, "blocks/fbl"),
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    pub fn to_oklab(&self, f: &mut Frame, linear: &wgpu::Texture) -> wgpu::Texture {
        let out = f.work_texture();
        self.to_oklab.dispatch(f, &[Res::Tex(linear), Res::Tex(&out)]);
        out
    }

    /// Separable Gaussian blur of all four channels.
    pub fn blur(&self, f: &mut Frame, src: &wgpu::Texture, sigma: f32) -> wgpu::Texture {
        let tmp = f.work_texture();
        let out = f.work_texture();
        let ux = f.uniform(&[1.0, 0.0, sigma]);
        let uy = f.uniform(&[0.0, 1.0, sigma]);
        self.blur.dispatch(f, &[Res::Tex(src), Res::Tex(&tmp), Res::Buf(&ux)]);
        self.blur.dispatch(f, &[Res::Tex(&tmp), Res::Tex(&out), Res::Buf(&uy)]);
        out
    }

    /// Tangent flow map of an Oklab image: (tangent.xy, anisotropy, gradient magnitude).
    pub fn flow(&self, f: &mut Frame, lab: &wgpu::Texture, sigma: f32) -> wgpu::Texture {
        let st = f.work_texture();
        self.sst.dispatch(f, &[Res::Tex(lab), Res::Tex(&st)]);
        let st = self.blur(f, &st, sigma);
        let out = f.work_texture();
        self.tfm.dispatch(f, &[Res::Tex(&st), Res::Tex(&out)]);
        out
    }

    /// `iterations` rounds of separable flow-based bilateral filtering (gradient, then tangent).
    pub fn bilateral(
        &self,
        f: &mut Frame,
        lab: &wgpu::Texture,
        tfm: &wgpu::Texture,
        iterations: u32,
        sigma_d: f32,
        sigma_r: f32,
    ) -> wgpu::Texture {
        let ug = f.uniform(&[sigma_d, sigma_r, 1.0]);
        let ut = f.uniform(&[sigma_d, sigma_r, 0.0]);
        let mut cur = lab.clone();
        for _ in 0..iterations {
            for u in [&ug, &ut] {
                let out = f.work_texture();
                self.fbl.dispatch(f, &[Res::Tex(&cur), Res::Tex(tfm), Res::Smp(&self.linear), Res::Tex(&out), Res::Buf(u)]);
                cur = out;
            }
        }
        cur
    }
}
