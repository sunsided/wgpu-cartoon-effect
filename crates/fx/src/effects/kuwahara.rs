//! Anisotropic Kuwahara filter (Kyprianidis et al. 2009) with polynomial weights (2011):
//! oil-paint look, flat regions with crisp, direction-aware edges.

use super::p;
use crate::blocks::{Blocks, kernel};
use crate::kernel::{Frame, Kernel, Res};
use crate::{EffectInfo, ParamDesc, Params};

const FLOW_SIGMA: ParamDesc = p("flow_sigma", "Flow smoothing", 0.5, 8.0, 0.1, 2.0);
const RADIUS: ParamDesc = p("radius", "Radius", 2.0, 20.0, 0.5, 6.0);
const Q: ParamDesc = p("q", "Sharpness", 1.0, 18.0, 0.5, 8.0);
const ALPHA: ParamDesc = p("alpha", "Anisotropy", 0.1, 4.0, 0.05, 1.0);
const VAR_SCALE: ParamDesc = p("var_scale", "Variance scale", 10.0, 3000.0, 10.0, 1000.0);
const PASSES: ParamDesc = p("passes", "Passes", 1.0, 4.0, 1.0, 1.0);

pub const INFO: EffectInfo = EffectInfo {
    key: "kuwahara",
    label: "Anisotropic Kuwahara",
    params: &[FLOW_SIGMA, RADIUS, Q, ALPHA, VAR_SCALE, PASSES],
};

pub struct Kuwahara {
    akf: Kernel,
}

impl Kuwahara {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            akf: kernel!(device, "kuwahara/akf"),
        }
    }

    /// Filters an Oklab image, returning Oklab. Reused as a pre-pass by other effects.
    pub fn filter(
        &self,
        b: &Blocks,
        f: &mut Frame,
        lab: &wgpu::Texture,
        p: &Params,
    ) -> wgpu::Texture {
        let u = f.uniform(&[RADIUS.get(p), Q.get(p), ALPHA.get(p), VAR_SCALE.get(p)]);
        let mut cur = lab.clone();
        for _ in 0..PASSES.get(p) as u32 {
            // Each pass re-estimates the flow so later passes follow the cleaned-up edges.
            let tfm = b.flow(f, &cur, FLOW_SIGMA.get(p));
            let out = f.work_texture();
            self.akf.dispatch(
                f,
                &[Res::Tex(&cur), Res::Tex(&tfm), Res::Tex(&out), Res::Buf(&u)],
            );
            cur = out;
        }
        cur
    }

    pub fn run(
        &self,
        b: &Blocks,
        f: &mut Frame,
        input: &wgpu::Texture,
        p: &Params,
    ) -> wgpu::Texture {
        let lab = b.linear_to_oklab(f, input);
        let lab = self.filter(b, f, &lab, p);
        b.oklab_to_linear(f, &lab)
    }
}
