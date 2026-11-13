//! Cartoon: flow-based bilateral abstraction, soft luminance quantization and FDoG/XDoG ink lines
//! (Winnemöller et al. 2006, Kang et al. 2007, Winnemöller 2011).

use crate::blocks::{Blocks, kernel};
use crate::kernel::{Frame, Kernel, Res};
use crate::{EffectInfo, ParamDesc, Params};

const fn p(key: &'static str, label: &'static str, min: f32, max: f32, step: f32, default: f32) -> ParamDesc {
    ParamDesc { key, label, min, max, step, default }
}

const FLOW_SIGMA: ParamDesc = p("flow_sigma", "Flow smoothing", 0.5, 8.0, 0.1, 2.5);
const ITERATIONS: ParamDesc = p("iterations", "Bilateral iterations", 0.0, 8.0, 1.0, 3.0);
const SIGMA_D: ParamDesc = p("sigma_d", "Bilateral radius", 0.5, 8.0, 0.1, 3.0);
const SIGMA_R: ParamDesc = p("sigma_r", "Bilateral color range", 0.01, 0.3, 0.005, 0.06);
const LEVELS: ParamDesc = p("levels", "Lightness levels", 2.0, 24.0, 1.0, 8.0);
const SHARPNESS: ParamDesc = p("sharpness", "Level sharpness", 0.1, 10.0, 0.1, 2.5);
const EDGE_SIGMA: ParamDesc = p("edge_sigma", "Line width", 0.3, 4.0, 0.05, 1.2);
const EDGE_TAU: ParamDesc = p("edge_tau", "Line sensitivity", 0.9, 1.0, 0.001, 0.98);
const EDGE_EPS: ParamDesc = p("edge_eps", "Line threshold", -0.05, 0.05, 0.001, 0.0);
const EDGE_PHI: ParamDesc = p("edge_phi", "Line hardness", 1.0, 300.0, 1.0, 200.0);
const LINE_FLOW: ParamDesc = p("line_flow", "Line coherence", 0.5, 10.0, 0.1, 3.0);
const LINE_STRENGTH: ParamDesc = p("line_strength", "Line strength", 0.0, 1.0, 0.01, 1.0);

pub const INFO: EffectInfo = EffectInfo {
    key: "cartoon",
    label: "Cartoon",
    params: &[
        FLOW_SIGMA, ITERATIONS, SIGMA_D, SIGMA_R, LEVELS, SHARPNESS, EDGE_SIGMA, EDGE_TAU, EDGE_EPS, EDGE_PHI,
        LINE_FLOW, LINE_STRENGTH,
    ],
};

pub struct Cartoon {
    dog: Kernel,
    fdog: Kernel,
    composite: Kernel,
}

impl Cartoon {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            dog: kernel!(device, "cartoon/dog"),
            fdog: kernel!(device, "cartoon/fdog"),
            composite: kernel!(device, "cartoon/composite"),
        }
    }

    pub fn run(&self, b: &Blocks, f: &mut Frame, input: &wgpu::Texture, p: &Params) -> wgpu::Texture {
        let lab = b.to_oklab(f, input);
        let tfm = b.flow(f, &lab, FLOW_SIGMA.get(p));
        let smooth = b.bilateral(f, &lab, &tfm, ITERATIONS.get(p) as u32, SIGMA_D.get(p), SIGMA_R.get(p));

        let dog = f.work_texture();
        let u = f.uniform(&[EDGE_SIGMA.get(p), EDGE_TAU.get(p)]);
        self.dog.dispatch(f, &[Res::Tex(&smooth), Res::Tex(&tfm), Res::Smp(&b.linear), Res::Tex(&dog), Res::Buf(&u)]);
        let edges = f.work_texture();
        let u = f.uniform(&[LINE_FLOW.get(p), EDGE_EPS.get(p), EDGE_PHI.get(p)]);
        self.fdog.dispatch(f, &[Res::Tex(&dog), Res::Tex(&tfm), Res::Smp(&b.linear), Res::Tex(&edges), Res::Buf(&u)]);

        let out = f.work_texture();
        let u = f.uniform(&[LEVELS.get(p), SHARPNESS.get(p), LINE_STRENGTH.get(p)]);
        self.composite.dispatch(f, &[Res::Tex(&smooth), Res::Tex(&edges), Res::Tex(&out), Res::Buf(&u)]);
        out
    }
}
