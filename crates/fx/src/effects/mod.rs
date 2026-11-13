mod cartoon;
mod kuwahara;
mod paint;
mod splat;

use crate::blocks::Blocks;
use crate::kernel::Frame;
use crate::{EffectInfo, Error, ParamDesc, Params};

const fn p(
    key: &'static str,
    label: &'static str,
    min: f32,
    max: f32,
    step: f32,
    default: f32,
) -> ParamDesc {
    ParamDesc {
        key,
        label,
        min,
        max,
        step,
        default,
    }
}

/// All effects, in the order frontends list them.
pub static EFFECTS: &[EffectInfo] = &[
    EffectInfo {
        key: "identity",
        label: "Original",
        params: &[],
    },
    cartoon::INFO,
    kuwahara::INFO,
    paint::INFO,
    splat::INFO,
];

/// Compiled pipelines of every effect.
pub struct Effects {
    blocks: Blocks,
    cartoon: cartoon::Cartoon,
    kuwahara: kuwahara::Kuwahara,
    paint: paint::Paint,
    splat: splat::Splat,
}

impl Effects {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            blocks: Blocks::new(device),
            cartoon: cartoon::Cartoon::new(device),
            kuwahara: kuwahara::Kuwahara::new(device),
            paint: paint::Paint::new(device),
            splat: splat::Splat::new(device),
        }
    }

    /// Runs effect `key` on the linear working-format `input`, returning a linear result.
    pub fn run(
        &self,
        key: &str,
        f: &mut Frame,
        input: &wgpu::Texture,
        params: &Params,
    ) -> Result<wgpu::Texture, Error> {
        let b = &self.blocks;
        Ok(match key {
            "identity" => input.clone(),
            "cartoon" => self.cartoon.run(b, f, input, params),
            "kuwahara" => self.kuwahara.run(b, f, input, params),
            "paint" => self.paint.run(b, &self.kuwahara, f, input, params),
            "splat" => self.splat.run(b, f, input, params),
            _ => return Err(Error::UnknownEffect(key.to_owned())),
        })
    }
}
