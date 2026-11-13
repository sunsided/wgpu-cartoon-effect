mod cartoon;

use crate::blocks::Blocks;
use crate::kernel::Frame;
use crate::{EffectInfo, Error, Params};

/// All effects, in the order frontends list them.
pub static EFFECTS: &[EffectInfo] =
    &[EffectInfo { key: "identity", label: "Original", params: &[] }, cartoon::INFO];

/// Compiled pipelines of every effect.
pub struct Effects {
    blocks: Blocks,
    cartoon: cartoon::Cartoon,
}

impl Effects {
    pub fn new(device: &wgpu::Device) -> Self {
        Self { blocks: Blocks::new(device), cartoon: cartoon::Cartoon::new(device) }
    }

    /// Runs effect `key` on the linear working-format `input`, returning a linear result.
    pub fn run(&self, key: &str, f: &mut Frame, input: &wgpu::Texture, params: &Params) -> Result<wgpu::Texture, Error> {
        let b = &self.blocks;
        Ok(match key {
            "identity" => input.clone(),
            "cartoon" => self.cartoon.run(b, f, input, params),
            _ => return Err(Error::UnknownEffect(key.to_owned())),
        })
    }
}
