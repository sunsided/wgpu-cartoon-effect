use std::collections::BTreeMap;

use serde::Serialize;

/// User-supplied parameter values by key; missing keys fall back to [`ParamDesc::default`].
pub type Params = BTreeMap<String, f32>;

/// Describes one tweakable scalar. Frontends build their controls from this.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ParamDesc {
    pub key: &'static str,
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub default: f32,
}

impl ParamDesc {
    /// The value from `params`, clamped to range, or the default.
    pub fn get(&self, params: &Params) -> f32 {
        params.get(self.key).map_or(self.default, |v| v.clamp(self.min, self.max))
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EffectInfo {
    pub key: &'static str,
    pub label: &'static str,
    pub params: &'static [ParamDesc],
}
