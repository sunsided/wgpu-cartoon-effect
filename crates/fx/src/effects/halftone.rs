//! Halftone print: round dots on rotated screens, one per ink, drawn with the `rotated-grid`
//! crate's WGSL lookups. Three variants share the shader: mono, duotone and CMYK.

use super::p;
use crate::blocks::Blocks;
use crate::kernel::{Frame, Kernel, Res};
use crate::{EffectInfo, ParamDesc, Params};
use rotated_grid::Grid;

const CELL: ParamDesc = p("cell", "Dot spacing", 3.0, 32.0, 0.5, 8.0);
const HUE: ParamDesc = p("hue", "Spot ink hue", 0.0, 360.0, 1.0, 230.0);

pub const HALFTONE: EffectInfo = EffectInfo {
    key: "halftone",
    label: "Halftone",
    description: "Black ink dots on a 45° screen, sized by the tone of the area they cover.",
    params: &[CELL],
};

pub const DUOTONE: EffectInfo = EffectInfo {
    key: "duotone",
    label: "Duotone",
    description: "Two-ink halftone: a colored spot ink carries the tones, black screened at a \
                  different angle deepens the shadows.",
    params: &[CELL, HUE],
};

pub const CMYK: EffectInfo = EffectInfo {
    key: "cmyk",
    label: "CMYK print",
    description: "Four-color process print: cyan, magenta, yellow and black dots on screens at \
                  15°, 75°, 0° and 45°, overlapping into rosettes.",
    params: &[CELL],
};

/// Which inks to print, as the shader's `mode`.
#[derive(Clone, Copy)]
pub enum Mode {
    Mono = 0,
    Duotone = 1,
    Cmyk = 2,
}

pub struct Halftone {
    kernel: Kernel,
}

impl Halftone {
    pub fn new(device: &wgpu::Device) -> Self {
        let source = format!(
            "{}\n{}",
            rotated_grid::WGSL,
            include_str!("../shaders/halftone/halftone.wgsl")
        );
        Self {
            kernel: Kernel::new(device, "halftone/halftone", &source, "main"),
        }
    }

    pub fn run(
        &self,
        b: &Blocks,
        f: &mut Frame,
        input: &wgpu::Texture,
        p: &Params,
        mode: Mode,
    ) -> wgpu::Texture {
        let cell = CELL.get(p);
        // Each dot samples its center, so average over roughly the cell it stands for.
        let src = b.blur(f, input, cell / 3.0);
        let grid = Grid::new(f.size.0.into(), f.size.1.into()).spacing(cell.into(), cell.into());
        // Classic screen angles; duotone puts its spot ink in slot 0 (75°) against black at 45°.
        let angles = match mode {
            Mode::Duotone => [75.0, 0.0, 0.0, 45.0],
            Mode::Mono | Mode::Cmyk => [15.0, 75.0, 0.0, 45.0],
        }
        .map(f32::to_radians);
        let mut values = grid.to_uniform().to_vec();
        values.extend(angles);
        values.extend([mode as i32 as f32, HUE.get(p).to_radians()]);
        let u = f.uniform(&values);
        let out = f.work_texture();
        self.kernel
            .dispatch(f, &[Res::Tex(&src), Res::Tex(&out), Res::Buf(&u)]);
        out
    }
}
