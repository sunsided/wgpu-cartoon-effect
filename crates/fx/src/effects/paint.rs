//! Painterly rendering with curved brush strokes (Hertzmann 1998), on the GPU:
//! coarse-to-fine layers; per layer a compute pass places strokes where the canvas still
//! differs from a blurred reference, then they are drawn as instanced bristle-textured strips.

use super::{kuwahara::Kuwahara, p};
use crate::blocks::{Blocks, kernel};
use crate::kernel::{Frame, Kernel, Res, WORK_FORMAT, shader_module, view};
use crate::{EffectInfo, ParamDesc, Params};

const BRUSH: ParamDesc = p("brush", "Largest brush radius", 2.0, 64.0, 1.0, 16.0);
const LAYERS: ParamDesc = p("layers", "Layers", 1.0, 5.0, 1.0, 4.0);
const THRESHOLD: ParamDesc = p("threshold", "Repaint threshold", 0.0, 0.3, 0.005, 0.03);
const BLUR: ParamDesc = p("blur", "Reference blur", 0.1, 2.0, 0.05, 0.5);
const MAX_LEN: ParamDesc = p("max_len", "Max stroke length", 1.0, 15.0, 1.0, 10.0);
const MIN_LEN: ParamDesc = p("min_len", "Min stroke length", 0.0, 8.0, 1.0, 2.0);
const CURVATURE: ParamDesc = p("curvature", "Curvature filter", 0.0, 0.95, 0.05, 0.5);
const OPACITY: ParamDesc = p("opacity", "Opacity", 0.1, 1.0, 0.01, 0.9);
const BRISTLES: ParamDesc = p("bristles", "Bristles", 0.0, 1.0, 0.01, 0.6);
const DRY: ParamDesc = p("dry", "Dry brush", 0.0, 1.0, 0.01, 0.4);
const JITTER_L: ParamDesc = p("jitter_l", "Value jitter", 0.0, 0.2, 0.005, 0.03);
const JITTER_AB: ParamDesc = p("jitter_ab", "Hue jitter", 0.0, 0.1, 0.002, 0.01);
const PREFILTER: ParamDesc = p("prefilter", "Kuwahara pre-pass", 0.0, 1.0, 1.0, 1.0);
const SEED: ParamDesc = p("seed", "Seed", 0.0, 100.0, 1.0, 0.0);

pub const INFO: EffectInfo = EffectInfo {
    key: "paint",
    label: "Brush strokes",
    params: &[
        BRUSH, LAYERS, THRESHOLD, BLUR, MAX_LEN, MIN_LEN, CURVATURE, OPACITY, BRISTLES, DRY,
        JITTER_L, JITTER_AB, PREFILTER, SEED,
    ],
};

/// Bytes per `Stroke` in the shaders (vec4 + 4×u32 + 16×vec2).
const STROKE_SIZE: u64 = 16 + 16 + 16 * 8;
/// Strip vertices per stroke: two per point plus both caps.
const STRIP_VERTICES: u32 = 2 * (16 + 2);
/// Smallest brush radius; finer grids get expensive and look like noise.
const MIN_RADIUS: f32 = 2.0;

pub struct Paint {
    stroke: Kernel,
    render: wgpu::RenderPipeline,
}

impl Paint {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = shader_module(
            device,
            "paint/render",
            include_str!("../shaders/paint/render.wgsl"),
        );
        let render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("paint/render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: WORK_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            stroke: kernel!(device, "paint/stroke"),
            render,
        }
    }

    pub fn run(
        &self,
        b: &Blocks,
        kuwahara: &Kuwahara,
        f: &mut Frame,
        input: &wgpu::Texture,
        p: &Params,
    ) -> wgpu::Texture {
        let lab = b.linear_to_oklab(f, input);
        let source = if PREFILTER.get(p) > 0.5 {
            kuwahara.filter(b, f, &lab, &Params::new())
        } else {
            lab
        };
        let canvas = f.work_texture();
        let canvas_view = view(&canvas);
        let (w, h) = f.size;
        let seed = SEED.get(p);

        for layer in 0..LAYERS.get(p) as u32 {
            let r = (BRUSH.get(p) / (1 << layer) as f32).max(MIN_RADIUS);
            let reference = b.blur(f, &source, BLUR.get(p) * r);
            // Big brushes follow broad structure, small ones the details.
            let tfm = b.flow(f, &reference, (0.5 * r).max(1.0));

            let cols = (w as f32 / r).ceil() as u32;
            let rows = (h as f32 / r).ceil() as u32;
            let count = cols * rows;
            let strokes = f.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("strokes"),
                size: u64::from(count) * STROKE_SIZE,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let u = f.uniform(&[
                r,
                THRESHOLD.get(p),
                MAX_LEN.get(p),
                MIN_LEN.get(p),
                CURVATURE.get(p),
                if layer == 0 { 1.0 } else { 0.0 },
                cols as f32,
                seed,
            ]);
            self.stroke.dispatch_size(
                f,
                &[
                    Res::Tex(&reference),
                    Res::Tex(&canvas),
                    Res::Tex(&tfm),
                    Res::Smp(&b.linear),
                    Res::Buf(&strokes),
                    Res::Buf(&u),
                ],
                (cols, rows),
            );

            let bits = count.next_power_of_two().trailing_zeros().max(1);
            let u = f.uniform(&[
                OPACITY.get(p),
                BRISTLES.get(p),
                DRY.get(p),
                JITTER_L.get(p),
                count as f32,
                bits as f32,
                // Different shuffle per layer and seed.
                (seed + 1.0) * 7919.0 + layer as f32 * 104729.0,
                JITTER_AB.get(p),
                w as f32,
                h as f32,
            ]);
            let bind_group = f.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.render.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: strokes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u.as_entire_binding(),
                    },
                ],
            });
            let load = if layer == 0 {
                // Off-white paper in Oklab; the first layer covers it anyway.
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.96,
                    g: 0.0,
                    b: 0.01,
                    a: 1.0,
                })
            } else {
                wgpu::LoadOp::Load
            };
            let mut pass = f.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &canvas_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.render);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..STRIP_VERTICES, 0..count);
        }
        b.oklab_to_linear(f, &canvas)
    }
}
