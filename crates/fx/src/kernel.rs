//! Thin helpers around compute pipelines and transient textures.

use wgpu::util::DeviceExt;

/// Format of all intermediate (linear, high precision) images.
pub const WORK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Format of the final, sRGB-encoded output image.
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const WORKGROUP: u32 = 8;

/// WGSL shared by every kernel (color conversions, helpers).
const COMMON: &str = include_str!("shaders/common.wgsl");

/// A compute pipeline with an auto-derived layout using bind group 0.
///
/// Convention: the shader declares its resources as `@group(0) @binding(i)` in the same order
/// they are passed to [`Kernel::dispatch`], and its entry point uses `@workgroup_size(8, 8)`
/// with one invocation per output pixel.
pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
}

impl Kernel {
    pub fn new(device: &wgpu::Device, label: &str, source: &str, entry: &str) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{source}").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Records one dispatch covering every pixel of the frame.
    pub fn dispatch(&self, f: &mut Frame, resources: &[Res]) {
        let size = f.size;
        let views: Vec<_> = resources
            .iter()
            .map(|r| match r {
                Res::Tex(t) => Some(view(t)),
                Res::Buf(_) | Res::Smp(_) => None,
            })
            .collect();
        let entries: Vec<_> = resources
            .iter()
            .zip(&views)
            .enumerate()
            .map(|(i, (r, v))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: match (r, v) {
                    (_, Some(v)) => wgpu::BindingResource::TextureView(v),
                    (Res::Buf(b), None) => b.as_entire_binding(),
                    (Res::Smp(s), None) => wgpu::BindingResource::Sampler(s),
                    (Res::Tex(_), None) => unreachable!(),
                },
            })
            .collect();
        let bind_group = f.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = f.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(WORKGROUP), size.1.div_ceil(WORKGROUP), 1);
    }
}

/// A resource bound to a kernel; textures are bound as whole views.
pub enum Res<'a> {
    Tex(&'a wgpu::Texture),
    Buf(&'a wgpu::Buffer),
    Smp(&'a wgpu::Sampler),
}

/// Recording context for one effect run.
pub struct Frame<'a> {
    pub device: &'a wgpu::Device,
    pub encoder: wgpu::CommandEncoder,
    pub size: (u32, u32),
}

impl Frame<'_> {
    /// Allocates a texture usable as both compute input and storage output.
    // shortcut: allocates fresh textures per run; add a pool if re-rendering at slider rate gets slow.
    pub fn texture(&self, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: self.size.0, height: self.size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    pub fn work_texture(&self) -> wgpu::Texture {
        self.texture(WORK_FORMAT)
    }

    /// Uniform buffer for the shader-side `struct U { p0: vec4<f32>, p1: vec4<f32> }`.
    pub fn uniform(&self, values: &[f32]) -> wgpu::Buffer {
        let mut padded = [0f32; 8];
        padded[..values.len()].copy_from_slice(values);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
}

/// Shorthand for binding a whole texture view.
pub fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&Default::default())
}
