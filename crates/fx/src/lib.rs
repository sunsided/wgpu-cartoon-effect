//! Image stylization effects on wgpu.
//!
//! Pipeline per render: 8-bit sRGB source → linear `Rgba16Float` → effect → sRGB-encoded
//! `Rgba8Unorm` output, which can be presented ([`Presenter`]) or read back.

mod blocks;
mod effects;
pub mod kernel;
mod params;
mod present;

use kernel::{Frame, Kernel, OUTPUT_FORMAT, Res};
pub use params::{EffectInfo, ParamDesc, Params};
pub use present::Presenter;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown effect {0:?}")]
    UnknownEffect(String),
    #[error("no image set")]
    NoImage,
    #[error("no GPU adapter: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("device request failed: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("readback failed: {0}")]
    Readback(String),
}

/// Lists all available effects and their parameters.
pub fn effects() -> &'static [EffectInfo] {
    effects::EFFECTS
}

pub struct Processor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    source: Option<wgpu::Texture>,
    output: Option<wgpu::Texture>,
    decode: Kernel,
    encode: Kernel,
    effects: effects::Effects,
}

impl Processor {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            decode: Kernel::new(&device, "decode", include_str!("shaders/decode.wgsl"), "decode"),
            encode: Kernel::new(&device, "encode", include_str!("shaders/encode.wgsl"), "encode"),
            effects: effects::Effects::new(&device),
            device,
            queue,
            source: None,
            output: None,
        }
    }

    /// Creates a processor on a fresh device without any surface.
    /// Set `WGPU_FORCE_FALLBACK_ADAPTER=1` to use a software adapter.
    pub async fn headless() -> Result<Self, Error> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: std::env::var_os("WGPU_FORCE_FALLBACK_ADAPTER").is_some(),
                compatible_surface: None,
                ..Default::default()
            })
            .await?;
        log::info!("adapter: {:?}", adapter.get_info());
        let (device, queue) = adapter.request_device(&Default::default()).await?;
        Ok(Self::new(device, queue))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The last rendered output, sRGB-encoded `Rgba8Unorm`.
    pub fn output(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref()
    }

    /// Uploads a tightly packed 8-bit sRGB RGBA image.
    pub fn set_image(&mut self, rgba: &[u8], width: u32, height: u32) {
        assert_eq!(rgba.len(), (width * height * 4) as usize, "rgba buffer size mismatch");
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("source"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 4), rows_per_image: None },
            size,
        );
        self.source = Some(texture);
        self.output = None;
    }

    /// Runs `effect` on the current image and submits the work.
    pub fn render(&mut self, effect: &str, params: &Params) -> Result<&wgpu::Texture, Error> {
        let source = self.source.as_ref().ok_or(Error::NoImage)?;
        let mut f = Frame {
            device: &self.device,
            encoder: self.device.create_command_encoder(&Default::default()),
            size: (source.width(), source.height()),
        };
        let linear = f.work_texture();
        self.decode.dispatch(&mut f, &[Res::Tex(source), Res::Tex(&linear)]);
        let result = self.effects.run(effect, &mut f, &linear, params)?;
        let output = f.texture(OUTPUT_FORMAT);
        self.encode.dispatch(&mut f, &[Res::Tex(&result), Res::Tex(&output)]);
        self.queue.submit([f.encoder.finish()]);
        Ok(self.output.insert(output))
    }

    /// Blocks until the output is read back as tightly packed sRGB RGBA bytes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_output(&self) -> Result<Vec<u8>, Error> {
        let output = self.output.as_ref().ok_or(Error::NoImage)?;
        let (w, h) = (output.width(), output.height());
        let padded = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: None },
            },
            output.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| _ = tx.send(r));
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| Error::Readback(e.to_string()))?;
        rx.recv()
            .map_err(|e| Error::Readback(e.to_string()))?
            .map_err(|e| Error::Readback(e.to_string()))?;
        let mapped = buffer.get_mapped_range(..).map_err(|e| Error::Readback(e.to_string()))?;
        Ok(mapped.chunks(padded as usize).flat_map(|row| &row[..(w * 4) as usize]).copied().collect())
    }
}

/// [`effects`] as JSON, for frontends.
pub fn effects_json() -> String {
    serde_json::to_string(effects::EFFECTS).expect("static data serializes")
}
