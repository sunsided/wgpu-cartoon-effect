//! Browser bindings: renders effects into a `<canvas>` via WebGPU.
#![cfg(target_arch = "wasm32")]

use fx::{Params, Presenter, Processor};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// JSON array of all effects and their parameter descriptions.
#[wasm_bindgen]
pub fn effects() -> String {
    fx::effects_json()
}

#[wasm_bindgen]
pub struct Renderer {
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    presenter: Presenter,
    processor: Processor,
}

#[wasm_bindgen]
impl Renderer {
    /// Requests a WebGPU device and binds it to `canvas`.
    pub async fn create(canvas: HtmlCanvasElement) -> Result<Renderer, JsError> {
        console_error_panic_hook::set_once();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| JsError::new(&format!("WebGPU unavailable: {e}")))?;
        let (device, queue) = adapter.request_device(&Default::default()).await?;
        let caps = surface.get_capabilities(&adapter);
        // Output holds sRGB-encoded bytes, so present to a non-sRGB format to pass them through.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            format,
            ..surface
                .get_default_config(&adapter, canvas.width().max(1), canvas.height().max(1))
                .ok_or_else(|| JsError::new("surface not supported by adapter"))?
        };
        surface.configure(&device, &config);
        Ok(Self {
            presenter: Presenter::new(&device, format),
            processor: Processor::new(device, queue),
            canvas,
            surface,
            config,
        })
    }

    /// Sets the source image (tightly packed sRGB RGBA) and resizes the canvas to match.
    #[wasm_bindgen(js_name = setImage)]
    pub fn set_image(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<(), JsError> {
        if rgba.len() != (width * height * 4) as usize {
            return Err(JsError::new(
                "rgba length does not match width * height * 4",
            ));
        }
        let max = self.processor.device().limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max || height > max {
            return Err(JsError::new(&format!(
                "image size must be 1..={max} per side"
            )));
        }
        self.processor.set_image(rgba, width, height);
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.config.width = width;
        self.config.height = height;
        self.surface
            .configure(self.processor.device(), &self.config);
        Ok(())
    }

    /// Renders `effect` with `params` (JSON object of key → number) and presents it.
    pub fn render(&mut self, effect: &str, params: &str) -> Result<(), JsError> {
        let params: Params = if params.is_empty() {
            Params::new()
        } else {
            serde_json::from_str(params)?
        };
        let output = self.processor.render(effect, &params)?.clone();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => return Err(JsError::new(&format!("surface unavailable: {other:?}"))),
        };
        let device = self.processor.device();
        let mut encoder = device.create_command_encoder(&Default::default());
        self.presenter.draw(
            device,
            &mut encoder,
            &output,
            &frame.texture.create_view(&Default::default()),
        );
        self.processor.queue().submit([encoder.finish()]);
        self.processor.queue().present(frame);
        Ok(())
    }
}
