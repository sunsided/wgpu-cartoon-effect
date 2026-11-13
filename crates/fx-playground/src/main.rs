//! Bare dev page for shader iteration: `trunk serve`, then `?effect=<key>&<param>=<value>`.

#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused_imports))]

#[cfg(target_arch = "wasm32")]
use fx_wasm::Renderer;
use wasm_bindgen::{JsCast, JsValue};

const SAMPLE: &[u8] = include_bytes!("../../../assets/grumpy-cat.jpg");

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("browser only: run `trunk serve` in crates/fx-playground");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run().await {
            set_status(&format!("error: {e:?}"));
        }
    });
}

#[cfg(target_arch = "wasm32")]
async fn run() -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let doc = window.document().unwrap();
    let canvas: web_sys::HtmlCanvasElement = doc.get_element_by_id("canvas").unwrap().dyn_into()?;
    let query = web_sys::UrlSearchParams::new_with_str(&window.location().search()?)?;
    let effect = query.get("effect").unwrap_or_else(|| "identity".into());
    let info = fx::effects()
        .iter()
        .find(|e| e.key == effect)
        .ok_or(format!("unknown effect {effect}"))?;
    let params: fx::Params = info
        .params
        .iter()
        .filter_map(|p| Some((p.key.to_owned(), query.get(p.key)?.parse().ok()?)))
        .collect();

    let img = image::load_from_memory(SAMPLE)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    let mut renderer = Renderer::create(canvas).await?;
    renderer.set_image(&img, img.width(), img.height())?;
    let start = web_sys::js_sys::Date::now();
    renderer.render(&effect, &serde_json::to_string(&params).unwrap())?;
    let keys: Vec<_> = info
        .params
        .iter()
        .map(|p| format!("{}={}", p.key, p.get(&params)))
        .collect();
    set_status(&format!(
        "{effect} {} ({:.0} ms)",
        keys.join(" "),
        web_sys::js_sys::Date::now() - start
    ));
    Ok(())
}

fn set_status(s: &str) {
    let doc = web_sys::window().unwrap().document().unwrap();
    doc.get_element_by_id("status")
        .unwrap()
        .set_text_content(Some(s));
}
