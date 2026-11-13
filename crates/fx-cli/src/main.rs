use std::path::PathBuf;

use anyhow::{Context, bail};
use clap::Parser;
use fx::{Params, Processor};

/// Apply an image effect on the GPU.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Effect key (see --list).
    #[arg(required_unless_present = "list")]
    effect: Option<String>,
    /// Input image.
    #[arg(required_unless_present = "list")]
    input: Option<PathBuf>,
    /// Output image (format from extension).
    #[arg(short, long, default_value = "out.png")]
    output: PathBuf,
    /// Parameter override, repeatable: --set key=value
    #[arg(short = 's', long = "set", value_parser = parse_kv)]
    params: Vec<(String, f32)>,
    /// List effects and their parameters as JSON.
    #[arg(long)]
    list: bool,
}

fn parse_kv(s: &str) -> Result<(String, f32), String> {
    let (k, v) = s.split_once('=').ok_or("expected key=value")?;
    Ok((k.to_owned(), v.parse().map_err(|e| format!("{v:?}: {e}"))?))
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    if args.list {
        println!("{}", serde_json::to_string_pretty(fx::effects())?);
        return Ok(());
    }
    let (Some(effect), Some(input)) = (args.effect, args.input) else { unreachable!("clap enforces") };
    let Some(info) = fx::effects().iter().find(|e| e.key == effect) else {
        bail!("unknown effect {effect:?}, see --list");
    };
    for (k, _) in &args.params {
        if !info.params.iter().any(|p| p.key == k) {
            bail!("effect {effect:?} has no parameter {k:?}");
        }
    }

    let img = image::open(&input).with_context(|| format!("reading {}", input.display()))?.into_rgba8();
    let mut p = pollster::block_on(Processor::headless())?;
    p.set_image(&img, img.width(), img.height());
    let start = std::time::Instant::now();
    p.render(&effect, &args.params.into_iter().collect::<Params>())?;
    let out = p.read_output()?;
    log::info!("{effect}: {:?}", start.elapsed());
    image::RgbaImage::from_raw(img.width(), img.height(), out)
        .context("output size")?
        .save(&args.output)
        .with_context(|| format!("writing {}", args.output.display()))?;
    Ok(())
}
