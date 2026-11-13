use fx::{Params, Processor};

fn processor() -> Processor {
    pollster::block_on(Processor::headless()).expect("GPU adapter")
}

/// Deterministic 37x23 test pattern (odd size catches workgroup-edge bugs).
fn pattern() -> (Vec<u8>, u32, u32) {
    let (w, h) = (37, 23);
    let px = (0..w * h).flat_map(|i| {
        let (x, y) = (i % w, i / w);
        [(x * 7) as u8, (y * 11) as u8, ((x ^ y) * 5) as u8, 255]
    });
    (px.collect(), w, h)
}

#[test]
fn identity_round_trips() {
    let mut p = processor();
    let (img, w, h) = pattern();
    p.set_image(&img, w, h);
    p.render("identity", &Params::new()).unwrap();
    let out = p.read_output().unwrap();
    let max_err = img
        .iter()
        .zip(&out)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(max_err <= 1, "max error {max_err}");
}

#[test]
fn every_effect_runs() {
    let mut p = processor();
    let (img, w, h) = pattern();
    p.set_image(&img, w, h);
    for e in fx::effects() {
        p.render(e.key, &Params::new()).unwrap();
        let out = p.read_output().unwrap();
        assert_eq!(out.len(), img.len(), "{}", e.key);
        assert!(
            out.chunks(4).any(|c| c != &out[..4]),
            "{} output is constant",
            e.key
        );
    }
}

#[test]
fn unknown_effect_errors() {
    let mut p = processor();
    let (img, w, h) = pattern();
    p.set_image(&img, w, h);
    assert!(matches!(
        p.render("nope", &Params::new()),
        Err(fx::Error::UnknownEffect(_))
    ));
}

fn render(p: &mut Processor, effect: &str, params: &[(&str, f32)]) -> Vec<u8> {
    let params: Params = params.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    p.render(effect, &params).unwrap();
    p.read_output().unwrap()
}

#[test]
fn paint_is_deterministic() {
    let mut p = processor();
    let (img, w, h) = pattern();
    p.set_image(&img, w, h);
    assert_eq!(render(&mut p, "paint", &[]), render(&mut p, "paint", &[]));
    assert_ne!(render(&mut p, "paint", &[]), render(&mut p, "paint", &[("seed", 1.0)]));
}

#[test]
fn splat_render_scale_reuses_fit() {
    let mut p = processor();
    let (img, w, h) = pattern();
    p.set_image(&img, w, h);
    let fit = [("count", 200.0), ("iterations", 30.0)];
    let a = render(&mut p, "splat", &[fit[0], fit[1], ("blob_scale", 1.0)]);
    let b = render(&mut p, "splat", &[fit[0], fit[1], ("blob_scale", 0.5)]);
    assert_ne!(a, b);
    assert_eq!(a, render(&mut p, "splat", &[fit[0], fit[1], ("blob_scale", 1.0)]));
    // A new image must invalidate the cached fit.
    let flipped: Vec<u8> = img.chunks(4).rev().flatten().copied().collect();
    p.set_image(&flipped, w, h);
    assert_ne!(a, render(&mut p, "splat", &[fit[0], fit[1], ("blob_scale", 1.0)]));
}
