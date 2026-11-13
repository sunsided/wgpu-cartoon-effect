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
    let max_err = img.iter().zip(&out).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
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
        assert!(out.chunks(4).any(|c| c != &out[..4]), "{} output is constant", e.key);
    }
}

#[test]
fn unknown_effect_errors() {
    let mut p = processor();
    let (img, w, h) = pattern();
    p.set_image(&img, w, h);
    assert!(matches!(p.render("nope", &Params::new()), Err(fx::Error::UnknownEffect(_))));
}
