//! Cross-language conformance: reproduce every vector in `pgw/tests/vectors/core_v1.json`, which is
//! generated from the Python reference by `rust/tools/gen_vectors.py`.

use std::path::PathBuf;

use pgw_core_rs::*;
use serde_json::Value;

fn vectors() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/vectors/core_v1.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let v: Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(v["format"], 1);
    assert_eq!(v["protocol"], PROTOCOL_VERSION);
    v
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0);
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn labels(v: &Value) -> Vec<Label> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|l| {
            if let Some(s) = l.get("s") {
                Label::Str(s.as_str().unwrap().to_owned())
            } else if let Some(i) = l.get("i") {
                Label::Int(i.as_i64().unwrap())
            } else {
                Label::Bytes(unhex(l["b"].as_str().unwrap()))
            }
        })
        .collect()
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f64::from_bits(u64::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
        .collect()
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
        .collect()
}

fn usize_of(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn expander(v: &Value) -> SeedExpander {
    SeedExpander::new(&unhex(v["seed"].as_str().unwrap())).unwrap()
}

#[test]
fn prng_label_encoding() {
    let v = vectors();
    for case in v["prng"]["encode_labels"].as_array().unwrap() {
        assert_eq!(hex(&encode_labels(&labels(&case["labels"])).unwrap()), case["info"].as_str().unwrap());
    }
}

#[test]
fn prng_streams_keys_and_children() {
    let v = vectors();
    let streams = v["prng"]["streams"].as_array().unwrap();
    assert!(streams.len() >= 12);
    for case in streams {
        let sx = expander(case);
        let lab = labels(&case["labels"]);
        assert_eq!(hex(sx.key(&lab).unwrap().as_slice()), case["key"].as_str().unwrap());
        let want = case["bytes"].as_str().unwrap();
        assert_eq!(hex(&sx.random_bytes(want.len() / 2, &lab).unwrap()), want);
        // every offset of the seekable stream agrees with the prefix
        for off in [0usize, 1, 63, 64, 65, 130] {
            let part = sx.stream_bytes(off as u64, 7, &lab).unwrap();
            assert_eq!(hex(&part), want[2 * off..2 * (off + 7)]);
        }
    }
    for case in v["prng"]["derive_seed"].as_array().unwrap() {
        let d = expander(case).derive_seed(&labels(&case["labels"])).unwrap();
        assert_eq!(hex(d.as_slice()), case["derived"].as_str().unwrap());
    }
}

#[test]
fn prng_ring_elements() {
    let v = vectors();
    for case in v["prng"]["ring_uniform"].as_array().unwrap() {
        let want = u64s(&case["values"]);
        let k = case["k"].as_u64().unwrap() as u32;
        assert_eq!(expander(case).ring_uniform(want.len(), k, &labels(&case["labels"])).unwrap(), want, "k={k}");
    }
    for case in v["prng"]["ring_uniform_rows"].as_array().unwrap() {
        let got = expander(case)
            .ring_uniform_rows(
                usize_of(&case["cols"]),
                case["k"].as_u64().unwrap() as u32,
                case["row_start"].as_u64().unwrap(),
                usize_of(&case["row_count"]),
                &labels(&case["labels"]),
            )
            .unwrap();
        assert_eq!(got, u64s(&case["values"]), "{case}", case = case["row_start"]);
    }
    for case in v["prng"]["uniform_int"].as_array().unwrap() {
        let want = i64s(&case["values"]);
        let got = expander(case)
            .uniform_int(
                want.len(),
                case["low"].as_i64().unwrap(),
                case["high"].as_i64().unwrap(),
                &labels(&case["labels"]),
            )
            .unwrap();
        assert_eq!(got, want);
    }
}

#[test]
fn ring_params_and_bounds() {
    let v = vectors();
    for case in v["ring"]["params"].as_array().unwrap() {
        let p = RingParams::from_name(case["name"].as_str().unwrap()).unwrap();
        assert_eq!(p.k() as u64, case["k"].as_u64().unwrap());
        assert_eq!(p.a() as u64, case["a"].as_u64().unwrap());
        assert_eq!(p.qmax(), case["qmax"].as_u64().unwrap());
        assert_eq!(p.max_in_features(), case["max_in_features"].as_u64().unwrap());
        for b in case["bounds"].as_array().unwrap() {
            let n = b["n"].as_u64().unwrap();
            assert_eq!(p.output_bound(n), b["output_bound"].as_u64().unwrap() as u128);
            assert_eq!(soundness_bits(&p, n, 2) as u64, b["soundness_bits_c2"].as_u64().unwrap(), "{p} n={n}");
        }
    }
}

#[test]
fn ring_digits_and_packing() {
    let v = vectors();
    for case in v["ring"]["balanced_digits"].as_array().unwrap() {
        let k = case["k"].as_u64().unwrap() as u32;
        let bits = case["bits"].as_u64().unwrap() as u32;
        for (val, ds) in u64s(&case["values"]).into_iter().zip(case["digits"].as_array().unwrap()) {
            let want = i64s(ds);
            assert_eq!(balanced_digits(val, k, bits).unwrap(), want, "k={k} bits={bits} v={val}");
            assert_eq!(recompose_digits(&want, k, bits), val);
        }
    }
    for case in v["ring"]["pack_ring"].as_array().unwrap() {
        let k = case["k"].as_u64().unwrap() as u32;
        let vals = u64s(&case["values"]);
        let packed = pack_ring(&vals, k).unwrap();
        assert_eq!(hex(&packed), case["packed"].as_str().unwrap());
        assert_eq!(unpack_ring(&packed, vals.len(), k).unwrap(), vals);
    }
}

#[test]
fn linear_end_to_end() {
    let v = vectors();
    let cases = v["linear"].as_array().unwrap();
    assert!(cases.len() >= 3);
    for case in cases {
        let p = RingParams::from_name(case["ring"].as_str().unwrap()).unwrap();
        let (rows, n, m, c) =
            (usize_of(&case["rows"]), usize_of(&case["n"]), usize_of(&case["m"]), usize_of(&case["checks"]));
        let lid = case["linear"].as_str().unwrap();
        let sx = expander(case);

        // quantization
        let (xq, scale_x) = quantize_activations(&f64s(&case["x_f64"]), rows, n, &p).unwrap();
        assert_eq!(xq, i64s(&case["xq"]), "{p} xq");
        assert_eq!(
            scale_x.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            f64s(&case["sx_f64"]).iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
        let (mut w8, sw) = quantize_weight_rows(&f64s(&case["w_f64"]), m, n).unwrap();
        assert_eq!(
            sw.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            f32s(&case["sw_f32"]).iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
        let want_w8: Vec<i8> = i64s(&case["w8"]).into_iter().map(|x| x as i8).collect();
        if want_w8[0] == -128 {
            w8[0] = -128; // the vector forces -128 to exercise the kernels
        }
        assert_eq!(w8, want_w8, "{p} w8");

        // masks regenerate from the seed
        let r = sx
            .ring_uniform_rows(
                n,
                p.k(),
                case["mask_row_start"].as_u64().unwrap(),
                rows,
                &[Label::from("r"), Label::from(lid)],
            )
            .unwrap();
        assert_eq!(r, u64s(&case["r"]));
        let u = sx.ring_uniform(c * m, p.k(), &[Label::from("u"), Label::from(lid)]).unwrap();
        assert_eq!(u, u64s(&case["u"]));

        // mask, server product (both algorithms), unmask
        let x_hat = mask_activations(&xq, &r, &p).unwrap();
        assert_eq!(x_hat, u64s(&case["x_hat"]));
        let digest = {
            use sha2::{Digest, Sha256};
            hex(&Sha256::digest(pack_ring(&x_hat, p.k()).unwrap()))
        };
        assert_eq!(digest, case["x_hat_packed_sha256"].as_str().unwrap());
        let y_hat = ringmm_limbs(&x_hat, rows, n, &w8, m, &p).unwrap();
        assert_eq!(y_hat, u64s(&case["y_hat"]));
        assert_eq!(ringmm_reference(&x_hat, rows, n, &w8, m, &p).unwrap(), y_hat);
        let wr = ringmm_limbs(&r, rows, n, &w8, m, &p).unwrap();
        assert_eq!(wr, u64s(&case["wr"]));
        let check_range = if w8.contains(&-128) { None } else { Some(n as u64) };
        let yq = unmask_outputs(&y_hat, &wr, &p, check_range).unwrap();
        assert_eq!(yq, i64s(&case["yq"]));

        // Freivalds
        let uw = freivalds_precompute(&u, c, &w8, m, n, &p).unwrap();
        assert_eq!(uw, u64s(&case["uw"]));
        assert!(freivalds_check(&x_hat, &y_hat, rows, n, m, &u, &uw, c, &p).unwrap().iter().all(|&b| b));
        let tampered = u64s(&case["tampered_y_hat"]);
        let verdicts: Vec<bool> =
            case["tampered_verdicts"].as_array().unwrap().iter().map(|b| b.as_bool().unwrap()).collect();
        assert_eq!(freivalds_check(&x_hat, &tampered, rows, n, m, &u, &uw, c, &p).unwrap(), verdicts);
        assert_eq!(
            freivalds_verify(&x_hat, &tampered, rows, n, m, &u, &uw, c, &p, lid).unwrap_err().code(),
            ErrorCode::Integrity
        );
    }
}
