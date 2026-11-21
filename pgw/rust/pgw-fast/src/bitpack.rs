//! Maskgen wire v2 bit packing (`pgw.hemask.wire.pack_bits` / `unpack_bits`) on the customer side,
//! fused with the steps around it, multi-threaded and with the GIL released.
//!
//! | function | Python reference |
//! |---|---|
//! | [`he_switch_pack`] | `pack_bits(switch_down(b, q, bits, rng), bits)` with the dither `d` passed in |
//! | [`he_unpack_shift`] | `unpack_bits(data, count, bits) << shift` |
//!
//! Layout (see `bit_fields`): for each whole-byte field `(offset, width)` (widest of 64/32/16/8 bits that
//! still fits, from bit 0 up), bits `offset .. offset+width-1` of every value as one little-endian array;
//! then the last `t = bits mod 8` bits of every value as an LSB-first stream, eight values per `t` bytes.
//! Bit-identical to the numpy code (`pgw/tests/hemask/test_bitpack_native.py`).

use chacha20::cipher::{KeyIvInit, StreamCipher};
use chacha20::ChaCha20;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};
use zeroize::Zeroizing;

/// Below this many values a kernel stays on the calling thread.
const PAR_MIN: usize = 1 << 17;
/// Values per worker thread at least.
const PAR_PER_THREAD: usize = 1 << 16;
/// At most this many threads (the copies are memory-bound).
const MAX_THREADS: usize = 8;

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// `(fields, tail)`: `(offset, width)` byte fields and the tail width (`pgw.hemask.wire.bit_fields`).
pub fn bit_fields(bits: u32) -> (Vec<(u32, u32)>, u32) {
    let mut fields = Vec::new();
    let mut off = 0;
    while bits - off >= 8 {
        let width = [64u32, 32, 16, 8].into_iter().find(|&c| c <= bits - off).unwrap_or(8);
        fields.push((off, width));
        off += width;
    }
    (fields, bits - off)
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
// SAFETY: every thread writes a disjoint byte range of the buffer behind the pointer, which outlives the
// scoped threads.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(self) -> *mut u8 {
        self.0
    }
}

fn threads_for(count: usize) -> usize {
    if count < PAR_MIN {
        return 1;
    }
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    cpus.min(MAX_THREADS).min(count / PAR_PER_THREAD).max(1)
}

/// Split `count` values (a multiple of 8) into `[v0, v1)` runs of whole groups of 8 and run `f` on each.
fn par_groups(count: usize, max_threads: usize, f: impl Fn(usize, usize) + Sync) {
    let groups = count / 8;
    let auto = threads_for(count);
    let threads = if max_threads == 0 { auto } else { auto.min(max_threads) }.min(groups.max(1));
    if threads <= 1 {
        f(0, count);
        return;
    }
    let per = groups.div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (g0, g1) = (t * per, ((t + 1) * per).min(groups));
            if g0 >= g1 {
                break;
            }
            let f = &f;
            s.spawn(move || f(8 * g0, 8 * g1));
        }
    });
}

/// One whole-byte field of the layout: where its array starts, its byte width and bit offset.
#[derive(Clone, Copy)]
struct Field {
    pos: usize,
    wb: usize,
    off: u32,
}

/// The fields of `bits` for `count` values, and the byte position of the tail stream.
fn layout(count: usize, bits: u32) -> ([Field; 4], usize, usize, u32) {
    let (fields, t) = bit_fields(bits);
    let mut out = [Field { pos: 0, wb: 0, off: 0 }; 4];
    let mut pos = 0usize;
    for (k, &(off, width)) in fields.iter().enumerate() {
        let wb = (width / 8) as usize;
        out[k] = Field { pos, wb, off };
        pos += count * wb;
    }
    (out, fields.len(), pos, t)
}

#[inline(always)]
unsafe fn store(p: *mut u8, wb: usize, x: u64) {
    match wb {
        8 => (p as *mut u64).write_unaligned(x.to_le()),
        4 => (p as *mut u32).write_unaligned((x as u32).to_le()),
        2 => (p as *mut u16).write_unaligned((x as u16).to_le()),
        _ => *p = x as u8,
    }
}

#[inline(always)]
unsafe fn load(p: *const u8, wb: usize) -> u64 {
    match wb {
        8 => u64::from_le((p as *const u64).read_unaligned()),
        4 => u32::from_le((p as *const u32).read_unaligned()) as u64,
        2 => u16::from_le((p as *const u16).read_unaligned()) as u64,
        _ => *p as u64,
    }
}

/// Pack values `[v0, v1)` given by `value(i)` into `out` (the whole packed buffer of `count` values).
///
/// # Safety
/// `out` points to `count * bits / 8` writable bytes; concurrent callers use disjoint `[v0, v1)`
/// (multiples of 8).
unsafe fn pack_range(out: SendPtr, count: usize, bits: u32, v0: usize, v1: usize, value: impl Fn(usize) -> u64) {
    let (fields, nf, tail_pos, t) = layout(count, bits);
    let base = out.get();
    let mask = if t > 0 { (1u64 << t) - 1 } else { 0 };
    let top = bits - t;
    for g in v0 / 8..v1 / 8 {
        let mut word = 0u64;
        for k in 0..8 {
            let i = 8 * g + k;
            let x = value(i);
            for f in &fields[..nf] {
                store(base.add(f.pos + i * f.wb), f.wb, x >> f.off);
            }
            word |= (x.wrapping_shr(top) & mask) << (t as usize * k);
        }
        if t > 0 {
            let w = word.to_le_bytes();
            std::ptr::copy_nonoverlapping(w.as_ptr(), base.add(tail_pos + g * t as usize), t as usize);
        }
    }
}

/// Unpack values `[v0, v0 + dst.len())` of `src` (all `count` values) into `dst`, each shifted left by
/// `shift`. `v0` and `dst.len()` are multiples of 8; `src` has `count * bits / 8` bytes (checked by the caller).
fn unpack_range(src: &[u8], count: usize, bits: u32, shift: u32, v0: usize, dst: &mut [u64]) {
    let (fields, nf, tail_pos, t) = layout(count, bits);
    assert!(src.len() >= count * bits as usize / 8 && v0 + dst.len() <= count && v0 % 8 == 0 && dst.len() % 8 == 0);
    let base = src.as_ptr();
    let mask = if t > 0 { (1u64 << t) - 1 } else { 0 };
    let top = bits - t;
    let tb = t as usize;
    for (g, chunk) in dst.chunks_exact_mut(8).enumerate() {
        let gi = v0 / 8 + g;
        let word = if t > 0 {
            let mut x = [0u8; 8];
            x[..tb].copy_from_slice(&src[tail_pos + gi * tb..tail_pos + (gi + 1) * tb]);
            u64::from_le_bytes(x)
        } else {
            0
        };
        for (k, d) in chunk.iter_mut().enumerate() {
            let i = 8 * gi + k;
            let mut v = ((word >> (t as usize * k)) & mask).wrapping_shl(top);
            for f in &fields[..nf] {
                // SAFETY: i < count, so the field read is inside src (length asserted above).
                v |= unsafe { load(base.add(f.pos + i * f.wb), f.wb) } << f.off;
            }
            *d = v << shift;
        }
    }
}

/// `pack_bits(switch_down(b, q_bits, bits, ·), bits)` with the dither given: ``len(b)·bits/8`` bytes.
///
/// ``b``: uint64 coefficients below ``2^q_bits`` (1-D, C-contiguous, a multiple of 8 values). ``dither``:
/// one uint8 per coefficient, uniform in ``[0, 2^(q_bits-bits))``, or a fresh 32-byte seed (``bytes``)
/// from which the dither is drawn here with ChaCha20 (ignored, may be ``None``, when ``bits == q_bits``).
/// Each value is ``floor((b + d) / 2^s)`` with ``s = q_bits - bits``, computed without overflow for
/// ``q_bits = 64``; only its low ``bits`` bits are packed (that is the ``mod 2^bits``).
#[pyfunction]
#[pyo3(signature = (b, dither, q_bits, bits, threads=0))]
fn he_switch_pack<'py>(
    py: Python<'py>,
    b: PyReadonlyArray1<'py, u64>,
    dither: &Bound<'py, PyAny>,
    q_bits: u32,
    bits: u32,
    threads: usize,
) -> PyResult<Bound<'py, PyByteArray>> {
    if !(1..=64).contains(&bits) || !(bits..=64).contains(&q_bits) {
        return Err(verr(format!("bits={bits}, q_bits={q_bits} out of range")));
    }
    let s = q_bits - bits;
    if s >= 8 {
        return Err(verr(format!("switch of {s} bits not supported")));
    }
    let bs = b.as_slice().map_err(|_| verr("b must be C-contiguous"))?;
    let count = bs.len();
    if count % 8 != 0 {
        return Err(verr(format!("bit packing needs a multiple of 8 values, got {count}")));
    }
    // The dither: none (no switch), one byte per value (uint8 array), or a 32-byte seed from which
    // ChaCha20 (zero nonce) draws it here (a uniform byte masked to s bits is uniform in [0, 2^s)).
    let arr: Option<PyReadonlyArray1<'py, u8>> = if s > 0 && !dither.is_instance_of::<PyBytes>() {
        Some(dither.extract().map_err(|_| verr("dither must be a uint8 array or a 32-byte seed"))?)
    } else {
        None
    };
    let mut drawn: Zeroizing<Vec<u8>> = Zeroizing::new(Vec::new());
    let ds: Option<&[u8]> = if s == 0 {
        None
    } else if let Some(a) = &arr {
        let d = a.as_slice().map_err(|_| verr("dither must be C-contiguous"))?;
        if d.len() != count {
            return Err(verr("dither must have one byte per value"));
        }
        Some(d)
    } else {
        let seed: [u8; 32] = dither
            .downcast::<PyBytes>()?
            .as_bytes()
            .try_into()
            .map_err(|_| verr("a dither seed must be exactly 32 bytes"))?;
        let seed = Zeroizing::new(seed);
        drawn.resize(count, 0);
        let mut c = ChaCha20::new((&*seed).into(), &[0u8; 12].into());
        c.apply_keystream(&mut drawn);
        let lo = ((1u32 << s) - 1) as u8;
        for x in drawn.iter_mut() {
            *x &= lo;
        }
        Some(&drawn[..])
    };
    let nbytes = count * bits as usize / 8;
    let out = PyByteArray::new_with(py, nbytes, |buf| {
        let ptr = SendPtr(buf.as_mut_ptr());
        py.allow_threads(|| {
            let lo = if s == 0 { 0 } else { (1u64 << s) - 1 };
            par_groups(count, threads, |v0, v1| {
                let value = |i: usize| -> u64 {
                    let x = bs[i];
                    match ds {
                        None => x,
                        Some(d) if q_bits < 64 => x.wrapping_add(d[i] as u64) >> s,
                        Some(d) => (x >> s) + (((x & lo) + d[i] as u64) >> s),
                    }
                };
                // SAFETY: `buf` has `nbytes` bytes and the runs are disjoint (par_groups).
                unsafe { pack_range(ptr, count, bits, v0, v1, value) };
            });
        });
        Ok(())
    })?;
    Ok(out)
}

/// ``unpack_bits(data, count, bits) << shift`` (uint64, wrapping): a fresh 1-D array.
#[pyfunction]
#[pyo3(signature = (data, count, bits, shift=0, threads=0))]
fn he_unpack_shift<'py>(
    py: Python<'py>,
    data: &[u8],
    count: usize,
    bits: u32,
    shift: u32,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if !(1..=64).contains(&bits) || shift >= 64 {
        return Err(verr(format!("bits={bits}, shift={shift} out of range")));
    }
    if count % 8 != 0 {
        return Err(verr(format!("bit packing needs a multiple of 8 values, got {count}")));
    }
    if data.len() != count * bits as usize / 8 {
        return Err(verr(format!("packed data has {} bytes, expected {}", data.len(), count * bits as usize / 8)));
    }
    let arr = PyArray1::<u64>::zeros(py, [count], false);
    // SAFETY: freshly allocated, C-contiguous, not yet shared with Python code.
    let dst = unsafe { arr.as_slice_mut() }.map_err(|e| verr(e.to_string()))?;
    let ptr = SendPtr(dst.as_mut_ptr() as *mut u8);
    py.allow_threads(|| {
        par_groups(count, threads, |v0, v1| {
            // SAFETY: disjoint runs of the freshly allocated output (par_groups).
            let part = unsafe { std::slice::from_raw_parts_mut((ptr.get() as *mut u64).add(v0), v1 - v0) };
            unpack_range(data, count, bits, shift, v0, part);
        });
    });
    debug_assert_eq!(arr.len(), count);
    Ok(arr)
}

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(he_switch_pack, m)?)?;
    m.add_function(wrap_pyfunction!(he_unpack_shift, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_match_python() {
        assert_eq!(bit_fields(35), (vec![(0, 32)], 3));
        assert_eq!(bit_fields(53), (vec![(0, 32), (32, 16)], 5));
        assert_eq!(bit_fields(56), (vec![(0, 32), (32, 16), (48, 8)], 0));
        assert_eq!(bit_fields(64), (vec![(0, 64)], 0));
        assert_eq!(bit_fields(5), (vec![], 5));
    }

    #[test]
    fn round_trip() {
        for bits in [5u32, 8, 13, 35, 43, 53, 56, 63, 64] {
            let count = 8 * 37;
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            let vals: Vec<u64> = (0..count as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask).collect();
            let mut buf = vec![0u8; count * bits as usize / 8];
            let ptr = SendPtr(buf.as_mut_ptr());
            unsafe { pack_range(ptr, count, bits, 0, count, |i| vals[i]) };
            let mut out = vec![0u64; count];
            unpack_range(&buf, count, bits, 0, 0, &mut out);
            assert_eq!(out, vals, "bits {bits}");
            let mut part = vec![0u64; 8 * 17];
            unpack_range(&buf, count, bits, 0, 8, &mut part);
            assert_eq!(&part[..], &vals[8..8 + 8 * 17]);
        }
    }
}
