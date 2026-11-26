//! `torch.multinomial(probs, 1, generator=g)` for a 1-D float32 CPU tensor, exactly: the same index and the
//! same generator state afterwards (`Sampler.sample` / `Sampler.propose` in `pgw/engine/sampling.py`).
//!
//! torch draws one sample (`n_sample == 1`) with the exponential race: `q = empty_like(p).exponential_(1, g)`
//! (one `random64()` per element, in index order: `u = (r & (2^53 - 1)) * 2^-53`, `q = (float)(-log1p(-u))`),
//! then `argmax(p / q)` in float32 (the first maximal index; a NaN wins). The generator is torch's
//! `at::mt19937` (the standard MT19937; `random64()` = two outputs, the first in the high half), whose state
//! `Generator.get_state()` serializes (`CPUGeneratorImplState`, 5056 bytes).
//!
//! Cost: torch evaluates `log1p` for every element (152k at the Qwen vocabulary: ~4 ms on one core). Here the
//! generator runs for every element too (the state must advance identically), but `log1p` runs only for the
//! elements that can win: `-log1p(-u) >= u`, so `p / u` bounds an element's score from above, and an element
//! whose bound is below the exact score of the most probable element (with a 1e-5 relative margin, far above
//! the rounding of `log1p`, the float casts and the division) is skipped. The survivors are scored with the
//! same operations in the same precision as torch (glibc's `log1p` through Rust's `f64::ln_1p`, the cast to
//! float32, the float32 division) and compared in index order with torch's argmax rule. The Python side
//! checks the result against `torch.multinomial` once per process before using it (`pgw.engine.sampling`).

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(torch_multinomial1, m)?)?;
    Ok(())
}

thread_local! {
    /// The uniform draws of one call (reused: 1.2 MB at a 152k vocabulary).
    static SCRATCH: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

const N: usize = 624;
const M: usize = 397;
/// `sizeof(CPUGeneratorImplState)`: the legacy pod (seed u64, left i32, seeded i32, next u64, state u64[624],
/// normal_x / normal_y / normal_rho f64, normal_is_valid i32 + padding), the float normal sample, its flag.
pub const STATE_BYTES: usize = 5056;
const OFF_LEFT: usize = 8;
const OFF_NEXT: usize = 16;
const OFF_STATE: usize = 24;

/// MT19937 with numpy's position convention (`pos == N`: regenerate before the next output).
pub struct Mt {
    key: [u32; N],
    pos: usize,
}

impl Mt {
    /// From torch's serialized state (`left == 1`: the next output regenerates; else `next` is the position).
    pub fn from_torch(st: &[u8]) -> Result<Mt, String> {
        if st.len() != STATE_BYTES {
            return Err(format!("generator state has {} bytes, expected {STATE_BYTES}", st.len()));
        }
        let left = i32::from_le_bytes(st[OFF_LEFT..OFF_LEFT + 4].try_into().unwrap());
        let next = u64::from_le_bytes(st[OFF_NEXT..OFF_NEXT + 8].try_into().unwrap());
        if !(1..=N as i32 + 1).contains(&left) || next > N as u64 {
            return Err("generator state out of range".into());
        }
        let mut key = [0u32; N];
        for (i, k) in key.iter_mut().enumerate() {
            let o = OFF_STATE + 8 * i;
            let v = u64::from_le_bytes(st[o..o + 8].try_into().unwrap());
            if v > u32::MAX as u64 {
                return Err("generator state word out of range".into());
            }
            *k = v as u32;
        }
        let pos = if left == 1 { N } else { next as usize };
        if left != 1 && left as usize + pos != N + 1 {
            return Err("generator state inconsistent".into());
        }
        Ok(Mt { key, pos })
    }

    /// Writes the state back in torch's layout (`left = N + 1 - next`; every other field unchanged).
    pub fn to_torch(&self, st: &mut [u8]) {
        let left = (N + 1 - self.pos) as i32;
        st[OFF_LEFT..OFF_LEFT + 4].copy_from_slice(&left.to_le_bytes());
        st[OFF_NEXT..OFF_NEXT + 8].copy_from_slice(&(self.pos as u64).to_le_bytes());
        for (i, k) in self.key.iter().enumerate() {
            let o = OFF_STATE + 8 * i;
            st[o..o + 8].copy_from_slice(&(*k as u64).to_le_bytes());
        }
    }

    fn twist(&mut self) {
        let mt = &mut self.key;
        for i in 0..N {
            let y = (mt[i] & 0x8000_0000) | (mt[(i + 1) % N] & 0x7fff_ffff);
            let mut v = mt[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            mt[i] = v;
        }
        self.pos = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.twist();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `CPUGeneratorImpl::random64`: the first output is the high half.
    #[inline]
    pub fn random64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }
}

/// `uniform_real_distribution<double>(0, 1)` of torch: the low 53 bits times 2^-53.
#[inline]
fn uniform(r: u64) -> f64 {
    ((r & ((1u64 << 53) - 1)) as f64) * (1.0 / (1u64 << 53) as f64)
}

/// torch's float32 score of one element: `p / (float)(-log1p(-u))`.
#[inline]
fn score(p: f32, u: f64) -> f32 {
    let q = (-(-u).ln_1p()) as f32; // torch: -1 / lambda * log1p(-u), lambda = 1
    p / q
}

/// The index `torch.multinomial(probs, 1, generator)` returns, advancing `mt` exactly as torch does
/// (`probs.len()` draws). `Err` for inputs torch rejects (empty, a negative / non-finite value, sum 0): the
/// generator is then untouched by the caller (it does not write the state back).
pub fn multinomial1(mt: &mut Mt, p: &[f32], scratch: &mut Vec<f64>) -> Result<usize, String> {
    let n = p.len();
    if n == 0 {
        return Err("empty".into());
    }
    let mut best_p = -1f32;
    let mut jmax = 0usize;
    let mut any = false;
    for (i, &v) in p.iter().enumerate() {
        if v.is_nan() || v < 0.0 || v.is_infinite() {
            return Err("probability tensor contains either `inf`, `nan` or element < 0".into());
        }
        if v > best_p {
            best_p = v;
            jmax = i;
        }
        any |= v > 0.0;
    }
    if !any {
        return Err("invalid multinomial distribution (sum of probabilities <= 0)".into());
    }
    scratch.clear();
    scratch.reserve(n);
    for _ in 0..n {
        scratch.push(uniform(mt.random64()));
    }
    let u = &scratch[..];
    // lower bound on the winning score: the exact score of the most probable element
    let lb = score(p[jmax], u[jmax]) as f64;
    // `p / u < cut` written as `p < cut * u` (u > 0); the margin covers the product's rounding as well
    let cut = lb / (1.0 + 1e-5);
    let mut best = f32::NEG_INFINITY;
    let mut at = usize::MAX;
    let mut nan = false;
    for i in 0..n {
        let (pi, ui) = (p[i], u[i]);
        if ui > 0.0 {
            if pi == 0.0 || (pi as f64) < cut * ui {
                continue; // its exact score is below `lb`: it cannot be the (first) maximum
            }
        } else if pi == 0.0 {
            // q = 0 and p = 0: 0 / 0 is NaN, which torch's argmax takes as the maximum (the first one)
            if !nan {
                nan = true;
                at = i;
            }
            continue;
        }
        if nan {
            continue;
        }
        let s = score(pi, ui);
        if s.is_nan() {
            nan = true;
            at = i;
        } else if s > best {
            best = s;
            at = i;
        }
    }
    if at == usize::MAX {
        return Err("no candidate".into()); // unreachable: the most probable element is always a candidate
    }
    Ok(at)
}

/// ``torch.multinomial(probs, 1, generator=g)`` for a 1-D float32 array ``probs``, given ``state`` (a writable
/// uint8 array holding ``g.get_state()``): returns the index and writes the advanced state into ``state``
/// (the caller then calls ``g.set_state``), or ``None`` (state untouched) for an input torch would reject or a
/// state this function does not recognise; the caller then runs ``torch.multinomial`` itself.
#[pyfunction]
fn torch_multinomial1(py: Python<'_>, state: &Bound<'_, PyArray1<u8>>, probs: PyReadonlyArray1<'_, f32>)
    -> PyResult<Option<i64>> {
    if !state.is_c_contiguous() || !probs.is_c_contiguous() {
        return Err(PyValueError::new_err("state and probs must be C-contiguous"));
    }
    let mut stw = state.try_readwrite().map_err(|e| PyValueError::new_err(format!("state: {e}")))?;
    let st = stw.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let p = probs.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let got = py.allow_threads(|| -> Option<i64> {
        let mut mt = Mt::from_torch(st).ok()?;
        let i = SCRATCH.with(|c| {
            let mut s = c.borrow_mut();
            let r = multinomial1(&mut mt, p, &mut s);
            if s.capacity() > (1 << 20) {
                *s = Vec::new(); // above 8 MB of draws: not kept between calls
            }
            r
        });
        let i = i.ok()?;
        mt.to_torch(st);
        Some(i as i64)
    });
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MT19937 reference outputs for the default seed 5489 (the standard's 1st and 10000th outputs).
    #[test]
    fn mt_reference() {
        let mut key = [0u32; N];
        key[0] = 5489;
        for i in 1..N {
            key[i] = 1_812_433_253u32.wrapping_mul(key[i - 1] ^ (key[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let mut mt = Mt { key, pos: N };
        assert_eq!(mt.next_u32(), 3_499_211_612);
        for _ in 1..9999 {
            mt.next_u32();
        }
        assert_eq!(mt.next_u32(), 4_123_659_995);
    }

    fn exhaustive(mt: &mut Mt, p: &[f32]) -> usize {
        let mut best = f32::NEG_INFINITY;
        let mut at = 0;
        let mut nan = false;
        for (i, &pi) in p.iter().enumerate() {
            let s = score(pi, uniform(mt.random64()));
            if nan {
                continue;
            }
            if s.is_nan() {
                nan = true;
                at = i;
            } else if s > best {
                best = s;
                at = i;
            }
        }
        at
    }

    #[test]
    fn pruned_equals_exhaustive() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..200 {
            let n = 1 + (rnd() % 5000) as usize;
            let mut p: Vec<f32> = (0..n).map(|_| ((rnd() % 1_000_000) as f32 / 1e6).powi(6)).collect();
            if case % 3 == 0 {
                for v in p.iter_mut().step_by(2) {
                    *v = 0.0;
                }
                p[n - 1] = 0.5;
            }
            if case % 5 == 0 && n > 3 {
                p[1] = p[2]; // ties
            }
            let mut key = [0u32; N];
            for k in key.iter_mut() {
                *k = rnd() as u32;
            }
            let pos = (rnd() % (N as u64 + 1)) as usize;
            let mut a = Mt { key, pos };
            let mut b = Mt { key, pos };
            let mut s = Vec::new();
            let got = multinomial1(&mut a, &p, &mut s).unwrap();
            assert_eq!(got, exhaustive(&mut b, &p), "case {case}");
            assert_eq!(a.key, b.key);
            assert_eq!(a.pos, b.pos);
        }
    }

    #[test]
    fn rejects_like_torch() {
        let mut mt = Mt { key: [7u32; N], pos: N };
        let mut s = Vec::new();
        assert!(multinomial1(&mut mt, &[], &mut s).is_err());
        assert!(multinomial1(&mut mt, &[0.0, 0.0], &mut s).is_err());
        assert!(multinomial1(&mut mt, &[0.5, -0.1], &mut s).is_err());
        assert!(multinomial1(&mut mt, &[0.5, f32::NAN], &mut s).is_err());
        assert!(multinomial1(&mut mt, &[0.5, f32::INFINITY], &mut s).is_err());
        assert_eq!(mt.pos, N); // untouched by a rejected input
    }

    #[test]
    fn state_roundtrip() {
        let mut st = vec![0u8; STATE_BYTES];
        st[OFF_LEFT..OFF_LEFT + 4].copy_from_slice(&477i32.to_le_bytes());
        st[OFF_NEXT..OFF_NEXT + 8].copy_from_slice(&148u64.to_le_bytes());
        for i in 0..N {
            let o = OFF_STATE + 8 * i;
            st[o..o + 8].copy_from_slice(&((i as u64) * 2_654_435_761 % (1 << 32)).to_le_bytes());
        }
        let mt = Mt::from_torch(&st).unwrap();
        assert_eq!(mt.pos, 148);
        let mut back = vec![0u8; STATE_BYTES];
        mt.to_torch(&mut back);
        assert_eq!(&back[8..OFF_STATE + 8 * N], &st[8..OFF_STATE + 8 * N]);
    }
}
