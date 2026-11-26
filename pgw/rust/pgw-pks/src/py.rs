//! Python module `pgw_pks` (wrapped by `pgw.hemask.pks_native`). Every function releases the GIL while
//! it computes; results equal the numpy reference `pgw.hemask.pks` bit for bit (`tests/hemask/test_pks.py`).

use crate::arith::{unpack_bits as unpack, Isa};
use crate::kernel::{self, DecItem, C0, EncItem, KsItem, Wiping, D, NR, NU, PACK, RANK};
use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn kerr(e: kernel::Error) -> PyErr {
    verr(e.0)
}

fn isa_of(name: &str) -> PyResult<Isa> {
    Isa::parse(name).ok_or_else(|| verr(format!("unknown isa {name:?} (auto, avx512ifma, avx512, avx2, scalar)")))
}

/// The packed reply of :func:`decrypt` / :func:`decrypt_into`: ``bytes`` or any C-contiguous byte buffer (the factory
/// passes a read-only ``memoryview`` into the HTTP response body: no copy of the 4.5 bytes per output and slot).
struct ReplyBytes {
    buf: pyo3::buffer::PyBuffer<u8>,
}

impl ReplyBytes {
    fn get(y: &Bound<'_, PyAny>) -> PyResult<Self> {
        let buf = pyo3::buffer::PyBuffer::<u8>::get(y)?;
        if !buf.is_c_contiguous() {
            return Err(verr("y must be a C-contiguous byte buffer"));
        }
        Ok(ReplyBytes { buf })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: a C-contiguous buffer of `len_bytes` bytes that `self.buf` keeps exported (so its exporter cannot
        // resize or free it) for as long as the returned slice lives; only read.
        unsafe { std::slice::from_raw_parts(self.buf.buf_ptr() as *const u8, self.buf.len_bytes()) }
    }
}

fn key32(b: &[u8]) -> PyResult<[u8; 32]> {
    b.try_into().map_err(|_| verr("stream keys are 32 bytes"))
}

/// Constants: primes (``q1``, ``ps``, ``ep``), degrees, the ISA ``auto`` resolves to, version.
#[pyfunction]
fn info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let cx = kernel::ctx();
    let d = PyDict::new(py);
    d.set_item("q1_primes", vec![cx.qp[0].p, cx.qp[1].p])?;
    d.set_item("ps_primes", vec![cx.qp[2].p, cx.qp[3].p])?;
    d.set_item("ep_primes", cx.ep.iter().map(|p| p.p).collect::<Vec<u32>>())?;
    d.set_item("nr", NR)?;
    d.set_item("nu", NU)?;
    d.set_item("isa", Isa::parse("auto").unwrap_or(Isa::Scalar).name())?;
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    Ok(d)
}

/// ``n`` uniform ternary values (int8) from the stream ``key``, or ``None`` (fall back to the reference).
#[pyfunction]
fn ternary<'py>(py: Python<'py>, key: &[u8], n: usize) -> PyResult<Option<Bound<'py, PyArray1<i8>>>> {
    let k = key32(key)?;
    let got = py.allow_threads(|| kernel::ternary(&k, n));
    Ok(got.map(|v| PyArray1::from_slice(py, &v.0)))
}

/// The public key-switching ``a``-parts in Montgomery form (opaque, reused for every reply key).
#[pyclass(frozen)]
struct KHat {
    km: Vec<u32>,
}

#[pymethods]
impl KHat {
    #[new]
    fn new(py: Python<'_>, k_hat: PyReadonlyArray3<'_, u32>) -> PyResult<Self> {
        let s = k_hat.as_slice().map_err(|_| verr("k_hat must be C-contiguous (64, 8192) or (4*16, 8192)"))?;
        let km = py.allow_threads(|| kernel::k_mont(s)).map_err(kerr)?;
        Ok(KHat { km })
    }
}

/// Upload of one item: ``b`` (``n x 256 G``, uint64 mod 2^56) for masks ``r`` (``rows x n`` int64), upload
/// secrets ``t`` (``G x 4096`` int8), noise stream keys ``e_keys`` (one per group).
#[pyfunction]
#[pyo3(signature = (a_key, r, t, e_keys, threads = 1, isa = "auto"))]
fn encrypt<'py>(
    py: Python<'py>,
    a_key: &[u8],
    r: PyReadonlyArray2<'py, i64>,
    t: PyReadonlyArray2<'py, i8>,
    e_keys: Vec<Vec<u8>>,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let isa = isa_of(isa)?;
    let (rows, n) = (r.shape()[0], r.shape()[1]);
    let groups = e_keys.len();
    let ek: Vec<[u8; 32]> = e_keys.iter().map(|k| key32(k)).collect::<PyResult<_>>()?;
    let it = EncItem {
        a_key: key32(a_key)?,
        n,
        r: r.as_slice().map_err(|_| verr("r must be C-contiguous"))?,
        rows,
        t: t.as_slice().map_err(|_| verr("t must be C-contiguous"))?,
        e_keys: &ek,
    };
    let mut b = vec![0u64; n * groups * D];
    py.allow_threads(|| kernel::encrypt_item(isa, &it, &mut b, threads)).map_err(kerr)?;
    Ok(Array2::from_shape_vec((n, groups * D), b).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// :func:`encrypt` straight into a maskgen v3 request body: returns ``head + pack_bits(b, 56) + mid +
/// pack_ksk(beta, ksk_bits)`` as one ``bytes`` object (:func:`pgw.hemask.wire.maskgen_v3_frame` gives ``head`` and
/// ``mid``), the packed bodies written in place by the kernel and the reply keys' wire form packed in place: no
/// ``b`` array, no packing pass over it, no copy of the payloads through Python. Same ``b`` as :func:`encrypt` and
/// the same key bytes as :func:`pack_ksk` for the same inputs.
#[pyfunction]
#[pyo3(signature = (a_key, r, t, e_keys, head, mid, beta, ksk_bits, threads = 1, isa = "auto"))]
#[allow(clippy::too_many_arguments)]
fn encrypt_wire<'py>(
    py: Python<'py>,
    a_key: &[u8],
    r: PyReadonlyArray2<'py, i64>,
    t: PyReadonlyArray2<'py, i8>,
    e_keys: Vec<Vec<u8>>,
    head: &[u8],
    mid: &[u8],
    beta: PyReadonlyArray4<'py, u32>,
    ksk_bits: Vec<u32>,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let isa = isa_of(isa)?;
    let (rows, n) = (r.shape()[0], r.shape()[1]);
    let groups = e_keys.len();
    let ek: Vec<[u8; 32]> = e_keys.iter().map(|k| key32(k)).collect::<PyResult<_>>()?;
    let it = EncItem {
        a_key: key32(a_key)?,
        n,
        r: r.as_slice().map_err(|_| verr("r must be C-contiguous"))?,
        rows,
        t: t.as_slice().map_err(|_| verr("t must be C-contiguous"))?,
        e_keys: &ek,
    };
    let sh = beta.shape();
    if sh[0] != groups || sh[1] != ksk_bits.len() || sh[2] != RANK || sh[3] != NR
        || ksk_bits.iter().any(|&b| !(8..=32).contains(&b))
    {
        return Err(verr("beta must be (groups, len(ksk_bits), 16, 8192) with 8 <= bits <= 32"));
    }
    let bsl = beta.as_slice().map_err(|_| verr("beta must be C-contiguous"))?;
    let ksk_len: usize = groups * ksk_bits.iter().map(|&b| RANK * NR * b as usize / 8).sum::<usize>();
    let payload = n.checked_mul(groups * D).and_then(|c| c.checked_mul(7)).ok_or_else(|| verr("size overflow"))?;
    let total = head.len() + payload + mid.len() + ksk_len;
    // SAFETY: a fresh, uninitialised bytes object of `total` bytes, filled completely below before any Python code
    // can see it (the GIL is released while the kernels write their ranges; the object is not yet shared).
    let obj = unsafe {
        let p = pyo3::ffi::PyBytes_FromStringAndSize(std::ptr::null(), total as pyo3::ffi::Py_ssize_t);
        Bound::from_owned_ptr_or_err(py, p)?.downcast_into_unchecked::<PyBytes>()
    };
    struct Buf(*mut u8);
    unsafe impl Send for Buf {}
    let buf = Buf(unsafe { pyo3::ffi::PyBytes_AsString(obj.as_ptr()) as *mut u8 });
    let res = py.allow_threads(|| -> Result<(), kernel::Error> {
        let buf = buf;
        // SAFETY: the whole fresh object, exclusively ours until it is returned.
        let all = unsafe { std::slice::from_raw_parts_mut(buf.0, total) };
        let (h, rest) = all.split_at_mut(head.len());
        let (body, rest) = rest.split_at_mut(payload);
        let (m, kk) = rest.split_at_mut(mid.len());
        h.copy_from_slice(head);
        m.copy_from_slice(mid);
        let mut pos = 0usize;
        for (j, chunk) in bsl.chunks_exact(RANK * NR).enumerate() {
            pos += crate::arith::pack_bits_to(chunk, ksk_bits[j % ksk_bits.len()], &mut kk[pos..]).map_err(kernel::Error)?;
        }
        debug_assert_eq!(pos, kk.len());
        kernel::encrypt_item_to(isa, &it, kernel::EncOut::Packed56(body), threads)
    });
    res.map_err(kerr)?;
    Ok(obj)
}

/// Reply key of one group: ``(beta (4, 16, 8192) uint32, trhat (2, 8192) uint32)``.
#[pyfunction]
#[pyo3(signature = (khat, t, tr, e_key, isa = "auto"))]
#[allow(clippy::type_complexity)]
fn reply_key<'py>(
    py: Python<'py>,
    khat: &KHat,
    t: &Bound<'py, PyArray1<i8>>,
    tr: &Bound<'py, PyArray1<i8>>,
    e_key: &[u8],
    isa: &str,
) -> PyResult<(Bound<'py, PyArray3<u32>>, Bound<'py, PyArray2<u32>>)> {
    let isa = isa_of(isa)?;
    let ek = key32(e_key)?;
    let tv = Wiping(t.readonly().as_slice().map_err(|_| verr("t must be contiguous"))?.to_vec());
    let trv = Wiping(tr.readonly().as_slice().map_err(|_| verr("tr must be contiguous"))?.to_vec());
    let mut beta = vec![0u32; 4 * RANK * NR];
    let mut trhat = vec![0u32; 2 * NR];
    py.allow_threads(|| kernel::reply_key(isa, &khat.km, &tv.0, &trv.0, &ek, &mut beta, &mut trhat)).map_err(kerr)?;
    Ok((
        Array3::from_shape_vec((4, RANK, NR), beta).map_err(|e| verr(e.to_string()))?.into_pyarray(py),
        Array2::from_shape_vec((2, NR), trhat).map_err(|e| verr(e.to_string()))?.into_pyarray(py),
    ))
}

/// Decrypt one item: ``hint`` (``(nb, 2, 8192, 3)`` uint8), the packed reply ``y`` (``(G, nb, 8192)`` values
/// at ``yb`` bits, :func:`pgw.hemask.wire.pack_bits` layout), ``trhat`` (``(G, 2, 8192)``), ``rows`` per group.
/// Returns ``(sum(rows), m)`` uint32 masks, groups stacked.
#[pyfunction]
#[pyo3(signature = (hint, y, yb, trhat, rows, threads = 1, isa = "auto"))]
#[allow(clippy::too_many_arguments)]
fn decrypt<'py>(
    py: Python<'py>,
    hint: PyReadonlyArray4<'py, u8>,
    y: &Bound<'py, PyAny>,
    yb: u32,
    trhat: PyReadonlyArray3<'py, u32>,
    rows: Vec<usize>,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let isa = isa_of(isa)?;
    let nb = hint.shape()[0];
    let m = nb * PACK;
    let groups = rows.len();
    let yb_buf = ReplyBytes::get(y)?;
    let data = yb_buf.as_slice();
    let hs = hint.as_slice().map_err(|_| verr("hint must be C-contiguous"))?;
    let th = trhat.as_slice().map_err(|_| verr("trhat must be C-contiguous"))?;
    let total: usize = rows.iter().sum();
    let mut out = vec![0u32; total * m];
    py.allow_threads(|| -> Result<(), kernel::Error> {
        let count = groups * nb * NR;
        let words;
        let c0r = if yb == 36 {
            C0::Packed36 { data, count }
        } else {
            words = unpack(data, count, yb).map_err(kernel::Error)?;
            C0::Words(&words)
        };
        let mut outs: Vec<&mut [u32]> = Vec::with_capacity(groups);
        let mut rest = out.as_mut_slice();
        for &r in &rows {
            let (a, b) = rest.split_at_mut(r * m);
            outs.push(a);
            rest = b;
        }
        let it = DecItem { hint: hs, m, c0r, yb, trhat: th, rows: &rows };
        kernel::decrypt_item(isa, &it, &mut outs, threads)
    })
    .map_err(kerr)?;
    Ok(Array2::from_shape_vec((total, m), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// :func:`decrypt` written into ``out`` (a C-contiguous, writable ``(sum(rows), m)`` uint32 array; every element is
/// overwritten): the same values without a fresh allocation (the mask factory reuses its zeroed output buffers).
#[pyfunction]
#[pyo3(signature = (hint, y, yb, trhat, rows, out, threads = 1, isa = "auto"))]
#[allow(clippy::too_many_arguments)]
fn decrypt_into<'py>(
    py: Python<'py>,
    hint: PyReadonlyArray4<'py, u8>,
    y: &Bound<'py, PyAny>,
    yb: u32,
    trhat: PyReadonlyArray3<'py, u32>,
    rows: Vec<usize>,
    out: &Bound<'py, PyArray2<u32>>,
    threads: usize,
    isa: &str,
) -> PyResult<()> {
    let isa = isa_of(isa)?;
    let nb = hint.shape()[0];
    let m = nb * PACK;
    let groups = rows.len();
    let yb_buf = ReplyBytes::get(y)?;
    let data = yb_buf.as_slice();
    let hs = hint.as_slice().map_err(|_| verr("hint must be C-contiguous"))?;
    let th = trhat.as_slice().map_err(|_| verr("trhat must be C-contiguous"))?;
    let total: usize = rows.iter().sum();
    if out.shape() != [total, m] {
        return Err(verr(format!("out must be ({total}, {m})")));
    }
    let mut rw = out.try_readwrite().map_err(|e| verr(format!("out must be writable: {e}")))?;
    let outs_all = rw.as_slice_mut().map_err(|_| verr("out must be C-contiguous"))?;
    py.allow_threads(|| -> Result<(), kernel::Error> {
        let count = groups * nb * NR;
        let words;
        let c0r = if yb == 36 {
            C0::Packed36 { data, count }
        } else {
            words = unpack(data, count, yb).map_err(kernel::Error)?;
            C0::Words(&words)
        };
        let mut outs: Vec<&mut [u32]> = Vec::with_capacity(groups);
        let mut rest = outs_all;
        for &r in &rows {
            let (a, b) = rest.split_at_mut(r * m);
            outs.push(a);
            rest = b;
        }
        let it = DecItem { hint: hs, m, c0r, yb, trhat: th, rows: &rows };
        kernel::decrypt_item(isa, &it, &mut outs, threads)
    })
    .map_err(kerr)
}

/// Provider reply (``y`` given: ``(G, nb, 8192)`` uint64 below ``2^yb``) or compact-hint build (``y=None``,
/// one operand: ``(nb, 2, 8192)`` hint-form residues as uint64). ``hs``: ``(m, 4096)`` uint64 below ``q1``;
/// ``ops``: ``(G, 4, 16, 8192)`` uint32.
#[pyfunction]
#[pyo3(signature = (hs, ops, y = None, yb = 36, threads = 1, isa = "auto"))]
fn keyswitch<'py>(
    py: Python<'py>,
    hs: PyReadonlyArray2<'py, u64>,
    ops: PyReadonlyArray4<'py, u32>,
    y: Option<PyReadonlyArray2<'py, u64>>,
    yb: u32,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray3<u64>>> {
    let isa = isa_of(isa)?;
    let m = hs.shape()[0];
    if hs.shape()[1] != NU || m % PACK != 0 {
        return Err(verr(format!("hs must be (m, {NU}) with m a multiple of {PACK}")));
    }
    let groups = ops.shape()[0];
    let nb = m / PACK;
    let hsl = hs.as_slice().map_err(|_| verr("hs must be C-contiguous"))?;
    let osl = ops.as_slice().map_err(|_| verr("ops must be C-contiguous"))?;
    let ysl = match &y {
        Some(a) => Some(a.as_slice().map_err(|_| verr("y must be C-contiguous"))?),
        None => None,
    };
    let (shape, len) = if ysl.is_some() { ((groups, nb, NR), groups * nb * NR) } else { ((nb, 2, NR), nb * 2 * NR) };
    let mut out = vec![0u64; len];
    let it = KsItem { hs: hsl, m, ops: osl, groups, y: ysl, yb, raw: false };
    py.allow_threads(|| kernel::keyswitch(isa, &it, &mut out, threads)).map_err(kerr)?;
    Ok(Array3::from_shape_vec(shape, out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// :func:`pgw.hemask.pks.switch_hint`: ``round((h mod 2^56) q1 / 2^56) mod q1`` element-wise (``(m, k)`` uint64).
#[pyfunction]
#[pyo3(signature = (h, threads = 1, isa = "auto"))]
fn switch_hint<'py>(py: Python<'py>, h: PyReadonlyArray2<'py, u64>, threads: usize, isa: &str)
    -> PyResult<Bound<'py, PyArray2<u64>>> {
    let isa = isa_of(isa)?;
    let (m, k) = (h.shape()[0], h.shape()[1]);
    let hsl = h.as_slice().map_err(|_| verr("h must be C-contiguous"))?;
    let mut out = vec![0u64; m * k];
    py.allow_threads(|| kernel::switch_hint(isa, hsl, &mut out, threads)).map_err(kerr)?;
    Ok(Array2::from_shape_vec((m, k), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// The compact hint of ``m`` rows in its stored form ``(m/32, 2, 8192, 3)`` uint8 (24-bit little-endian
/// residues): the key switch of ``switch_hint(h)`` with the public ``k`` (``ops`` ``(1, 4, 16, 8192)``).
/// ``raw=True``: ``h`` is ``H = W8 A mod 2^56`` and the switch is fused (no switched copy); ``raw=False``:
/// ``h`` is already switched (below ``q1``). Same bytes as ``keyswitch`` followed by the 3-byte packing.
#[pyfunction]
#[pyo3(signature = (h, ops, raw = true, threads = 1, isa = "auto"))]
fn hint_keyswitch<'py>(py: Python<'py>, h: PyReadonlyArray2<'py, u64>, ops: PyReadonlyArray4<'py, u32>, raw: bool,
                       threads: usize, isa: &str) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let isa = isa_of(isa)?;
    let m = h.shape()[0];
    if h.shape()[1] != NU || m % PACK != 0 || ops.shape()[0] != 1 {
        return Err(verr(format!("h must be (m, {NU}) with m a multiple of {PACK}, ops one operand")));
    }
    let nb = m / PACK;
    let hsl = h.as_slice().map_err(|_| verr("h must be C-contiguous"))?;
    let osl = ops.as_slice().map_err(|_| verr("ops must be C-contiguous"))?;
    let it = KsItem { hs: hsl, m, ops: osl, groups: 1, y: None, yb: 36, raw };
    let out = py.allow_threads(|| -> Result<Vec<u8>, kernel::Error> {
        let mut v = vec![0u64; nb * 2 * NR];
        kernel::keyswitch(isa, &it, &mut v, threads)?;
        let mut b = vec![0u8; nb * 2 * NR * 3];
        for (o, x) in b.chunks_exact_mut(3).zip(&v) {
            o.copy_from_slice(&(*x as u32).to_le_bytes()[..3]);
        }
        Ok(b)
    }).map_err(kerr)?;
    Ok(Array4::from_shape_vec((nb, 2, NR, 3), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// ``(sum_d (part[:, d] + off) << 8 d) mod 2^56``: the recombination of the byte-plane GEMM ``W8 A``
/// (``part`` ``(rows, planes, cols)`` int32, ``off`` ``(rows,)`` int32); ``(rows, cols)`` uint64.
#[pyfunction]
#[pyo3(signature = (part, off, threads = 1, isa = "auto"))]
fn combine_planes<'py>(py: Python<'py>, part: PyReadonlyArray3<'py, i32>, off: PyReadonlyArray1<'py, i32>,
                       threads: usize, isa: &str) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let isa = isa_of(isa)?;
    let (rows, planes, cols) = (part.shape()[0], part.shape()[1], part.shape()[2]);
    if off.shape()[0] != rows {
        return Err(verr("off must have one entry per row"));
    }
    let psl = part.as_slice().map_err(|_| verr("part must be C-contiguous"))?;
    let fsl = off.as_slice().map_err(|_| verr("off must be C-contiguous"))?;
    let mut out = vec![0u64; rows * cols];
    py.allow_threads(|| kernel::combine_planes(isa, psl, fsl, planes, cols, &mut out, threads)).map_err(kerr)?;
    Ok(Array2::from_shape_vec((rows, cols), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// Byte planes of ``A = a_mlwe_rows(ring)`` for the hint GEMM: ``(7 * 4096, n)`` int8, row
/// ``(col // cb) * 7 cb + d * cb + col % cb`` holding ``byte_d(A[:, col]) - 128`` (``ring``: ``(n/16, 4096)`` uint64
/// below ``2^56``; ``cb`` divides 4096).
#[pyfunction]
#[pyo3(signature = (ring, cb = 4096, threads = 1, isa = "auto"))]
fn a_planes<'py>(py: Python<'py>, ring: PyReadonlyArray2<'py, u64>, cb: usize, threads: usize, isa: &str)
    -> PyResult<Bound<'py, PyArray2<i8>>> {
    let isa = isa_of(isa)?;
    if ring.shape()[1] != NU {
        return Err(verr(format!("ring must be (n/16, {NU})")));
    }
    let n = ring.shape()[0] * RANK;
    let rsl = ring.as_slice().map_err(|_| verr("ring must be C-contiguous"))?;
    let mut out = vec![0i8; 7 * NU * n];
    // SAFETY: i8 and u8 have the same size and alignment; the view lives only during the call.
    let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len()) };
    py.allow_threads(|| kernel::a_planes(isa, rsl, n, cb, ob, threads)).map_err(kerr)?;
    Ok(Array2::from_shape_vec((7 * NU, n), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// :func:`pgw.service.pks_provider.pack_ksk`: the reply keys' wire form, ``beta`` ``(G, 4, 16, 8192)`` uint32 packed per
/// (group, prime) with ``bits[i]`` bits (:func:`pgw.hemask.wire.pack_bits` layout).
#[pyfunction]
fn pack_ksk<'py>(py: Python<'py>, beta: PyReadonlyArray4<'py, u32>, bits: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
    let sh = beta.shape();
    if sh[1] != bits.len() || sh[2] != RANK || sh[3] != NR {
        return Err(verr("beta must be (G, len(bits), 16, 8192)"));
    }
    let bsl = beta.as_slice().map_err(|_| verr("beta must be C-contiguous"))?;
    let per = RANK * NR;
    let out = py.allow_threads(|| -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(bsl.len() * 4);
        for (j, chunk) in bsl.chunks_exact(per).enumerate() {
            crate::arith::pack_bits_into(chunk, bits[j % bits.len()], &mut out)?;
        }
        Ok(out)
    }).map_err(verr)?;
    Ok(PyBytes::new(py, &out))
}

/// :func:`pgw.hemask.wire.unpack_bits` (uint64 values).
#[pyfunction]
fn unpack_bits<'py>(py: Python<'py>, data: &Bound<'py, PyBytes>, count: usize, bits: u32) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let d = data.as_bytes();
    let v = py.allow_threads(|| unpack(d, count, bits)).map_err(verr)?;
    Ok(PyArray1::from_vec(py, v))
}

/// ``vecmat_wrap32(x, w, threads)``: ``x @ w mod 2^32`` as uint32 ``(c, n)`` for ``x`` ``(c, m)`` uint32 and the int8
/// weight ``w`` ``(m, n)`` (the factory's ring check vectors and private self-check sets; ``crate::vecmat``).
#[pyfunction]
#[pyo3(signature = (x, w, threads = 1))]
fn vecmat_wrap32<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, u32>, w: PyReadonlyArray2<'py, i8>, threads: usize)
                      -> PyResult<Bound<'py, PyArray2<u32>>> {
    let (c, m) = (x.shape()[0], x.shape()[1]);
    let (wm, n) = (w.shape()[0], w.shape()[1]);
    if wm != m {
        return Err(verr(format!("x is ({c}, {m}) but w is ({wm}, {n})")));
    }
    let xs = x.as_slice().map_err(|_| verr("x must be C-contiguous"))?;
    let ws = w.as_slice().map_err(|_| verr("w must be C-contiguous"))?;
    let mut out = vec![0u32; c * n];
    py.allow_threads(|| crate::vecmat::vecmat_wrap32(xs, c, ws, m, n, &mut out, threads)).map_err(|e| verr(e.0))?;
    Ok(Array2::from_shape_vec((c, n), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// ``vecmat_p31(u, w, threads)``: ``u @ w mod (2^31 - 1)`` as uint64 ``(c, n)`` for ``u`` ``(c, m)`` uint64 in
/// ``[0, 2^31 - 1)`` and the int8 weight ``w`` ``(m, n)`` (the prime-field check vectors; ``crate::vecmat``).
#[pyfunction]
#[pyo3(signature = (u, w, threads = 1))]
fn vecmat_p31<'py>(py: Python<'py>, u: PyReadonlyArray2<'py, u64>, w: PyReadonlyArray2<'py, i8>, threads: usize)
                   -> PyResult<Bound<'py, PyArray2<u64>>> {
    let (c, m) = (u.shape()[0], u.shape()[1]);
    let (wm, n) = (w.shape()[0], w.shape()[1]);
    if wm != m {
        return Err(verr(format!("u is ({c}, {m}) but w is ({wm}, {n})")));
    }
    let us = u.as_slice().map_err(|_| verr("u must be C-contiguous"))?;
    let ws = w.as_slice().map_err(|_| verr("w must be C-contiguous"))?;
    let mut out = vec![0u64; c * n];
    py.allow_threads(|| crate::vecmat::vecmat_p31(us, c, ws, m, n, &mut out, threads)).map_err(|e| verr(e.0))?;
    Ok(Array2::from_shape_vec((c, n), out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// ``check_rows32(r, wr, cu, cw, mask, threads)``: the factory's every-row self-check for ``K <= 32``, uint8
/// ``(rows,)`` flags (``1`` where ``(wr[t] @ cu) & mask != (r[t] @ cw) & mask``; arithmetic mod 2^32): ``r``
/// ``(rows, n)`` and ``wr`` ``(rows, m)`` int64, ``cu`` ``(m, c)`` and ``cw`` ``(n, c)`` uint32, ``c <= 16``.
#[pyfunction]
#[pyo3(signature = (r, wr, cu, cw, mask, threads = 1))]
fn check_rows32<'py>(py: Python<'py>, r: PyReadonlyArray2<'py, i64>, wr: &Bound<'py, PyAny>,
                     cu: PyReadonlyArray2<'py, u32>, cw: PyReadonlyArray2<'py, u32>, mask: u32, threads: usize)
                     -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (rows, n) = (r.shape()[0], r.shape()[1]);
    let (m, c) = (cu.shape()[0], cu.shape()[1]);
    if cw.shape()[0] != n || cw.shape()[1] != c {
        return Err(verr(format!("shape mismatch: r {:?}, cu {:?}, cw {:?}", r.shape(), cu.shape(), cw.shape())));
    }
    let cerr = |w: &str| verr(format!("{w} must be C-contiguous"));
    let (rs, us, vs) = (r.as_slice().map_err(|_| cerr("r"))?, cu.as_slice().map_err(|_| cerr("cu"))?,
                        cw.as_slice().map_err(|_| cerr("cw"))?);
    let mut bad = vec![0u8; rows];
    macro_rules! go {
        ($t:ty) => {{
            let a = wr.downcast::<PyArray2<$t>>()?.try_readonly().map_err(|e| verr(format!("wr: {e}")))?;
            if a.shape()[0] != rows || a.shape()[1] != m {
                return Err(verr(format!("wr is {:?}, expected ({rows}, {m})", a.shape())));
            }
            let ws = a.as_slice().map_err(|_| cerr("wr"))?;
            py.allow_threads(|| crate::vecmat::check_rows32(rs, n, ws, m, us, vs, c, mask, &mut bad, threads))
                .map_err(|e| verr(e.0))?;
        }};
    }
    if wr.downcast::<PyArray2<i64>>().is_ok() {
        go!(i64)
    } else if wr.downcast::<PyArray2<u32>>().is_ok() {
        go!(u32)
    } else {
        return Err(verr("wr must be a 2-D int64 or uint32 array"));
    }
    Ok(PyArray1::from_vec(py, bad))
}

#[pymodule]
fn pgw_pks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(ternary, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt_wire, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_into, m)?)?;
    m.add_function(wrap_pyfunction!(reply_key, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt, m)?)?;
    m.add_function(wrap_pyfunction!(keyswitch, m)?)?;
    m.add_function(wrap_pyfunction!(switch_hint, m)?)?;
    m.add_function(wrap_pyfunction!(hint_keyswitch, m)?)?;
    m.add_function(wrap_pyfunction!(combine_planes, m)?)?;
    m.add_function(wrap_pyfunction!(a_planes, m)?)?;
    m.add_function(wrap_pyfunction!(pack_ksk, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(vecmat_wrap32, m)?)?;
    m.add_function(wrap_pyfunction!(vecmat_p31, m)?)?;
    m.add_function(wrap_pyfunction!(check_rows32, m)?)?;
    m.add_class::<KHat>()?;
    Ok(())
}
