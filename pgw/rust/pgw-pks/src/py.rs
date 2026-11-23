//! Python module `pgw_pks` (wrapped by `pgw.hemask.pks_native`). Every function releases the GIL while
//! it computes; results equal the numpy reference `pgw.hemask.pks` bit for bit (`tests/hemask/test_pks.py`).

use crate::arith::{unpack_bits as unpack, Isa};
use crate::kernel::{self, DecItem, C0, EncItem, KsItem, Wiping, D, NR, NU, PACK, RANK};
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
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
    y: &Bound<'py, PyBytes>,
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
    let data = y.as_bytes();
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
    let it = KsItem { hs: hsl, m, ops: osl, groups, y: ysl, yb };
    py.allow_threads(|| kernel::keyswitch(isa, &it, &mut out, threads)).map_err(kerr)?;
    Ok(Array3::from_shape_vec(shape, out).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// :func:`pgw.hemask.wire.unpack_bits` (uint64 values).
#[pyfunction]
fn unpack_bits<'py>(py: Python<'py>, data: &Bound<'py, PyBytes>, count: usize, bits: u32) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let d = data.as_bytes();
    let v = py.allow_threads(|| unpack(d, count, bits)).map_err(verr)?;
    Ok(PyArray1::from_vec(py, v))
}

#[pymodule]
fn pgw_pks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(ternary, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(reply_key, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt, m)?)?;
    m.add_function(wrap_pyfunction!(keyswitch, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bits, m)?)?;
    m.add_class::<KHat>()?;
    Ok(())
}
