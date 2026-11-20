//! Python module `pgw_ntt` (wrapped by `pgw.hemask.ntt`). Every function releases the GIL while it
//! computes and returns `uint64` arrays equal, bit for bit, to the dense products of
//! `pgw.hemask.fastmm` / `pgw.hemask.lwe.matmul_mod64`.

use crate::arith::{bound_ok, params, N};
use crate::kernel::{self, Big, Isa, Secrets};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn isa_of(name: &str) -> PyResult<Isa> {
    Isa::parse(name).ok_or_else(|| verr(format!("unknown isa {name:?} (auto, avx512, avx2, scalar)")))
}

fn kerr(e: kernel::Error) -> PyErr {
    verr(e.to_string())
}

/// Secrets `(G, rank, 256)` int8, C-contiguous.
fn secrets_of(s: &Bound<'_, PyAny>) -> PyResult<(Vec<i8>, usize, usize)> {
    let a: PyReadonlyArray3<'_, i8> =
        s.extract().map_err(|_| verr("secrets must be a 3-D int8 array (G, rank, 256)"))?;
    let sh = a.shape();
    if sh[2] != N || sh[0] == 0 || sh[1] == 0 {
        return Err(verr(format!("secrets must be (G, rank, {N}), got {sh:?}")));
    }
    let v = a.as_slice().map_err(|_| verr("secrets must be C-contiguous"))?.to_vec();
    Ok((v, sh[0], sh[1]))
}

fn to_out<'py>(py: Python<'py>, v: Vec<u64>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyArray2<u64>>> {
    Ok(Array2::from_shape_vec((rows, cols), v).map_err(|e| verr(e.to_string()))?.into_pyarray(py))
}

/// Constants: ``primes`` (list), ``half`` (decimal string of ``p0 p1 (p2 - 1) / 2``: products with
/// ``|Z| <= half`` are exact), ``isa`` (instruction set ``auto`` resolves to), ``lanes``, ``version``.
#[pyfunction]
fn info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    let p = params();
    d.set_item("primes", p.primes.iter().map(|q| q.p).collect::<Vec<u32>>())?;
    d.set_item("half", p.crt.half.to_string())?;
    d.set_item("isa", Isa::Auto.resolve().name())?;
    d.set_item("lanes", kernel::L)?;
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    Ok(d)
}

/// The instruction set ``isa`` resolves to on this CPU.
#[pyfunction]
#[pyo3(signature = (isa = "auto"))]
fn resolve_isa(isa: &str) -> PyResult<&'static str> {
    Ok(isa_of(isa)?.resolve().name())
}

/// ``True`` when a big operand below ``2^big_bits`` times secrets of L1 norm ``l1`` is exact.
#[pyfunction]
fn exact_for(big_bits: u32, l1: u64) -> bool {
    bound_ok(big_bits, l1)
}

/// Secrets ``(G, rank, 256)`` of a block-negacyclic int8 matrix ``S`` (``rank*256 x G*256``, as
/// ``pgw.hemask.lwe.secret_matrix`` builds it, groups concatenated along columns), or ``None`` if
/// ``S`` does not have exactly that structure.
#[pyfunction]
fn extract_secrets<'py>(py: Python<'py>, smat: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyArray3<i8>>>> {
    let a: PyReadonlyArray2<'py, i8> = smat.extract().map_err(|_| verr("smat must be a 2-D int8 array"))?;
    let (dim, cols) = (a.shape()[0], a.shape()[1]);
    let s = a.as_slice().map_err(|_| verr("smat must be C-contiguous"))?;
    let got = py.allow_threads(|| kernel::extract_secrets(s, dim, cols));
    match got {
        None => Ok(None),
        Some(v) => Ok(Some(
            Array3::from_shape_vec((cols / N, dim / N, N), v).map_err(|e| verr(e.to_string()))?.into_pyarray(py),
        )),
    }
}

/// ``big @ S mod 2^64`` for a ``(rows, rank*256)`` uint64 ``big`` and the block-negacyclic ``S`` of
/// ``secrets`` ``(G, rank, 256)``: ``(rows, G*256)`` uint64.
#[pyfunction]
#[pyo3(signature = (big, secrets, threads = 0, isa = "auto"))]
fn mm_u64<'py>(
    py: Python<'py>,
    big: &Bound<'py, PyAny>,
    secrets: &Bound<'py, PyAny>,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let isa = isa_of(isa)?;
    let (s, g, rank) = secrets_of(secrets)?;
    let a: PyReadonlyArray2<'py, u64> = big.extract().map_err(|_| verr("big must be a 2-D uint64 array"))?;
    let (rows, dim) = (a.shape()[0], a.shape()[1]);
    let data = a.as_slice().map_err(|_| verr("big must be C-contiguous"))?;
    let out = py.allow_threads(|| {
        let sec = Secrets::new(&s, g, rank);
        kernel::product(Big::U64 { data, rows }, dim, &sec, threads, isa)
    });
    to_out(py, out.map_err(kerr)?, rows, g * N)
}

/// ``U @ S mod 2^64`` with ``U`` given as int8 byte planes ``(rows, P, rank*256)`` (plane ``d`` holds
/// byte ``first_byte + d`` of each word, xor 0x80, as ``pgw.hemask.fastmm.to_planes``); missing
/// bytes are zero, exactly as ``fastmm.big_planes_mm_small(planes, S, first_byte=...)``.
#[pyfunction]
#[pyo3(signature = (planes, secrets, first_byte, threads = 0, isa = "auto"))]
fn mm_planes<'py>(
    py: Python<'py>,
    planes: &Bound<'py, PyAny>,
    secrets: &Bound<'py, PyAny>,
    first_byte: usize,
    threads: usize,
    isa: &str,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let isa = isa_of(isa)?;
    let (s, g, rank) = secrets_of(secrets)?;
    let a: PyReadonlyArray3<'py, i8> = planes.extract().map_err(|_| verr("planes must be a 3-D int8 array"))?;
    let (rows, npl, dim) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let data = a.as_slice().map_err(|_| verr("planes must be C-contiguous"))?;
    let out = py.allow_threads(|| {
        let sec = Secrets::new(&s, g, rank);
        kernel::product(Big::Planes { data, rows, npl, first: first_byte }, dim, &sec, threads, isa)
    });
    to_out(py, out.map_err(kerr)?, rows, g * N)
}

/// A big operand transformed once (residues + forward NTTs, 12 bytes per coefficient), for operands
/// reused across calls: the public matrix ``A`` (encryption) or a hint kept in NTT form.
/// ``mm(secrets, rows)`` multiplies its first ``rows`` rows.
#[pyclass(frozen, module = "pgw_ntt")]
struct Transformed {
    data: Vec<u32>,
    rows: usize,
    rank: usize,
    bits: u32,
}

#[pymethods]
impl Transformed {
    /// From ``(rows, rank*256)`` uint64 words.
    #[staticmethod]
    #[pyo3(signature = (big, threads = 0, isa = "auto"))]
    fn from_u64(py: Python<'_>, big: &Bound<'_, PyAny>, threads: usize, isa: &str) -> PyResult<Transformed> {
        let isa = isa_of(isa)?;
        let a: PyReadonlyArray2<'_, u64> = big.extract().map_err(|_| verr("big must be a 2-D uint64 array"))?;
        let (rows, dim) = (a.shape()[0], a.shape()[1]);
        if dim == 0 || dim % N != 0 {
            return Err(verr(format!("columns must be a positive multiple of {N}, got {dim}")));
        }
        let data = a.as_slice().map_err(|_| verr("big must be C-contiguous"))?;
        let rank = dim / N;
        let t = py.allow_threads(|| kernel::transform(Big::U64 { data, rows }, rank, threads, isa)).map_err(kerr)?;
        Ok(Transformed { data: t, rows, rank, bits: 64 })
    }

    /// From int8 byte planes ``(rows, P, rank*256)`` holding bytes ``first_byte .. first_byte + P``.
    #[staticmethod]
    #[pyo3(signature = (planes, first_byte, threads = 0, isa = "auto"))]
    fn from_planes(
        py: Python<'_>,
        planes: &Bound<'_, PyAny>,
        first_byte: usize,
        threads: usize,
        isa: &str,
    ) -> PyResult<Transformed> {
        let isa = isa_of(isa)?;
        let a: PyReadonlyArray3<'_, i8> = planes.extract().map_err(|_| verr("planes must be a 3-D int8 array"))?;
        let (rows, npl, dim) = (a.shape()[0], a.shape()[1], a.shape()[2]);
        if dim == 0 || dim % N != 0 {
            return Err(verr(format!("columns must be a positive multiple of {N}, got {dim}")));
        }
        let data = a.as_slice().map_err(|_| verr("planes must be C-contiguous"))?;
        let rank = dim / N;
        let big = Big::Planes { data, rows, npl, first: first_byte };
        let bits = big.bits();
        let t = py.allow_threads(|| kernel::transform(big, rank, threads, isa)).map_err(kerr)?;
        Ok(Transformed { data: t, rows, rank, bits })
    }

    /// ``U[:rows] @ S mod 2^64`` (``rows`` defaults to all): ``(rows, G*256)`` uint64.
    #[pyo3(signature = (secrets, rows = None, threads = 0, isa = "auto"))]
    fn mm<'py>(
        &self,
        py: Python<'py>,
        secrets: &Bound<'py, PyAny>,
        rows: Option<usize>,
        threads: usize,
        isa: &str,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let isa = isa_of(isa)?;
        let rows = rows.unwrap_or(self.rows);
        if rows > self.rows {
            return Err(verr(format!("asked for {rows} rows, the operand has {}", self.rows)));
        }
        let (s, g, rank) = secrets_of(secrets)?;
        if rank != self.rank {
            return Err(verr(format!("secrets have rank {rank}, the operand rank {}", self.rank)));
        }
        let out = py.allow_threads(|| {
            let sec = Secrets::new(&s, g, rank);
            kernel::product(Big::Pre { data: &self.data, rows, bits: self.bits }, rank * N, &sec, threads, isa)
        });
        to_out(py, out.map_err(kerr)?, rows, g * N)
    }

    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    #[getter]
    fn rank(&self) -> usize {
        self.rank
    }

    /// Bits of the largest value of the original operand (for the exactness bound).
    #[getter]
    fn big_bits(&self) -> u32 {
        self.bits
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// The `pgw_ntt` Python module.
#[pymodule]
fn pgw_ntt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_isa, m)?)?;
    m.add_function(wrap_pyfunction!(exact_for, m)?)?;
    m.add_function(wrap_pyfunction!(extract_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(mm_u64, m)?)?;
    m.add_function(wrap_pyfunction!(mm_planes, m)?)?;
    m.add_class::<Transformed>()?;
    Ok(())
}
