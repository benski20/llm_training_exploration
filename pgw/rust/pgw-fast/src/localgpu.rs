//! Exact local linear with the integer product on a gateway-owned GPU, in one GIL-free call
//! (`local_gpu_linear_into`; `pgw.engine.local_gpu`).
//!
//! The same steps as [`crate::local::local_linear`], with the GEMV replaced by a CUDA graph that the
//! Python side captured with torch (`[H2D pinned h_in -> device, ring GEMV mod 2^32, D2H -> pinned h_out]`):
//!
//! 1. the TripEngine's prepare ([`crate::trip::prepare_into`]) with a zero mask writes the K = 32 payload
//!    (`x_hat = xq mod 2^32`, little-endian u32) straight into the pinned input buffer;
//! 2. `cuGraphLaunch` on the graph's stream through a `dlopen`ed driver (as `pgw-lane`'s executor), then a
//!    spin on `cuStreamQuery` for `spin_us` (then `cuStreamSynchronize`): no Python, no GIL;
//! 3. the centered int32 results (`|y| < 2^31` for every accepted output: `bound < 2^31` is required) get the
//!    masked path's range check over every output and the finish kernel's rescale
//!    `((y * sx_t) * scale_j + bias_j) as f32` in float64, exactly as `local_linear`.
//!
//! Both pinned buffers are zeroed before returning (also on errors). The integer product is exact on the
//! device (wrapping 32-bit arithmetic), so the outputs are `local_linear`'s bits; the Python side
//! additionally self-checks this path against the CPU reference per linear before using it.

use crate::finish::wipe;
use crate::prepare::{PrepareError, PrepareParams};
use crate::trip::{prepare_into, PrepOut};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::ffi::{c_int, c_uint, c_void, CStr};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

type CuResult = c_int;
type CuContext = *mut c_void;
const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;

struct Driver {
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, c_int) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_get_current: unsafe extern "C" fn(*mut CuContext) -> CuResult,
    graph_launch: unsafe extern "C" fn(*mut c_void, *mut c_void) -> CuResult,
    stream_query: unsafe extern "C" fn(*mut c_void) -> CuResult,
    stream_synchronize: unsafe extern "C" fn(*mut c_void) -> CuResult,
}

// SAFETY: plain function pointers into the driver library, which stays loaded for the process.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

/// # Safety
/// `h` is a live `dlopen` handle and `T` an `extern "C"` function pointer type of the symbol.
unsafe fn sym<T: Copy>(h: *mut c_void, name: &CStr) -> Result<T, String> {
    let p = libc::dlsym(h, name.as_ptr());
    if p.is_null() {
        return Err(format!("libcuda has no {}", name.to_string_lossy()));
    }
    Ok(std::mem::transmute_copy::<*mut c_void, T>(&p))
}

fn load_driver() -> Result<Driver, String> {
    // SAFETY: dlopen/dlsym with NUL-terminated names and the driver API's documented signatures; the
    // handle is never closed.
    unsafe {
        let mut h = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if h.is_null() {
            h = libc::dlopen(c"libcuda.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        }
        if h.is_null() {
            return Err("libcuda not loadable".to_owned());
        }
        let init: unsafe extern "C" fn(c_uint) -> CuResult = sym(h, c"cuInit")?;
        let rc = init(0);
        if rc != CUDA_SUCCESS {
            return Err(format!("cuInit failed ({rc})"));
        }
        Ok(Driver {
            device_get: sym(h, c"cuDeviceGet")?,
            primary_ctx_retain: sym(h, c"cuDevicePrimaryCtxRetain")?,
            ctx_set_current: sym(h, c"cuCtxSetCurrent")?,
            ctx_get_current: sym(h, c"cuCtxGetCurrent")?,
            graph_launch: sym(h, c"cuGraphLaunch")?,
            stream_query: sym(h, c"cuStreamQuery")?,
            stream_synchronize: sym(h, c"cuStreamSynchronize")?,
        })
    }
}

fn driver() -> Result<&'static Driver, String> {
    static D: OnceLock<Result<Driver, String>> = OnceLock::new();
    D.get_or_init(load_driver).as_ref().map_err(Clone::clone)
}

/// The primary context of `device` current on this thread (retained once per device and process).
fn make_current(d: &Driver, device: i32) -> Result<(), String> {
    static CTX: std::sync::Mutex<Vec<(i32, usize)>> = std::sync::Mutex::new(Vec::new());
    let ctx = {
        let mut g = CTX.lock().unwrap_or_else(|e| e.into_inner());
        match g.iter().find(|(dv, _)| *dv == device) {
            Some(&(_, c)) => c,
            None => {
                let mut dev: c_int = 0;
                let mut ctx: CuContext = std::ptr::null_mut();
                // SAFETY: out-pointers to locals; the driver was initialised.
                unsafe {
                    let rc = (d.device_get)(&mut dev, device);
                    if rc != CUDA_SUCCESS {
                        return Err(format!("cuDeviceGet({device}) failed ({rc})"));
                    }
                    let rc = (d.primary_ctx_retain)(&mut ctx, dev);
                    if rc != CUDA_SUCCESS {
                        return Err(format!("cuDevicePrimaryCtxRetain({device}) failed ({rc})"));
                    }
                }
                g.push((device, ctx as usize));
                ctx as usize
            }
        }
    };
    let mut cur: CuContext = std::ptr::null_mut();
    // SAFETY: out-pointer to a local.
    let rc = unsafe { (d.ctx_get_current)(&mut cur) };
    if rc != CUDA_SUCCESS || cur as usize != ctx {
        // SAFETY: a context retained above (never released while the process runs).
        let rc = unsafe { (d.ctx_set_current)(ctx as CuContext) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cuCtxSetCurrent failed ({rc})"));
        }
    }
    Ok(())
}

fn launch_and_wait(d: &Driver, device: i32, exec: usize, stream: usize, spin_us: u64) -> Result<(), String> {
    make_current(d, device)?;
    let s = stream as *mut c_void;
    // SAFETY: the caller registered a live graph exec and stream of this device (Python keeps them alive).
    let rc = unsafe { (d.graph_launch)(exec as *mut c_void, s) };
    if rc != CUDA_SUCCESS {
        return Err(format!("cuGraphLaunch failed ({rc})"));
    }
    let spin = Duration::from_micros(spin_us);
    let t0 = Instant::now();
    loop {
        // SAFETY: as above.
        let rc = unsafe { (d.stream_query)(s) };
        if rc == CUDA_SUCCESS {
            return Ok(());
        }
        if rc != CUDA_ERROR_NOT_READY {
            return Err(format!("cuStreamQuery failed ({rc})"));
        }
        if t0.elapsed() >= spin {
            break;
        }
        std::hint::spin_loop();
    }
    // SAFETY: as above.
    let rc = unsafe { (d.stream_synchronize)(s) };
    if rc != CUDA_SUCCESS {
        return Err(format!("cuStreamSynchronize failed ({rc})"));
    }
    Ok(())
}

thread_local! {
    static ZEROS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

/// Pinned host buffers and the graph of one call.
struct Graph {
    device: i32,
    exec: usize,
    stream: usize,
    h_in: *mut u8,
    h_in_len: usize,
    h_out: *mut u8,
    h_out_len: usize,
    spin_us: u64,
}

enum Fail {
    Prep(PrepareError),
    Cuda(String),
}

#[allow(clippy::too_many_arguments)]
fn run(
    x: &[f32],
    n: usize,
    prm: PrepareParams,
    scale: &[f32],
    bias: Option<&[f32]>,
    bound: u64,
    sx: &mut [f64],
    out: &mut [f32],
    g: &Graph,
    launch: impl FnOnce(&[u8], &mut [u8]) -> Result<(), String>,
) -> Result<u8, Fail> {
    let rows = sx.len();
    let m = scale.len();
    // SAFETY: the caller guarantees the pinned buffers are live, at least `h_*_len` bytes and not
    // used concurrently (the Python side holds its lock); lengths were checked against rows * n * 4.
    let h_in = unsafe { std::slice::from_raw_parts_mut(g.h_in, g.h_in_len) };
    let h_out = unsafe { std::slice::from_raw_parts_mut(g.h_out, g.h_out_len) };
    let pin_in = &mut h_in[..rows * n * 4];
    let res = ZEROS.with(|z| {
        let mut z = z.borrow_mut();
        if z.len() < rows * n {
            z.resize(rows * n, 0);
        }
        prepare_into(x, &z[..rows * n], n, prm, None, PrepOut { payload: pin_in, sx, xhat: &mut [], rhs: &mut [] }, 1)
    });
    if let Err(e) = res {
        wipe(&mut h_in[..rows * n * 4]);
        wipe(out);
        return Err(Fail::Prep(e));
    }
    let launched = launch(&h_in[..rows * n * 4], &mut h_out[..rows * m * 4]);
    wipe(&mut h_in[..rows * n * 4]);
    if let Err(e) = launched {
        wipe(&mut h_out[..rows * m * 4]);
        wipe(out);
        return Err(Fail::Cuda(e));
    }
    // range check over every output (no early exit), then the rescale (local_linear's expressions)
    let mut amax: u64 = 0;
    for t in 0..rows {
        let sxt = sx[t];
        let o = &mut out[t * m..(t + 1) * m];
        let yr = &h_out[t * m * 4..(t + 1) * m * 4];
        for j in 0..m {
            let y = i32::from_le_bytes([yr[4 * j], yr[4 * j + 1], yr[4 * j + 2], yr[4 * j + 3]]) as i64;
            amax = amax.max(y.unsigned_abs());
            let v = (y as f64 * sxt) * scale[j] as f64;
            o[j] = match bias {
                Some(b) => (v + b[j] as f64) as f32,
                None => v as f32,
            };
        }
    }
    wipe(&mut h_out[..rows * m * 4]);
    if amax > bound {
        wipe(out);
        return Ok(2);
    }
    Ok(0)
}

/// Why the GPU local path cannot run here (``None`` when the CUDA driver loaded).
#[pyfunction]
fn local_gpu_driver_status() -> Option<String> {
    driver().err()
}

/// Exact local linear with the product in a captured CUDA graph (module docs).
///
/// Args:
///     x: float32 ``(rows, n)`` C-contiguous; scale/bias: float32 ``(m,)``; k must be 32, a <= 15.
///     cap, wnorm, bound: as ``local_linear_into`` (``bound < 2^31``).
///     sx: writable float64 ``(rows,)``; out: writable float32 ``(rows, m)``.
///     device: CUDA ordinal; graph_exec, stream: ``cudaGraphExec_t`` and ``cudaStream_t`` addresses;
///     h_in, h_in_bytes, h_out, h_out_bytes: the graph's pinned buffers (the graph reads exactly
///     ``rows * n`` u32 from ``h_in`` and writes ``rows * m`` int32 to ``h_out``).
///     spin_us: spin on ``cuStreamQuery`` this long before blocking.
///     hold_gil: keep the GIL for the call (no hand-off to other threads for a call of tens of microseconds).
///
/// Returns the status (0 ok, 2 an output out of range, ``out`` zeroed). Quantizer errors raise
/// ``ValueError`` (``ring_overflow: ...``), CUDA errors ``RuntimeError``. The GIL is released unless ``hold_gil``.
#[pyfunction]
#[pyo3(signature = (x, scale, bias, k, a, cap, wnorm, bound, sx, out, device, graph_exec, stream, h_in, h_in_bytes, h_out, h_out_bytes, spin_us = 2000, hold_gil = false))]
#[allow(clippy::too_many_arguments)]
fn local_gpu_linear_into<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f32>>,
    scale: &Bound<'py, PyArray1<f32>>,
    bias: Option<&Bound<'py, PyArray1<f32>>>,
    k: u32,
    a: u32,
    cap: u32,
    wnorm: f64,
    bound: u64,
    sx: &Bound<'py, PyArray1<f64>>,
    out: &Bound<'py, PyArray2<f32>>,
    device: i32,
    graph_exec: usize,
    stream: usize,
    h_in: usize,
    h_in_bytes: usize,
    h_out: usize,
    h_out_bytes: usize,
    spin_us: u64,
    hold_gil: bool,
) -> PyResult<u8> {
    let verr = |s: &str| PyValueError::new_err(s.to_string());
    if k != 32 || a > 15 {
        return Err(verr("local_gpu_linear_into needs K = 32 and A <= 15"));
    }
    if bound >= (1u64 << 31) {
        return Err(verr("bound must be below 2^31"));
    }
    if graph_exec == 0 || stream == 0 || h_in == 0 || h_out == 0 {
        return Err(verr("null graph, stream or buffer"));
    }
    let xa = x.try_readonly().map_err(|e| verr(&format!("x: {e}")))?;
    let sa = scale.try_readonly().map_err(|e| verr(&format!("scale: {e}")))?;
    let ba = match bias {
        Some(b) => Some(b.try_readonly().map_err(|e| verr(&format!("bias: {e}")))?),
        None => None,
    };
    let (rows, n) = (xa.shape()[0], xa.shape()[1]);
    let xs = xa.as_slice().map_err(|_| verr("x must be C-contiguous"))?;
    let ss = sa.as_slice().map_err(|_| verr("scale must be contiguous"))?;
    let m = ss.len();
    let bs = match &ba {
        Some(b) => Some(b.as_slice().map_err(|_| verr("bias must be contiguous"))?),
        None => None,
    };
    if bs.is_some_and(|b| b.len() != m) {
        return Err(verr("bias must have m entries"));
    }
    if rows == 0 || n == 0 || m == 0 || rows * n * 4 > h_in_bytes || rows * m * 4 > h_out_bytes {
        return Err(verr("shape does not fit the pinned buffers"));
    }
    let mut sxa = sx.try_readwrite().map_err(|e| verr(&format!("sx: {e}")))?;
    let mut oa = out.try_readwrite().map_err(|e| verr(&format!("out: {e}")))?;
    if oa.shape() != [rows, m] {
        return Err(verr("out must be (rows, m)"));
    }
    let sxs = sxa.as_slice_mut().map_err(|_| verr("sx must be contiguous"))?;
    if sxs.len() != rows {
        return Err(verr("sx must have rows entries"));
    }
    let os = oa.as_slice_mut().map_err(|_| verr("out must be C-contiguous"))?;
    let g = Graph {
        device,
        exec: graph_exec,
        stream,
        h_in: h_in as *mut u8,
        h_in_len: h_in_bytes,
        h_out: h_out as *mut u8,
        h_out_len: h_out_bytes,
        spin_us,
    };
    // raw pointers are not Send: move the addresses and rebuild inside
    let gaddr = &g as *const Graph as usize;
    let prm = PrepareParams { k, a, cap, wnorm };
    let mut work = move || {
        // SAFETY: `g` outlives this closure (it runs synchronously).
        let g = unsafe { &*(gaddr as *const Graph) };
        run(xs, n, prm, ss, bs, bound, sxs, os, g, |_, _| {
            driver().and_then(|d| launch_and_wait(d, g.device, g.exec, g.stream, g.spin_us))
        })
    };
    // hold_gil: a call takes tens of microseconds; keeping the GIL avoids handing it to another thread
    // (and waiting to get it back) once per local linear
    let res = if hold_gil { work() } else { py.allow_threads(work) };
    match res {
        Ok(s) => Ok(s),
        Err(Fail::Prep(e)) => Err(e.into_py()),
        Err(Fail::Cuda(e)) => Err(PyRuntimeError::new_err(e)),
    }
}

/// Add this module's functions to `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(local_gpu_driver_status, m)?)?;
    m.add_function(wrap_pyfunction!(local_gpu_linear_into, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::{local_linear, LocalArgs, LocalStatus};

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// The device product emulated on the CPU (wrapping u32, as the ring GEMV): same outputs as `local_linear`.
    #[test]
    fn emulated_device_product_equals_local_linear() {
        let mut seed = 5u64;
        for &(rows, n, m, with_bias) in &[(1usize, 896usize, 1152usize, true), (1, 4864, 896, false), (3, 100, 37, true)] {
            let x: Vec<f32> = (0..rows * n).map(|_| (lcg(&mut seed) as f32 / (1u64 << 31) as f32 - 0.5) * 6.0).collect();
            let w8t: Vec<i8> = (0..n * m).map(|_| (lcg(&mut seed) % 256) as u8 as i8).collect();
            let scale: Vec<f32> = (0..m).map(|_| lcg(&mut seed) as f32 * 1e-13).collect();
            let bias: Vec<f32> = (0..m).map(|_| lcg(&mut seed) as f32 * 1e-12).collect();
            let b = if with_bias { Some(&bias[..]) } else { None };
            let prm = PrepareParams { k: 32, a: 12, cap: 0, wnorm: 0.0 };
            let bound = (1u64 << 31) - 1;
            let args = LocalArgs { x: &x, w8t: &w8t, scale: &scale, bias: b, prm, bound, n };
            let mut sx_ref = vec![0f64; rows];
            let mut out_ref = vec![0f32; rows * m];
            assert_eq!(local_linear(&args, &mut sx_ref, &mut out_ref, 1).unwrap(), LocalStatus::Ok);
            let mut h_in = vec![7u8; rows * n * 4 + 8];
            let mut h_out = vec![7u8; rows * m * 4 + 8];
            let g = Graph {
                device: 0,
                exec: 1,
                stream: 1,
                h_in: h_in.as_mut_ptr(),
                h_in_len: h_in.len(),
                h_out: h_out.as_mut_ptr(),
                h_out_len: h_out.len(),
                spin_us: 0,
            };
            let mut sx = vec![0f64; rows];
            let mut out = vec![0f32; rows * m];
            let st = run(&x, n, prm, &scale, b, bound, &mut sx, &mut out, &g, |xin, yout| {
                for t in 0..rows {
                    for j in 0..m {
                        let mut acc: u32 = 0;
                        for i in 0..n {
                            let o = 4 * (t * n + i);
                            let xv = u32::from_le_bytes([xin[o], xin[o + 1], xin[o + 2], xin[o + 3]]);
                            acc = acc.wrapping_add(xv.wrapping_mul(w8t[i * m + j] as i32 as u32));
                        }
                        yout[4 * (t * m + j)..4 * (t * m + j) + 4].copy_from_slice(&acc.to_le_bytes());
                    }
                }
                Ok(())
            });
            assert!(matches!(st, Ok(0)));
            assert_eq!(sx, sx_ref);
            assert_eq!(out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                       out_ref.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "rows={rows} n={n} m={m}");
            assert!(h_in[..rows * n * 4].iter().all(|&v| v == 0) && h_out[..rows * m * 4].iter().all(|&v| v == 0));
            // a device error: outputs and pinned buffers zeroed
            let st = run(&x, n, prm, &scale, b, bound, &mut sx, &mut out, &g, |_, y| {
                y.fill(9);
                Err("boom".into())
            });
            assert!(matches!(st, Err(Fail::Cuda(_))));
            assert!(out.iter().all(|&v| v == 0.0) && h_out[..rows * m * 4].iter().all(|&v| v == 0));
            // out of range: status 2, outputs zeroed
            let st = run(&x, n, prm, &scale, b, 0, &mut sx, &mut out, &g, |_, y| {
                y.fill(1);
                Ok(())
            });
            assert!(matches!(st, Ok(2)));
            assert!(out.iter().all(|&v| v == 0.0));
        }
    }
}
