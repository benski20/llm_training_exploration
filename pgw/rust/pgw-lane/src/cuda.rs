//! CUDA-graph replay executor: `cuGraphLaunch` through a `dlopen`ed driver (`libcuda.so.1`), so the
//! crate builds and its tests run on hosts without CUDA.
//!
//! The Python service captures one graph per `(model, linear, ring, rows)` with torch:
//! `[H2D pinned h_in -> device x, GEMV (its output zeroed inside the graph), D2H -> pinned h_out]`,
//! checks it against the digit-GEMM path, and registers the graph's `cudaGraphExec_t`, its stream
//! and the two pinned buffers here ([`GraphExecutor::register`]). A call then is: wait for the
//! device lock (until the call's deadline), copy the body into `h_in`, launch, wait (spin on
//! `cuStreamQuery` for `spin_us`, then block in `cuStreamSynchronize`), copy `h_out` into the reply.
//! No Python runs on this path.
//!
//! Only the packed form that the device reads and writes directly qualifies: rings with 4 bytes
//! per element (K = 32), whose packed `x_hat` is the little-endian low 32 bits the fused GEMV reads
//! and whose compact int32 output is the packed `y_hat`. Anything not registered answers
//! [`Status::Fallback`] so the next executor (the Python kernel path) serves it.

use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::admission::Refusal;
use crate::exec::{micros, Call, DeviceGuard, DeviceLock, Done, Executor};
use crate::proto::Status;
use crate::sync::{MutexExt, RwLockExt};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuStream = *mut c_void;
type CuGraphExec = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_NOT_READY: CuResult = 600;

/// Resolved driver entry points.
struct Driver {
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_get_current: unsafe extern "C" fn(*mut CuContext) -> CuResult,
    graph_launch: unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult,
    stream_query: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_synchronize: unsafe extern "C" fn(CuStream) -> CuResult,
}

// SAFETY: plain function pointers into the driver library, which stays loaded for the process.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

/// A function pointer of type `T` for symbol `name` of the library `h`.
///
/// # Safety
/// `h` is a live `dlopen` handle and `T` an `extern "C"` function pointer type matching the
/// symbol's C signature.
unsafe fn sym<T: Copy>(h: *mut c_void, name: &CStr) -> Result<T, String> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    let p = libc::dlsym(h, name.as_ptr());
    if p.is_null() {
        return Err(format!("libcuda has no {}", name.to_string_lossy()));
    }
    Ok(std::mem::transmute_copy::<*mut c_void, T>(&p))
}

fn load_driver() -> Result<Driver, String> {
    // SAFETY: dlopen/dlsym with NUL-terminated names and the driver API's documented signatures;
    // the handle is never closed (the driver must stay mapped while any function pointer exists).
    unsafe {
        let mut h = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if h.is_null() {
            h = libc::dlopen(c"libcuda.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        }
        if h.is_null() {
            let e = libc::dlerror();
            let msg = if e.is_null() { "unknown".to_owned() } else { CStr::from_ptr(e).to_string_lossy().into_owned() };
            return Err(format!("libcuda not loadable: {msg}"));
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

static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

fn driver() -> Result<&'static Driver, String> {
    DRIVER.get_or_init(load_driver).as_ref().map_err(Clone::clone)
}

/// True if the CUDA driver could be loaded and initialised (`Err` explains why not).
pub fn driver_status() -> Result<(), String> {
    driver().map(|_| ())
}

/// One registered graph. The pointers belong to torch objects the Python side keeps alive until it
/// calls [`GraphExecutor::clear`] (which waits for the device lock, so no launch is running).
#[derive(Clone, Debug)]
pub struct GraphSpec {
    /// CUDA device ordinal.
    pub device: i32,
    /// `cudaGraphExec_t` (same object as the driver's `CUgraphExec`).
    pub exec: usize,
    /// Stream the graph was captured on (`cudaStream_t`).
    pub stream: usize,
    /// Pinned host input buffer and its byte length (`rows * n * 4`).
    pub h_in: usize,
    /// Byte length of `h_in` used by this graph.
    pub in_len: usize,
    /// Pinned host output buffer.
    pub h_out: usize,
    /// Byte length of `h_out` used by this graph (`rows * m * 4`).
    pub out_len: usize,
}

type Key = (u32, u32, u16, u16); // (model, linear, ring, rows)

/// Executor replaying registered CUDA graphs (see the module docs).
pub struct GraphExecutor {
    graphs: RwLock<HashMap<Key, GraphSpec>>,
    locks: Mutex<HashMap<i32, Arc<DeviceLock>>>,
    contexts: Mutex<HashMap<i32, usize>>,
    enabled: AtomicBool,
    spin_us: AtomicU64,
    disabled_reason: Mutex<Option<String>>,
}

impl Default for GraphExecutor {
    fn default() -> Self {
        GraphExecutor {
            graphs: RwLock::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
            contexts: Mutex::new(HashMap::new()),
            enabled: AtomicBool::new(true),
            spin_us: AtomicU64::new(50),
            disabled_reason: Mutex::new(None),
        }
    }
}

impl GraphExecutor {
    /// The device lock shared with the Python path for `device` (created on first use).
    pub fn device_lock(&self, device: i32) -> Arc<DeviceLock> {
        self.locks.plock().entry(device).or_default().clone()
    }

    /// Use `lock` as `device`'s lock (when Python created it first).
    pub fn set_device_lock(&self, device: i32, lock: Arc<DeviceLock>) {
        self.locks.plock().insert(device, lock);
    }

    /// Spin time before blocking in `cuStreamSynchronize`.
    pub fn set_spin_us(&self, us: u64) {
        self.spin_us.store(us, Ordering::Relaxed);
    }

    /// Register a graph for `(model, linear, ring, rows)`.
    ///
    /// # Safety
    /// `exec`, `stream`, `h_in[..in_len]` and `h_out[..out_len]` must be live CUDA objects and pinned
    /// host buffers that stay valid until [`GraphExecutor::clear`] returns, and nothing else may
    /// touch the buffers while the device lock is not held.
    pub unsafe fn register(
        &self,
        model: u32,
        linear: u32,
        ring: u16,
        rows: u16,
        spec: GraphSpec,
    ) -> Result<(), String> {
        driver()?;
        if spec.exec == 0 || spec.h_in == 0 || spec.h_out == 0 || spec.in_len == 0 || spec.out_len == 0 {
            return Err("null graph or buffer".to_owned());
        }
        self.device_lock(spec.device);
        self.graphs.pwrite().insert((model, linear, ring, rows), spec);
        Ok(())
    }

    /// Number of registered graphs.
    pub fn len(&self) -> usize {
        self.graphs.pread().len()
    }

    /// True if no graph is registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget every graph (waits for each device's in-flight launch first).
    pub fn clear(&self) {
        let locks: Vec<Arc<DeviceLock>> = self.locks.plock().values().cloned().collect();
        let guards: Vec<_> = locks.iter().map(|l| DeviceGuard::acquire(l, None)).collect();
        self.graphs.pwrite().clear();
        drop(guards);
    }

    /// Stop replaying graphs (every call answers `Fallback` from now on).
    pub fn disable(&self, reason: &str) {
        self.enabled.store(false, Ordering::SeqCst);
        *self.disabled_reason.plock() = Some(reason.to_owned());
    }

    /// `None` while enabled, else why it was disabled.
    pub fn disabled_reason(&self) -> Option<String> {
        if self.enabled.load(Ordering::SeqCst) {
            None
        } else {
            self.disabled_reason.plock().clone().or_else(|| Some("disabled".to_owned()))
        }
    }

    fn make_current(&self, d: &Driver, device: i32) -> Result<(), String> {
        let ctx = {
            let mut ctxs = self.contexts.plock();
            match ctxs.get(&device) {
                Some(&c) => c,
                None => {
                    let mut dev: CuDevice = 0;
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
                    ctxs.insert(device, ctx as usize);
                    ctx as usize
                }
            }
        };
        // Ask the driver rather than caching per thread: torch may switch this thread's context
        // between launches (the self-check runs on Python threads; multi-GPU), and a stale cache
        // would launch in the wrong context. cuCtxGetCurrent is a thread-local read in the driver.
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

    fn launch_and_wait(&self, d: &Driver, g: &GraphSpec) -> Result<(), String> {
        self.make_current(d, g.device)?;
        let stream = g.stream as CuStream;
        // SAFETY: registered live graph and stream (see `register`).
        let rc = unsafe { (d.graph_launch)(g.exec as CuGraphExec, stream) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cuGraphLaunch failed ({rc})"));
        }
        let spin = Duration::from_micros(self.spin_us.load(Ordering::Relaxed));
        let t0 = Instant::now();
        loop {
            // SAFETY: as above.
            let rc = unsafe { (d.stream_query)(stream) };
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
        let rc = unsafe { (d.stream_synchronize)(stream) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cuStreamSynchronize failed ({rc})"));
        }
        Ok(())
    }

    /// Run the registered graph for `key` on `body` into `out` (no deadline): the startup
    /// self-check uses exactly the serving path through this.
    pub fn run_key(
        &self,
        key: (u32, u32, u16, u16),
        body: &[u8],
        out: &mut Vec<u8>,
        deadline: Option<Instant>,
    ) -> Result<(u32, u32), Refusal> {
        let received = Instant::now();
        let g = match self.graphs.pread().get(&key) {
            Some(g) => g.clone(),
            None => return Err(Refusal::new(Status::Fallback)),
        };
        if body.len() != g.in_len {
            return Err(Refusal::new(Status::Fallback));
        }
        let d = driver().map_err(|_| Refusal::new(Status::Fallback))?;
        let lock = self.device_lock(g.device);
        let _guard = DeviceGuard::acquire(&lock, deadline).ok_or(Refusal::new(Status::DeadlineExceeded))?;
        if !self.enabled.load(Ordering::SeqCst) {
            return Err(Refusal::new(Status::Fallback));
        }
        let t_start = Instant::now();
        // SAFETY: h_in holds at least in_len bytes (register contract) and no launch runs (lock).
        unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), g.h_in as *mut u8, g.in_len) };
        if let Err(e) = self.launch_and_wait(d, &g) {
            self.disable(&e);
            return Err(Refusal::new(Status::Internal));
        }
        out.clear();
        out.reserve(g.out_len);
        // SAFETY: h_out holds out_len bytes written by the finished graph; `out` has the capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(g.h_out as *const u8, out.as_mut_ptr(), g.out_len);
            out.set_len(g.out_len);
        }
        Ok((micros(received, t_start), micros(t_start, Instant::now())))
    }
}

impl Executor for GraphExecutor {
    fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Err(Refusal::new(Status::Fallback));
        }
        let key = (call.model, call.linear, call.ring, call.rows);
        let (_, compute_us) = self.run_key(key, call.body, out, Some(call.deadline))?;
        if out.len() != call.out_len {
            out.clear();
            self.disable("graph output length differs from the call's");
            return Err(Refusal::new(Status::Internal));
        }
        Ok(Done {
            queue_us: micros(call.received, Instant::now()).saturating_sub(compute_us),
            compute_us,
            backend: "lane_cuda_graph",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unregistered_calls_fall_back() {
        let g = GraphExecutor::default();
        let body = [0u8; 8];
        let now = Instant::now();
        let call = Call {
            tenant: "t",
            model: 0,
            linear: 0,
            ring: 0,
            rows: 1,
            body: &body,
            out_len: 8,
            received: now,
            deadline: now + Duration::from_secs(1),
        };
        assert_eq!(g.execute(&call, &mut Vec::new()).unwrap_err().status, Status::Fallback);
        g.disable("test");
        assert_eq!(g.disabled_reason().as_deref(), Some("test"));
        assert_eq!(g.execute(&call, &mut Vec::new()).unwrap_err().status, Status::Fallback);
        assert!(g.is_empty());
    }

    #[test]
    fn register_needs_a_driver_or_refuses_null() {
        let g = GraphExecutor::default();
        let spec = GraphSpec { device: 0, exec: 0, stream: 0, h_in: 0, in_len: 0, h_out: 0, out_len: 0 };
        // SAFETY: refused before any pointer is used (no driver here, or null pointers).
        assert!(unsafe { g.register(0, 0, 0, 1, spec) }.is_err());
        assert!(g.is_empty());
    }
}
