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
    /// Output-row tensor parallel (opt-in): the graphs of every shard, shard 0 first. Empty for a
    /// single-GPU linear. When set, `device`/`exec`/`stream`/`h_out` above are ignored: every shard's
    /// graph copies the same pinned `h_in` (broadcast) to its device, computes its output columns and
    /// copies them to its own pinned buffer; the reply is gathered row by row in shard order.
    pub shards: Vec<GraphShard>,
}

/// One tensor-parallel shard of a registered graph.
#[derive(Clone, Debug)]
pub struct GraphShard {
    /// CUDA device ordinal.
    pub device: i32,
    /// `cudaGraphExec_t` of the shard.
    pub exec: usize,
    /// Its capture stream.
    pub stream: usize,
    /// Pinned host buffer of the shard's `(rows, m_j)` output.
    pub h_out: usize,
    /// Bytes per row of the shard's output (`m_j * 4`).
    pub row_bytes: usize,
    /// Byte offset of the shard's columns in a reply row (`m0_j * 4`).
    pub col0: usize,
}

/// Gather tensor-parallel shard outputs into the row-major reply: row `r` of shard `j`
/// (`parts[j][r * row_bytes_j ..]`) lands at `out[r * row + col0_j ..]`. Pure (tested on CPU).
pub fn gather_rows(parts: &[(&[u8], usize, usize)], rows: usize, row: usize, out: &mut [u8]) -> Result<(), String> {
    if out.len() != rows * row {
        return Err("reply length".to_owned());
    }
    let mut covered = 0;
    for (src, row_bytes, col0) in parts {
        if src.len() < rows * row_bytes || col0 + row_bytes > row {
            return Err("shard layout".to_owned());
        }
        covered += row_bytes;
        for r in 0..rows {
            out[r * row + col0..r * row + col0 + row_bytes].copy_from_slice(&src[r * row_bytes..(r + 1) * row_bytes]);
        }
    }
    if covered != row {
        return Err("shards do not cover the row".to_owned());
    }
    Ok(())
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
        if spec.shards.is_empty() {
            if spec.exec == 0 || spec.h_in == 0 || spec.h_out == 0 || spec.in_len == 0 || spec.out_len == 0 {
                return Err("null graph or buffer".to_owned());
            }
            self.device_lock(spec.device);
        } else {
            check_shards(&spec, rows)?;
            for p in &spec.shards {
                self.device_lock(p.device);
            }
        }
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
        self.launch(d, g.device, g.exec, g.stream)?;
        self.wait(d, g.device, g.stream)
    }

    fn launch(&self, d: &Driver, device: i32, exec: usize, stream: usize) -> Result<(), String> {
        self.make_current(d, device)?;
        // SAFETY: registered live graph and stream (see `register`).
        let rc = unsafe { (d.graph_launch)(exec as CuGraphExec, stream as CuStream) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cuGraphLaunch failed ({rc})"));
        }
        Ok(())
    }

    fn wait(&self, d: &Driver, device: i32, stream: usize) -> Result<(), String> {
        self.make_current(d, device)?;
        let stream = stream as CuStream;
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
        if !g.shards.is_empty() {
            return self.run_sharded(d, &g, key.3 as usize, body, out, deadline, received);
        }
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

/// Registration check of a tensor-parallel graph: shards on distinct devices tile each reply row.
fn check_shards(spec: &GraphSpec, rows: u16) -> Result<(), String> {
    if spec.h_in == 0 || spec.in_len == 0 || spec.out_len == 0 || rows == 0 {
        return Err("null buffer".to_owned());
    }
    let row = spec.out_len / rows as usize;
    let mut cols: Vec<(usize, usize)> = spec.shards.iter().map(|p| (p.col0, p.row_bytes)).collect();
    cols.sort_unstable();
    let mut at = 0;
    for (c0, w) in cols {
        if c0 != at || w == 0 {
            return Err("tensor-parallel shards do not tile the reply".to_owned());
        }
        at += w;
    }
    if row * rows as usize != spec.out_len || at != row || spec.shards.iter().any(|p| p.exec == 0 || p.h_out == 0) {
        return Err("tensor-parallel shards do not tile the reply".to_owned());
    }
    let mut devs: Vec<i32> = spec.shards.iter().map(|p| p.device).collect();
    devs.sort_unstable();
    devs.dedup();
    if devs.len() != spec.shards.len() {
        return Err("tensor-parallel shards must be on distinct devices".to_owned());
    }
    Ok(())
}

impl GraphExecutor {
    /// A tensor-parallel graph: all shard device locks (ascending), body into the shared pinned
    /// `h_in` once, launch every shard (they run concurrently on their GPUs), wait for all, gather.
    #[allow(clippy::too_many_arguments)]
    fn run_sharded(
        &self,
        d: &Driver,
        g: &GraphSpec,
        rows: usize,
        body: &[u8],
        out: &mut Vec<u8>,
        deadline: Option<Instant>,
        received: Instant,
    ) -> Result<(u32, u32), Refusal> {
        let mut devs: Vec<i32> = g.shards.iter().map(|p| p.device).collect();
        devs.sort_unstable();
        let locks: Vec<Arc<DeviceLock>> = devs.iter().map(|dv| self.device_lock(*dv)).collect();
        let mut guards = Vec::with_capacity(locks.len());
        for l in &locks {
            guards.push(DeviceGuard::acquire(l, deadline).ok_or(Refusal::new(Status::DeadlineExceeded))?);
        }
        if !self.enabled.load(Ordering::SeqCst) {
            return Err(Refusal::new(Status::Fallback));
        }
        let t_start = Instant::now();
        // SAFETY: h_in holds in_len bytes (register contract); no shard graph runs (locks held).
        unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), g.h_in as *mut u8, g.in_len) };
        let mut res = Ok(());
        let mut launched = 0;
        for p in &g.shards {
            res = self.launch(d, p.device, p.exec, p.stream);
            if res.is_err() {
                break;
            }
            launched += 1;
        }
        // wait for every launched shard, even after an error (the buffers stay ours until then)
        for p in &g.shards[..launched] {
            let w = self.wait(d, p.device, p.stream);
            if res.is_ok() {
                res = w;
            }
        }
        if let Err(e) = res {
            self.disable(&e);
            return Err(Refusal::new(Status::Internal));
        }
        out.clear();
        out.resize(g.out_len, 0);
        let parts: Vec<(&[u8], usize, usize)> = g
            .shards
            .iter()
            // SAFETY: each shard's pinned output holds rows * row_bytes bytes written by its finished graph.
            .map(|p| {
                (unsafe { std::slice::from_raw_parts(p.h_out as *const u8, rows * p.row_bytes) }, p.row_bytes, p.col0)
            })
            .collect();
        if gather_rows(&parts, rows, g.out_len / rows.max(1), out).is_err() {
            out.clear();
            self.disable("tensor-parallel gather layout");
            return Err(Refusal::new(Status::Internal));
        }
        drop(guards);
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
        let spec =
            GraphSpec { device: 0, exec: 0, stream: 0, h_in: 0, in_len: 0, h_out: 0, out_len: 0, shards: vec![] };
        // SAFETY: refused before any pointer is used (no driver here, or null pointers).
        assert!(unsafe { g.register(0, 0, 0, 1, spec) }.is_err());
        assert!(g.is_empty());
    }

    #[test]
    fn gather_rows_interleaves_shards_in_order() {
        // 2 rows, row = 7 u32 split 3 + 2 + 2
        let rows = 2;
        let want: Vec<u8> = (0u8..56).collect();
        let row = 28;
        let shard = |c0: usize, w: usize| -> Vec<u8> {
            (0..rows).flat_map(|r| want[r * row + c0..r * row + c0 + w].to_vec()).collect()
        };
        let (a, b, c) = (shard(0, 12), shard(12, 8), shard(20, 8));
        let mut out = vec![0u8; 56];
        gather_rows(&[(&a, 12, 0), (&b, 8, 12), (&c, 8, 20)], rows, row, &mut out).unwrap();
        assert_eq!(out, want);
        assert!(gather_rows(&[(&a, 12, 0), (&b, 8, 12)], rows, row, &mut out).is_err());
        assert!(gather_rows(&[(&a, 12, 0), (&b, 8, 12), (&c, 8, 24)], rows, row, &mut out).is_err());
    }

    #[test]
    fn shard_tiling_is_checked() {
        let sh = |device, col0, row_bytes| GraphShard { device, exec: 1, stream: 1, h_out: 1, row_bytes, col0 };
        let spec =
            |shards| GraphSpec { device: 0, exec: 0, stream: 0, h_in: 1, in_len: 8, h_out: 0, out_len: 40, shards };
        assert!(check_shards(&spec(vec![sh(0, 0, 12), sh(1, 12, 8)]), 2).is_ok());
        assert!(check_shards(&spec(vec![sh(1, 12, 8), sh(0, 0, 12)]), 2).is_ok());
        assert!(check_shards(&spec(vec![sh(0, 0, 12), sh(0, 12, 8)]), 2).is_err()); // same device twice
        assert!(check_shards(&spec(vec![sh(0, 0, 12), sh(1, 16, 4)]), 2).is_err()); // gap
        assert!(check_shards(&spec(vec![sh(0, 0, 12), sh(1, 8, 12)]), 2).is_err()); // overlap
        assert!(check_shards(&spec(vec![sh(0, 0, 12)]), 2).is_err()); // short
    }
}
