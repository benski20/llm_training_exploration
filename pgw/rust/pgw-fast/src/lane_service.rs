//! Python bindings of the service trip lane (`rust/pgw-lane`, chat-speed plan WP-A1; owned by WP-A1).
//!
//! Registers `LaneService`, `LaneDeviceLock` and `lane_cuda_driver_status` in `pgw_fast`. The Python
//! side (`pgw.service.lane`) owns the lifecycle: it starts the listener, publishes model tables,
//! issues tickets, shares admission with the stream channel, drains the counters into Prometheus,
//! captures and self-checks CUDA graphs and registers them here.
//!
//! The optional Python executor (`executor(model, linear, ring, rows, body, tenant, remaining_s)`)
//! serves calls no graph covers: it returns the packed `y_hat` (bytes-like) or an int lane status
//! to refuse with; an exception refuses with `internal` (the client falls back to the stream channel).

use std::sync::Arc;
use std::time::{Duration, Instant};

use pgw_lane::admission::{Admission, Limits, Refusal};
use pgw_lane::coalesce::Coalescer;
use pgw_lane::cuda::{driver_status, GraphExecutor, GraphShard, GraphSpec};
use pgw_lane::exec::{micros, Call, Chain, DeviceLock, Done, Executor, LinearEntry, ModelEntry, RingEntry};
use pgw_lane::persistent::{GroupSpec, LinearSlot, PersistentExecutor, WorkerSpec};
use pgw_lane::proto::{Status, TICKET_LEN};
use pgw_lane::server::{LaneServer, ServerConfig, TlsListen};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PyTuple};

struct PyExecutor {
    func: Py<PyAny>,
}

impl Executor for PyExecutor {
    fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
        let t0 = Instant::now();
        let remaining = call.deadline.saturating_duration_since(t0).as_secs_f64();
        if remaining <= 0.0 {
            return Err(Refusal::new(Status::DeadlineExceeded));
        }
        Python::with_gil(|py| {
            let t_gil = Instant::now();
            let body = PyBytes::new(py, call.body);
            let args = (call.model, call.linear, call.ring, call.rows, body, call.tenant, remaining);
            let r = match self.func.bind(py).call1(args) {
                Ok(r) => r,
                Err(e) => {
                    e.print_and_set_sys_last_vars(py);
                    return Err(Refusal::new(Status::Internal));
                }
            };
            if r.is_instance_of::<PyInt>() {
                let code: u16 = r.extract().unwrap_or(Status::Internal.code());
                let st = Status::from_code(code).filter(|s| *s != Status::Ok).unwrap_or(Status::Internal);
                return Err(Refusal::new(st));
            }
            match pyo3::buffer::PyBuffer::<u8>::get(&r) {
                Ok(buf) => {
                    if buf.len_bytes() != call.out_len {
                        return Err(Refusal::new(Status::Internal));
                    }
                    out.resize(call.out_len, 0);
                    if buf.copy_to_slice(py, out).is_err() {
                        out.clear();
                        return Err(Refusal::new(Status::Internal));
                    }
                }
                Err(_) => return Err(Refusal::new(Status::Internal)),
            }
            let now = Instant::now();
            Ok(Done { queue_us: micros(call.received, t_gil), compute_us: micros(t_gil, now), backend: "lane_python" })
        })
    }
}

/// A device lock shared between the lane and `pgw.service.gpu_exec.DeviceExecutor` (drop-in for
/// `threading.Lock`: `acquire(blocking=True, timeout=-1)`, `release()`, `locked()`, context manager).
#[pyclass(name = "LaneDeviceLock", module = "pgw_fast", frozen)]
pub struct PyDeviceLock {
    inner: Arc<DeviceLock>,
}

#[pymethods]
impl PyDeviceLock {
    #[new]
    fn new() -> Self {
        PyDeviceLock { inner: Arc::new(DeviceLock::default()) }
    }

    #[pyo3(signature = (blocking = true, timeout = -1.0))]
    fn acquire(&self, py: Python<'_>, blocking: bool, timeout: f64) -> bool {
        if !blocking {
            return self.inner.try_acquire();
        }
        if self.inner.try_acquire() {
            return true;
        }
        let deadline = if timeout < 0.0 { None } else { Some(Instant::now() + Duration::from_secs_f64(timeout)) };
        let l = self.inner.clone();
        py.allow_threads(move || l.acquire(deadline))
    }

    fn release(&self) -> PyResult<()> {
        if self.inner.release() {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("release unlocked lock"))
        }
    }

    fn locked(&self) -> bool {
        self.inner.locked()
    }

    fn __enter__(&self, py: Python<'_>) -> bool {
        self.acquire(py, true, -1.0)
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.release()?;
        Ok(false)
    }
}

/// The running lane (see the module docs).
#[pyclass(name = "LaneService", module = "pgw_fast", frozen)]
pub struct PyLaneService {
    server: LaneServer,
    admission: Arc<Admission>,
    graphs: Arc<GraphExecutor>,
    persistent: Arc<PersistentExecutor>,
    coalescer: Option<Arc<Coalescer>>,
    has_py: bool,
}

fn status_tuple(r: Result<(), Refusal>) -> (u16, u32) {
    match r {
        Ok(()) => (0, 0),
        Err(e) => (e.status.code(), e.retry_after_ms),
    }
}

#[allow(clippy::too_many_arguments)]
fn limits(
    max_inflight: u64,
    tenant_max_inflight: Option<u64>,
    tenant_rows_per_s: Option<f64>,
    tenant_burst_rows: Option<f64>,
    retry_after_ms: u32,
) -> PyResult<Limits> {
    if tenant_rows_per_s.is_some_and(|r| r <= 0.0 || !r.is_finite()) || tenant_burst_rows.is_some_and(|b| b <= 0.0) {
        return Err(PyValueError::new_err("rates and bursts must be positive"));
    }
    Ok(Limits { max_inflight, tenant_max_inflight, tenant_rows_per_s, tenant_burst_rows, retry_after_ms })
}

#[pymethods]
impl PyLaneService {
    #[new]
    #[pyo3(signature = (path, *, peer_uids = Vec::new(), max_rows = 8, deadline_cap_ms = 120_000,
                        max_connections = 256, max_inflight = 256, tenant_max_inflight = None,
                        tenant_rows_per_s = None, tenant_burst_rows = None, retry_after_ms = 1000,
                        executor = None, use_graphs = false, spin_us = 50, poll_ms = 100,
                        open_timeout_s = 10.0, stall_timeout_s = 30.0, tls_addr = None, tls_cert_pem = None,
                        tls_key_pem = None, tls_client_ca_pem = None, require_client_cert = false, read_spin_us = 0,
                        persistent = false, coalesce = false, coalesce_max_rows = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: std::path::PathBuf,
        peer_uids: Vec<u32>,
        max_rows: u16,
        deadline_cap_ms: u32,
        max_connections: usize,
        max_inflight: u64,
        tenant_max_inflight: Option<u64>,
        tenant_rows_per_s: Option<f64>,
        tenant_burst_rows: Option<f64>,
        retry_after_ms: u32,
        executor: Option<Py<PyAny>>,
        use_graphs: bool,
        spin_us: u64,
        poll_ms: u64,
        open_timeout_s: f64,
        stall_timeout_s: f64,
        tls_addr: Option<String>,
        tls_cert_pem: Option<Vec<u8>>,
        tls_key_pem: Option<Vec<u8>>,
        tls_client_ca_pem: Option<Vec<u8>>,
        require_client_cert: bool,
        read_spin_us: u64,
        persistent: bool,
        coalesce: bool,
        coalesce_max_rows: u16,
    ) -> PyResult<Self> {
        // TCP lane: TLS 1.3 only (there is no plaintext TCP listener)
        let tls = match tls_addr {
            None => {
                if tls_cert_pem.is_some() || tls_key_pem.is_some() || tls_client_ca_pem.is_some() {
                    return Err(PyValueError::new_err("TLS material given without tls_addr"));
                }
                None
            }
            Some(a) => {
                let addr: std::net::SocketAddr =
                    a.parse().map_err(|_| PyValueError::new_err("tls_addr must be 'ip:port'"))?;
                let (Some(cert), Some(key)) = (tls_cert_pem.as_deref(), tls_key_pem.as_deref()) else {
                    return Err(PyValueError::new_err("the TCP lane needs tls_cert_pem and tls_key_pem"));
                };
                let config = pgw_lane::tls::server_config(cert, key, tls_client_ca_pem.as_deref(), require_client_cert)
                    .map_err(PyValueError::new_err)?;
                Some(TlsListen { addr, config })
            }
        };
        let admission = Arc::new(Admission::new(limits(
            max_inflight,
            tenant_max_inflight,
            tenant_rows_per_s,
            tenant_burst_rows,
            retry_after_ms,
        )?));
        let graphs = Arc::new(GraphExecutor::default());
        graphs.set_spin_us(spin_us);
        let has_py = executor.is_some();
        let secondary: Option<Arc<dyn Executor>> =
            executor.map(|f| Arc::new(PyExecutor { func: f }) as Arc<dyn Executor>);
        // device executors, tried in order: persistent worker (opt-in), CUDA graphs; then optionally
        // wrapped in the coalescer (opt-in); the Python path last (it batches in the BatchScheduler)
        let gl = graphs.clone();
        let pers = Arc::new(PersistentExecutor::new(Arc::new(move |d| gl.device_lock(d))));
        pers.set_spin_us(spin_us.max(200));
        let graph_ex: Option<Arc<dyn Executor>> = if use_graphs { Some(graphs.clone()) } else { None };
        let device: Option<Arc<dyn Executor>> = if persistent {
            Some(Arc::new(Chain { primary: Some(pers.clone()), secondary: graph_ex }))
        } else {
            graph_ex
        };
        let mut coalescer = None;
        let primary: Option<Arc<dyn Executor>> = match device {
            Some(d) if coalesce => {
                let c = Arc::new(Coalescer::new(d, if coalesce_max_rows == 0 { max_rows } else { coalesce_max_rows }));
                coalescer = Some(c.clone());
                Some(c as Arc<dyn Executor>)
            }
            other => other,
        };
        let chain: Arc<dyn Executor> = Arc::new(Chain { primary, secondary });
        let cfg = ServerConfig {
            path,
            tls,
            peer_uids,
            max_rows,
            deadline_cap_ms,
            max_connections,
            open_timeout: Duration::from_secs_f64(open_timeout_s.max(0.001)),
            stall_timeout: Duration::from_secs_f64(stall_timeout_s.max(0.001)),
            poll: Duration::from_millis(poll_ms.max(1)),
            read_spin: Duration::from_micros(read_spin_us.min(100_000)),
        };
        let adm = admission.clone();
        let server = py
            .allow_threads(move || LaneServer::start(cfg, adm, Some(chain)))
            .map_err(|e| PyOSError::new_err(format!("lane listener: {e}")))?;
        Ok(PyLaneService { server, admission, graphs, persistent: pers, coalescer, has_py })
    }

    /// Socket path.
    #[getter]
    fn path(&self) -> String {
        self.server.path().to_string_lossy().into_owned()
    }

    /// True if a Python executor serves calls without a graph.
    #[getter]
    fn has_python_executor(&self) -> bool {
        self.has_py
    }

    /// Publish model `id`: `linears = [(n, m, kind)]` in table order, `rings = [(name, bytes_per_element)]`.
    fn set_model(&self, id: u32, linears: Vec<(u32, u32, String)>, rings: Vec<(String, u32)>) -> PyResult<()> {
        if rings.iter().any(|(_, b)| *b == 0 || *b > 8) {
            return Err(PyValueError::new_err("bytes_per_element must be in 1..=8"));
        }
        self.server.set_model(
            id,
            ModelEntry {
                linears: linears.into_iter().map(|(n, m, kind)| LinearEntry { n, m, kind }).collect(),
                rings: rings.into_iter().map(|(name, b)| RingEntry { name, bytes_per_element: b }).collect(),
            },
        );
        Ok(())
    }

    fn remove_model(&self, id: u32) {
        self.server.remove_model(id);
    }

    /// TCP port of the TLS listener (`None` on the Unix domain socket).
    #[getter]
    fn port(&self) -> Option<u16> {
        self.server.local_addr().map(|a| a.port())
    }

    /// Accept a single-use 32-byte `ticket` for `tenant` and model `model` for `ttl_s` seconds. With
    /// `cert_der` (the DER client certificate that asked for it), the ticket opens only a TLS connection
    /// presenting that certificate.
    #[pyo3(signature = (ticket, tenant, model, ttl_s, cert_der = None))]
    fn add_ticket(&self, ticket: &[u8], tenant: &str, model: u32, ttl_s: f64, cert_der: Option<&[u8]>) -> PyResult<()> {
        let t: [u8; TICKET_LEN] = ticket.try_into().map_err(|_| PyValueError::new_err("ticket must be 32 bytes"))?;
        let cert = cert_der.map(pgw_lane::tls::cert_digest);
        self.server.add_bound_ticket(t, tenant, model, Duration::from_secs_f64(ttl_s.max(0.0)), cert);
        Ok(())
    }

    fn pending_tickets(&self) -> usize {
        self.server.pending_tickets()
    }

    fn revoke_tenant(&self, tenant: &str) {
        self.server.revoke_tenant(tenant);
    }

    fn connections(&self) -> usize {
        self.server.connections()
    }

    /// Shared admission: `(status, retry_after_ms)`, status 0 = admitted (and, with `enter`,
    /// counted in flight until `exit`).
    #[pyo3(signature = (tenant, rows, enter = true))]
    fn admit(&self, tenant: &str, rows: u64, enter: bool) -> (u16, u32) {
        status_tuple(self.admission.admit(tenant, rows, enter))
    }

    fn exit(&self, tenant: &str) {
        self.admission.exit(tenant);
    }

    fn inflight(&self) -> u64 {
        self.admission.inflight()
    }

    fn tenant_inflight(&self, tenant: &str) -> u64 {
        self.admission.tenant_inflight(tenant)
    }

    fn set_draining(&self, on: bool) {
        self.admission.set_draining(on);
    }

    #[getter]
    fn draining(&self) -> bool {
        self.admission.draining()
    }

    #[pyo3(signature = (*, max_inflight, tenant_max_inflight = None, tenant_rows_per_s = None,
                        tenant_burst_rows = None, retry_after_ms = 1000))]
    fn set_limits(
        &self,
        max_inflight: u64,
        tenant_max_inflight: Option<u64>,
        tenant_rows_per_s: Option<f64>,
        tenant_burst_rows: Option<f64>,
        retry_after_ms: u32,
    ) -> PyResult<()> {
        self.admission.set_limits(limits(
            max_inflight,
            tenant_max_inflight,
            tenant_rows_per_s,
            tenant_burst_rows,
            retry_after_ms,
        )?);
        Ok(())
    }

    /// Counters since the last drain: `{"requests": [(tenant, status, [bucket counts], sum, count)],
    /// "rows": [(tenant, kind, rows)], "bytes": [(tenant, in, out)], "kernel": [(kind, backend, calls,
    /// rows, seconds)], "connections": (opened, refused)}`.
    fn drain_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.server.shared().stats.drain();
        let d = PyDict::new(py);
        let req: Vec<(String, &str, Vec<u64>, f64, u64)> =
            s.requests.into_iter().map(|((t, st), h)| (t, st, h.buckets, h.sum, h.count)).collect();
        d.set_item("requests", req)?;
        let rows: Vec<(String, String, u64)> = s.rows.into_iter().map(|((t, k), r)| (t, k, r)).collect();
        d.set_item("rows", rows)?;
        let bytes: Vec<(String, u64, u64)> = s.bytes.into_iter().map(|(t, (i, o))| (t, i, o)).collect();
        d.set_item("bytes", bytes)?;
        let kernel: Vec<(String, &str, u64, u64, f64)> =
            s.kernel.into_iter().map(|((k, b), (c, r, secs))| (k, b, c, r, secs)).collect();
        d.set_item("kernel", kernel)?;
        d.set_item("connections", s.connections)?;
        Ok(d)
    }

    /// The device lock of CUDA ordinal `device` (install it as the Python executor's lock).
    fn device_lock(&self, device: i32) -> PyDeviceLock {
        PyDeviceLock { inner: self.graphs.device_lock(device) }
    }

    /// Use an existing lock for `device`.
    fn set_device_lock(&self, device: i32, lock: &PyDeviceLock) {
        self.graphs.set_device_lock(device, lock.inner.clone());
    }

    /// Register a captured graph for `(model, linear, ring, rows)`. The caller keeps the graph,
    /// stream and pinned buffers alive until `clear_graphs()` returns. With `shards` (tensor
    /// parallel) every shard's graph reads `h_in` and writes its own pinned output.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, linear, ring, rows, device, exec, stream, h_in, in_len, h_out, out_len, shards = None))]
    fn register_graph(
        &self,
        model: u32,
        linear: u32,
        ring: u16,
        rows: u16,
        device: i32,
        exec: usize,
        stream: usize,
        h_in: usize,
        in_len: usize,
        h_out: usize,
        out_len: usize,
        shards: Option<Vec<(i32, usize, usize, usize, usize, usize)>>,
    ) -> PyResult<()> {
        // shards: tensor parallel, `[(device, exec, stream, h_out, row_bytes, col0)]` in shard order
        let shards = shards
            .unwrap_or_default()
            .into_iter()
            .map(|(device, exec, stream, h_out, row_bytes, col0)| GraphShard {
                device,
                exec,
                stream,
                h_out,
                row_bytes,
                col0,
            })
            .collect();
        let spec = GraphSpec { device, exec, stream, h_in, in_len, h_out, out_len, shards };
        // SAFETY: the Python owner guarantees the pointers stay valid until `clear_graphs` (documented contract).
        unsafe { self.graphs.register(model, linear, ring, rows, spec) }.map_err(PyRuntimeError::new_err)
    }

    /// Run one registered graph on `body` exactly as a lane call would (startup self-check).
    fn run_graph<'py>(
        &self,
        py: Python<'py>,
        model: u32,
        linear: u32,
        ring: u16,
        rows: u16,
        body: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let g = self.graphs.clone();
        let input = body.to_vec();
        let r = py.allow_threads(move || {
            let mut out = Vec::new();
            g.run_key((model, linear, ring, rows), &input, &mut out, None).map(|_| out)
        });
        match r {
            Ok(out) => Ok(PyBytes::new(py, &out)),
            Err(e) => Err(PyRuntimeError::new_err(format!("graph replay refused: {}", e.status.name()))),
        }
    }

    fn disable_graphs(&self, reason: &str) {
        self.graphs.disable(reason);
    }

    #[getter]
    fn graphs_disabled_reason(&self) -> Option<String> {
        self.graphs.disabled_reason()
    }

    #[getter]
    fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    fn clear_graphs(&self, py: Python<'_>) {
        let g = self.graphs.clone();
        py.allow_threads(move || g.clear());
    }

    /// Register persistent-worker group `group`: `workers = [(device, ctrl_addr)]` (shard order), the
    /// shared mapped input and output buffers. The caller keeps the memory and the resident kernels
    /// alive until `clear_persistent()` returns.
    #[allow(clippy::too_many_arguments)]
    fn register_persistent_group(
        &self,
        group: u32,
        workers: Vec<(i32, usize)>,
        h_in: usize,
        in_cap: usize,
        h_out: usize,
        out_cap: usize,
    ) -> PyResult<()> {
        let spec = GroupSpec {
            workers: workers.into_iter().map(|(device, ctrl)| WorkerSpec { device, ctrl }).collect(),
            h_in,
            in_cap,
            h_out,
            out_cap,
        };
        // SAFETY: the Python owner keeps the memory valid until `clear_persistent` (documented contract).
        unsafe { self.persistent.register_group(group, spec) }.map_err(PyRuntimeError::new_err)
    }

    /// Serve `(model, linear, ring)` from table entry `table` of group `group`.
    #[allow(clippy::too_many_arguments)]
    fn register_persistent_linear(
        &self,
        model: u32,
        linear: u32,
        ring: u16,
        group: u32,
        table: u32,
        n: u32,
        m: u32,
        max_rows: u16,
    ) -> PyResult<()> {
        self.persistent
            .register_linear(model, linear, ring, LinearSlot { group, table, n, m, max_rows })
            .map_err(PyRuntimeError::new_err)
    }

    /// One trip through the persistent workers exactly as a lane call would (startup self-check, bench).
    fn run_persistent<'py>(
        &self,
        py: Python<'py>,
        model: u32,
        linear: u32,
        ring: u16,
        rows: u16,
        body: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let p = self.persistent.clone();
        let input = body.to_vec();
        let r = py.allow_threads(move || {
            let mut out = Vec::new();
            p.run_key((model, linear, ring), rows, &input, &mut out, None).map(|_| out)
        });
        match r {
            Ok(out) => Ok(PyBytes::new(py, &out)),
            Err(e) => Err(PyRuntimeError::new_err(format!("persistent trip refused: {}", e.status.name()))),
        }
    }

    fn set_persistent_timeout_us(&self, us: u64) {
        self.persistent.set_timeout_us(us);
    }

    fn disable_persistent(&self, reason: &str) {
        self.persistent.disable(reason);
    }

    #[getter]
    fn persistent_disabled_reason(&self) -> Option<String> {
        self.persistent.disabled_reason()
    }

    #[getter]
    fn persistent_count(&self) -> usize {
        self.persistent.len()
    }

    /// Forget the persistent workers (waits for in-flight trips). Stop the kernels only after this.
    fn clear_persistent(&self, py: Python<'_>) {
        let p = self.persistent.clone();
        py.allow_threads(move || p.clear());
    }

    /// Coalescer counters (`None` when coalescing is off).
    fn coalesce_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(c) = &self.coalescer else { return Ok(None) };
        let s = c.stats();
        let d = PyDict::new(py);
        d.set_item("merged_batches", s.merged_batches)?;
        d.set_item("merged_calls", s.merged_calls)?;
        d.set_item("alone", s.alone)?;
        d.set_item("retried", s.retried)?;
        d.set_item("max_rows", s.max_rows)?;
        Ok(Some(d))
    }

    /// Stop the listener (idle connections close within one poll interval; a call in flight finishes).
    #[pyo3(signature = (timeout_s = 5.0))]
    fn stop(&self, py: Python<'_>, timeout_s: f64) {
        let t = Duration::from_secs_f64(timeout_s.max(0.0));
        py.allow_threads(|| self.server.stop(t));
    }
}

/// `None` if the CUDA driver loads and initialises, else why not.
#[pyfunction]
fn lane_cuda_driver_status() -> Option<String> {
    driver_status().err()
}

/// Lane wire format version.
#[pyfunction]
fn lane_protocol_version() -> u16 {
    pgw_lane::LANE_PROTOCOL_VERSION
}

/// Register this module's classes and functions in `pgw_fast`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLaneService>()?;
    m.add_class::<PyDeviceLock>()?;
    m.add_function(wrap_pyfunction!(lane_cuda_driver_status, m)?)?;
    m.add_function(wrap_pyfunction!(lane_protocol_version, m)?)?;
    Ok(())
}
