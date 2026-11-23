//! Persistent GPU worker executor (scaling plan, provider target 3; opt-in, default off).
//!
//! Instead of a graph launch per trip, one resident kernel per GPU (`pgw.service.persistent`, launched
//! once by the Python service) polls a small **control block** in mapped pinned host memory. A trip
//! is then, with no CUDA call and no Python:
//!
//! 1. wait for the device lock(s) of the call's group (the one-call-per-device invariant, shared with
//!    the graph and Python paths), in ascending device order;
//! 2. copy the packed body (`rows * n` little-endian u32 = the K = 32 `x_hat`) into the group's mapped
//!    input buffer: written **once** and read by every shard of a tensor-parallel group (broadcast);
//! 3. per shard worker: store the linear's table index and the row count, then the next sequence
//!    number into `doorbell` (release);
//! 4. spin until every worker's `done` equals that sequence number (acquire), then check `status`;
//! 5. copy the mapped output buffer (`rows * m` u32, row-major) into the reply. Each shard's kernel
//!    writes its own output columns `[col0, col0 + m_j)` of every row straight into that buffer, so
//!    the gather costs nothing beyond the reply copy and the bytes equal the single-GPU reply.
//!
//! Control block layout (little endian u32 words; the CUDA kernel and the Python emulator use the same
//! offsets, `pgw.service.persistent.CTRL_*`): `doorbell @0`, `linear @4`, `rows @8`, `quit @12`
//! (host to device; quit is the Python owner's), `done @128`, `status @132` (device to host, on their
//! own cache line). Block size [`CTRL_BYTES`].
//!
//! A worker that does not answer within the spin timeout, or reports a nonzero status, disables the
//! executor (every later call answers `fallback`, i.e. the graph or Python path serves it) and the
//! call is refused `internal` (the client resends it on the stream channel). Only K = 32 rings are
//! ever registered (4 bytes per element). Nothing the provider sees changes: same `x_hat` in, same
//! `y_hat` out.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::admission::Refusal;
use crate::exec::{micros, Call, DeviceGuard, DeviceLock, Done, Executor};
use crate::proto::Status;
use crate::sync::{MutexExt, RwLockExt};

/// Byte offsets in a control block.
pub const CTRL_DOORBELL: usize = 0;
/// Table index of the linear to run.
pub const CTRL_LINEAR: usize = 4;
/// Rows of the trip.
pub const CTRL_ROWS: usize = 8;
/// Nonzero: the kernel exits (written by the Python owner at shutdown).
pub const CTRL_QUIT: usize = 12;
/// Sequence number of the last finished trip (device to host).
pub const CTRL_DONE: usize = 128;
/// 0 = ok, else an error code of the last trip (device to host).
pub const CTRL_STATUS: usize = 132;
/// Size of one control block.
pub const CTRL_BYTES: usize = 256;

/// One shard worker (one resident kernel on one GPU).
#[derive(Clone, Debug)]
pub struct WorkerSpec {
    /// CUDA ordinal (device lock key).
    pub device: i32,
    /// Address of its control block (mapped pinned host memory, [`CTRL_BYTES`], 128-byte aligned).
    pub ctrl: usize,
}

/// A tensor-parallel group of workers sharing one input and one output buffer.
#[derive(Clone, Debug)]
pub struct GroupSpec {
    /// Shard workers (one per device; a single worker for one GPU).
    pub workers: Vec<WorkerSpec>,
    /// Mapped input buffer and its capacity in bytes.
    pub h_in: usize,
    /// Capacity of `h_in`.
    pub in_cap: usize,
    /// Mapped output buffer.
    pub h_out: usize,
    /// Capacity of `h_out`.
    pub out_cap: usize,
}

/// A linear served by a group: its index in every worker's table and its shape.
#[derive(Clone, Copy, Debug)]
pub struct LinearSlot {
    /// Group id.
    pub group: u32,
    /// Index in the workers' linear tables (the same in every shard of the group).
    pub table: u32,
    /// Input features.
    pub n: u32,
    /// Output features (whole linear).
    pub m: u32,
    /// Largest row count the kernel accepts.
    pub max_rows: u16,
}

struct Group {
    spec: GroupSpec,
    seq: Vec<AtomicU32>,
    locks: Vec<Arc<DeviceLock>>, // ascending device order
}

type Key = (u32, u32, u16); // (model, linear, ring)

/// The executor (see the module docs).
pub struct PersistentExecutor {
    groups: RwLock<HashMap<u32, Arc<Group>>>,
    linears: RwLock<HashMap<Key, LinearSlot>>,
    lock_source: Arc<dyn Fn(i32) -> Arc<DeviceLock> + Send + Sync>,
    enabled: AtomicBool,
    timeout_us: AtomicU64,
    spin_us: AtomicU64,
    disabled_reason: Mutex<Option<String>>,
}

fn word(base: usize, off: usize) -> &'static AtomicU32 {
    // SAFETY: callers only pass registered control blocks (`register_group` contract: live, aligned,
    // CTRL_BYTES long until `clear`), and AtomicU32 has the layout of u32.
    unsafe { &*((base + off) as *const AtomicU32) }
}

impl PersistentExecutor {
    /// An empty executor taking device locks from `lock_source` (the graph executor's, so all paths
    /// share one lock per device).
    pub fn new(lock_source: Arc<dyn Fn(i32) -> Arc<DeviceLock> + Send + Sync>) -> PersistentExecutor {
        PersistentExecutor {
            groups: RwLock::new(HashMap::new()),
            linears: RwLock::new(HashMap::new()),
            lock_source,
            enabled: AtomicBool::new(true),
            timeout_us: AtomicU64::new(2_000_000),
            spin_us: AtomicU64::new(200),
            disabled_reason: Mutex::new(None),
        }
    }

    /// How long a trip may wait for its workers before the executor disables itself.
    pub fn set_timeout_us(&self, us: u64) {
        self.timeout_us.store(us.max(1), Ordering::Relaxed);
    }

    /// Pure spin time before the wait starts yielding the core.
    pub fn set_spin_us(&self, us: u64) {
        self.spin_us.store(us, Ordering::Relaxed);
    }

    /// Register group `id`.
    ///
    /// # Safety
    /// Every `ctrl` must point to a live, 128-byte aligned, [`CTRL_BYTES`] control block, and
    /// `h_in[..in_cap]` / `h_out[..out_cap]` to live buffers, all valid until [`clear`](Self::clear)
    /// returns; the workers' kernels own the device-to-host words and nothing else writes the
    /// host-to-device words while a device lock of the group is not held.
    pub unsafe fn register_group(&self, id: u32, spec: GroupSpec) -> Result<(), String> {
        if spec.workers.is_empty() || spec.h_in == 0 || spec.h_out == 0 || spec.in_cap == 0 || spec.out_cap == 0 {
            return Err("empty group or null buffer".to_owned());
        }
        let mut devs: Vec<i32> = spec.workers.iter().map(|w| w.device).collect();
        for w in &spec.workers {
            if w.ctrl == 0 || w.ctrl % 128 != 0 {
                return Err("control blocks must be non-null and 128-byte aligned".to_owned());
            }
        }
        devs.sort_unstable();
        devs.dedup();
        let locks = devs.iter().map(|d| (self.lock_source)(*d)).collect();
        // continue each worker's sequence from its `done` word (a relaunched kernel keeps it)
        let seq =
            spec.workers.iter().map(|w| AtomicU32::new(word(w.ctrl, CTRL_DONE).load(Ordering::Acquire))).collect();
        self.groups.pwrite().insert(id, Arc::new(Group { spec, seq, locks }));
        Ok(())
    }

    /// Serve `(model, linear, ring)` with `slot` (its group must be registered).
    pub fn register_linear(&self, model: u32, linear: u32, ring: u16, slot: LinearSlot) -> Result<(), String> {
        let g = self.groups.pread().get(&slot.group).cloned().ok_or("unknown group")?;
        if slot.n == 0 || slot.m == 0 || slot.max_rows == 0 {
            return Err("empty linear".to_owned());
        }
        if slot.max_rows as usize * slot.n as usize * 4 > g.spec.in_cap
            || slot.max_rows as usize * slot.m as usize * 4 > g.spec.out_cap
        {
            return Err("buffers too small for this linear".to_owned());
        }
        self.linears.pwrite().insert((model, linear, ring), slot);
        Ok(())
    }

    /// Registered linears.
    pub fn len(&self) -> usize {
        self.linears.pread().len()
    }

    /// True if nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything (waits for in-flight trips: takes every device lock).
    pub fn clear(&self) {
        let groups: Vec<Arc<Group>> = self.groups.pread().values().cloned().collect();
        let mut locks: Vec<Arc<DeviceLock>> = groups.iter().flat_map(|g| g.locks.iter().cloned()).collect();
        locks.sort_by_key(|l| Arc::as_ptr(l) as usize);
        locks.dedup_by_key(|l| Arc::as_ptr(l) as usize);
        let guards: Vec<_> = locks.iter().map(|l| DeviceGuard::acquire(l, None)).collect();
        self.linears.pwrite().clear();
        self.groups.pwrite().clear();
        drop(guards);
    }

    /// Stop serving (every call answers `fallback`).
    pub fn disable(&self, reason: &str) {
        self.enabled.store(false, Ordering::SeqCst);
        *self.disabled_reason.plock() = Some(reason.to_owned());
    }

    /// `None` while enabled, else why not.
    pub fn disabled_reason(&self) -> Option<String> {
        if self.enabled.load(Ordering::SeqCst) {
            None
        } else {
            self.disabled_reason.plock().clone().or_else(|| Some("disabled".to_owned()))
        }
    }

    /// Run `(model, linear, ring)` on `body` (`rows` rows) into `out`; returns `(queue_us, compute_us)`.
    pub fn run_key(
        &self,
        key: Key,
        rows: u16,
        body: &[u8],
        out: &mut Vec<u8>,
        deadline: Option<Instant>,
    ) -> Result<(u32, u32), Refusal> {
        let received = Instant::now();
        if !self.enabled.load(Ordering::Relaxed) {
            return Err(Refusal::new(Status::Fallback));
        }
        let Some(slot) = self.linears.pread().get(&key).copied() else {
            return Err(Refusal::new(Status::Fallback));
        };
        let Some(g) = self.groups.pread().get(&slot.group).cloned() else {
            return Err(Refusal::new(Status::Fallback));
        };
        let in_len = rows as usize * slot.n as usize * 4;
        let out_len = rows as usize * slot.m as usize * 4;
        if rows == 0 || rows > slot.max_rows || body.len() != in_len {
            return Err(Refusal::new(Status::Fallback));
        }
        let mut guards = Vec::with_capacity(g.locks.len());
        for l in &g.locks {
            guards.push(DeviceGuard::acquire(l, deadline).ok_or(Refusal::new(Status::DeadlineExceeded))?);
        }
        if !self.enabled.load(Ordering::SeqCst) {
            return Err(Refusal::new(Status::Fallback));
        }
        let t_start = Instant::now();
        // SAFETY: h_in holds in_cap >= in_len bytes (register_linear check) and no trip of this group
        // runs (its device locks are held), so no kernel reads h_in now.
        unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), g.spec.h_in as *mut u8, in_len) };
        let mut want = Vec::with_capacity(g.spec.workers.len());
        for (w, s) in g.spec.workers.iter().zip(&g.seq) {
            let next = s.load(Ordering::Relaxed).wrapping_add(1);
            word(w.ctrl, CTRL_LINEAR).store(slot.table, Ordering::Relaxed);
            word(w.ctrl, CTRL_ROWS).store(rows as u32, Ordering::Relaxed);
            // release: the body and the descriptor are visible before the doorbell
            word(w.ctrl, CTRL_DOORBELL).store(next, Ordering::Release);
            want.push(next);
        }
        let spin = Duration::from_micros(self.spin_us.load(Ordering::Relaxed));
        let timeout = Duration::from_micros(self.timeout_us.load(Ordering::Relaxed));
        let mut status = 0u32;
        for ((w, s), next) in g.spec.workers.iter().zip(&g.seq).zip(&want) {
            let done = word(w.ctrl, CTRL_DONE);
            loop {
                if done.load(Ordering::Acquire) == *next {
                    break;
                }
                let el = t_start.elapsed();
                if el >= timeout {
                    self.disable(&format!(
                        "persistent worker on device {} did not answer within {:?}",
                        w.device, timeout
                    ));
                    return Err(Refusal::new(Status::Internal));
                }
                if el < spin {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
            s.store(*next, Ordering::Relaxed);
            status |= word(w.ctrl, CTRL_STATUS).load(Ordering::Acquire);
        }
        if status != 0 {
            self.disable(&format!("persistent worker reported status {status}"));
            return Err(Refusal::new(Status::Internal));
        }
        out.clear();
        out.reserve(out_len);
        // SAFETY: every worker finished (done == seq, acquire) and wrote its columns of the first
        // out_len bytes; out has the capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(g.spec.h_out as *const u8, out.as_mut_ptr(), out_len);
            out.set_len(out_len);
        }
        drop(guards);
        Ok((micros(received, t_start), micros(t_start, Instant::now())))
    }
}

impl Executor for PersistentExecutor {
    fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
        let (queue_us, compute_us) =
            self.run_key((call.model, call.linear, call.ring), call.rows, call.body, out, Some(call.deadline))?;
        if out.len() != call.out_len {
            out.clear();
            self.disable("persistent output length differs from the call's");
            return Err(Refusal::new(Status::Internal));
        }
        Ok(Done { queue_us, compute_us, backend: "lane_persistent" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Owned, 128-byte aligned zeroed memory (stands in for mapped pinned memory).
    struct Buf(Vec<u128>);

    impl Buf {
        fn new(bytes: usize) -> Buf {
            Buf(vec![0u128; bytes.div_ceil(16) + 8])
        }
        fn addr(&self) -> usize {
            (self.0.as_ptr() as usize).next_multiple_of(128)
        }
    }

    /// One emulated shard: weights for its rows of every linear (`table -> (n, m_total, col0, w)`).
    struct Shard {
        table: Vec<(usize, usize, usize, Vec<i8>)>,
    }

    /// The device side of the protocol in a thread: what the CUDA kernel does (same words, same
    /// order), with wrapping u32 arithmetic; `stall` stops answering, `bad` reports an error status.
    fn spawn_device(
        ctrl: usize,
        h_in: usize,
        h_out: usize,
        shard: Shard,
        stall: Arc<AtomicBool>,
        bad: Arc<AtomicBool>,
        trips: Arc<AtomicUsize>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut seen = word(ctrl, CTRL_DONE).load(Ordering::Acquire);
            loop {
                if word(ctrl, CTRL_QUIT).load(Ordering::Acquire) != 0 {
                    return;
                }
                let d = word(ctrl, CTRL_DOORBELL).load(Ordering::Acquire);
                if d == seen || stall.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                    continue;
                }
                seen = d;
                let li = word(ctrl, CTRL_LINEAR).load(Ordering::Relaxed) as usize;
                let rows = word(ctrl, CTRL_ROWS).load(Ordering::Relaxed) as usize;
                let (n, m_total, col0, w) = &shard.table[li];
                let m = w.len() / n;
                for r in 0..rows {
                    for i in 0..m {
                        let mut acc = 0u32;
                        for j in 0..*n {
                            // SAFETY: test buffers sized for max rows.
                            let x = unsafe { *((h_in + 4 * (r * n + j)) as *const u32) };
                            acc = acc.wrapping_add((w[i * n + j] as i32 as u32).wrapping_mul(x));
                        }
                        // SAFETY: as above.
                        unsafe { *((h_out + 4 * (r * m_total + col0 + i)) as *mut u32) = acc };
                    }
                }
                trips.fetch_add(1, Ordering::SeqCst);
                word(ctrl, CTRL_STATUS).store(bad.load(Ordering::Relaxed) as u32, Ordering::Release);
                word(ctrl, CTRL_DONE).store(seen, Ordering::Release);
            }
        })
    }

    fn weights(m: usize, n: usize, seed: u32) -> Vec<i8> {
        (0..m * n).map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 13) as i8).collect()
    }

    fn reference(w: &[i8], n: usize, rows: usize, body: &[u8]) -> Vec<u8> {
        let m = w.len() / n;
        let mut out = Vec::new();
        for r in 0..rows {
            for i in 0..m {
                let mut acc = 0u32;
                for j in 0..n {
                    let o = 4 * (r * n + j);
                    let x = u32::from_le_bytes(body[o..o + 4].try_into().unwrap());
                    acc = acc.wrapping_add((w[i * n + j] as i32 as u32).wrapping_mul(x));
                }
                out.extend_from_slice(&acc.to_le_bytes());
            }
        }
        out
    }

    struct Rig {
        ex: Arc<PersistentExecutor>,
        _bufs: Vec<Buf>,
        ctrls: Vec<usize>,
        threads: Vec<std::thread::JoinHandle<()>>,
        stall: Arc<AtomicBool>,
        bad: Arc<AtomicBool>,
        trips: Arc<AtomicUsize>,
        ws: Vec<Vec<i8>>,
    }

    impl Rig {
        /// `shards` workers over linears of shapes `dims` (m, n); max 4 rows.
        fn new(shards: usize, dims: &[(usize, usize)]) -> Rig {
            let locks: Arc<Mutex<HashMap<i32, Arc<DeviceLock>>>> = Arc::default();
            let ls = locks.clone();
            let ex = Arc::new(PersistentExecutor::new(Arc::new(move |d| ls.plock().entry(d).or_default().clone())));
            ex.set_timeout_us(300_000);
            let nmax = dims.iter().map(|d| d.1).max().unwrap();
            let mmax = dims.iter().map(|d| d.0).max().unwrap();
            let h_in = Buf::new(4 * 4 * nmax);
            let h_out = Buf::new(4 * 4 * mmax);
            let ws: Vec<Vec<i8>> = dims.iter().enumerate().map(|(i, (m, n))| weights(*m, *n, i as u32)).collect();
            let stall = Arc::new(AtomicBool::new(false));
            let bad = Arc::new(AtomicBool::new(false));
            let trips = Arc::new(AtomicUsize::new(0));
            let mut bufs = vec![];
            let mut workers = vec![];
            let mut ctrls = vec![];
            let mut threads = vec![];
            for s in 0..shards {
                let ctrl = Buf::new(CTRL_BYTES);
                let mut table = vec![];
                for ((m, n), w) in dims.iter().zip(&ws) {
                    let (m0, m1) = (m * s / shards, m * (s + 1) / shards);
                    table.push((*n, *m, m0, w[m0 * n..m1 * n].to_vec()));
                }
                threads.push(spawn_device(
                    ctrl.addr(),
                    h_in.addr(),
                    h_out.addr(),
                    Shard { table },
                    stall.clone(),
                    bad.clone(),
                    trips.clone(),
                ));
                workers.push(WorkerSpec { device: s as i32, ctrl: ctrl.addr() });
                ctrls.push(ctrl.addr());
                bufs.push(ctrl);
            }
            let spec = GroupSpec {
                workers,
                h_in: h_in.addr(),
                in_cap: 4 * 4 * nmax,
                h_out: h_out.addr(),
                out_cap: 4 * 4 * mmax,
            };
            // SAFETY: the buffers live in the rig until the threads are stopped.
            unsafe { ex.register_group(0, spec).unwrap() };
            for (i, (m, n)) in dims.iter().enumerate() {
                ex.register_linear(
                    0,
                    i as u32,
                    0,
                    LinearSlot { group: 0, table: i as u32, n: *n as u32, m: *m as u32, max_rows: 4 },
                )
                .unwrap();
            }
            bufs.push(h_in);
            bufs.push(h_out);
            Rig { ex, _bufs: bufs, ctrls, threads, stall, bad, trips, ws }
        }

        fn stop(mut self) {
            for c in &self.ctrls {
                word(*c, CTRL_QUIT).store(1, Ordering::Release);
            }
            for t in self.threads.drain(..) {
                t.join().unwrap();
            }
        }
    }

    fn body(seed: u32, len: usize) -> Vec<u8> {
        (0..len as u32)
            .flat_map(|i| (i.wrapping_mul(0x9E3779B9) ^ seed.wrapping_mul(0x85EBCA6B)).to_le_bytes())
            .collect()
    }

    #[test]
    fn tp_shards_gather_to_the_single_gpu_bytes() {
        let dims = [(37usize, 12usize), (8, 20)];
        let single = Rig::new(1, &dims);
        let tp = Rig::new(3, &dims);
        for (li, (_m, n)) in dims.iter().enumerate() {
            for rows in 1..=4u16 {
                let b = body(li as u32 * 10 + rows as u32, rows as usize * n);
                let want = reference(&single.ws[li], *n, rows as usize, &b);
                let (mut o1, mut o3) = (Vec::new(), Vec::new());
                single.ex.run_key((0, li as u32, 0), rows, &b, &mut o1, None).unwrap();
                tp.ex.run_key((0, li as u32, 0), rows, &b, &mut o3, None).unwrap();
                assert_eq!(o1, want);
                assert_eq!(o3, want);
            }
        }
        // unregistered keys and bad lengths fall back without touching the workers
        let before = tp.trips.load(Ordering::SeqCst);
        assert_eq!(tp.ex.run_key((0, 9, 0), 1, &[0; 48], &mut Vec::new(), None).unwrap_err().status, Status::Fallback);
        assert_eq!(tp.ex.run_key((0, 0, 0), 1, &[0; 4], &mut Vec::new(), None).unwrap_err().status, Status::Fallback);
        assert_eq!(tp.ex.run_key((0, 0, 0), 5, &[0; 240], &mut Vec::new(), None).unwrap_err().status, Status::Fallback);
        assert_eq!(tp.trips.load(Ordering::SeqCst), before);
        single.stop();
        tp.stop();
    }

    #[test]
    fn concurrent_trips_are_serialized_and_owned() {
        let dims = [(24usize, 16usize)];
        let rig = Rig::new(2, &dims);
        let ex = rig.ex.clone();
        let w = rig.ws[0].clone();
        let hs: Vec<_> = (0..8u32)
            .map(|t| {
                let ex = ex.clone();
                let w = w.clone();
                std::thread::spawn(move || {
                    for i in 0..20u32 {
                        let rows = 1 + ((t + i) % 4) as u16;
                        let b = body(t * 1000 + i, rows as usize * 16);
                        let mut out = Vec::new();
                        let now = Instant::now();
                        let call = Call {
                            tenant: "t",
                            model: 0,
                            linear: 0,
                            ring: 0,
                            rows,
                            body: &b,
                            out_len: rows as usize * 24 * 4,
                            received: now,
                            deadline: now + Duration::from_secs(10),
                        };
                        let d = ex.execute(&call, &mut out).unwrap();
                        assert_eq!(d.backend, "lane_persistent");
                        assert_eq!(out, reference(&w, 16, rows as usize, &b));
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(rig.trips.load(Ordering::SeqCst), 2 * 8 * 20);
        rig.stop();
    }

    #[test]
    fn stalled_or_failing_worker_disables_the_executor() {
        let rig = Rig::new(2, &[(8, 8)]);
        let b = body(1, 8);
        rig.ex.run_key((0, 0, 0), 1, &b, &mut Vec::new(), None).unwrap();
        rig.bad.store(true, Ordering::SeqCst);
        assert_eq!(rig.ex.run_key((0, 0, 0), 1, &b, &mut Vec::new(), None).unwrap_err().status, Status::Internal);
        assert!(rig.ex.disabled_reason().unwrap().contains("status"));
        assert_eq!(rig.ex.run_key((0, 0, 0), 1, &b, &mut Vec::new(), None).unwrap_err().status, Status::Fallback);
        rig.stop();

        let rig = Rig::new(1, &[(8, 8)]);
        rig.ex.set_timeout_us(20_000);
        rig.stall.store(true, Ordering::SeqCst);
        assert_eq!(rig.ex.run_key((0, 0, 0), 1, &b, &mut Vec::new(), None).unwrap_err().status, Status::Internal);
        assert!(rig.ex.disabled_reason().unwrap().contains("did not answer"));
        rig.stop();
    }

    #[test]
    fn registration_checks() {
        let locks: Arc<Mutex<HashMap<i32, Arc<DeviceLock>>>> = Arc::default();
        let ex = PersistentExecutor::new(Arc::new(move |d| locks.plock().entry(d).or_default().clone()));
        let c = Buf::new(CTRL_BYTES);
        let spec = |ctrl| GroupSpec {
            workers: vec![WorkerSpec { device: 0, ctrl }],
            h_in: 64,
            in_cap: 64,
            h_out: 64,
            out_cap: 64,
        };
        // SAFETY: refused before use (misaligned / null), or never run.
        unsafe {
            assert!(ex.register_group(0, spec(c.addr() + 4)).is_err());
            assert!(ex.register_group(0, spec(0)).is_err());
            ex.register_group(0, spec(c.addr())).unwrap();
        }
        assert!(ex.register_linear(0, 0, 0, LinearSlot { group: 1, table: 0, n: 1, m: 1, max_rows: 1 }).is_err());
        assert!(ex.register_linear(0, 0, 0, LinearSlot { group: 0, table: 0, n: 64, m: 1, max_rows: 1 }).is_err());
        ex.register_linear(0, 0, 0, LinearSlot { group: 0, table: 0, n: 4, m: 4, max_rows: 4 }).unwrap();
        assert_eq!(ex.len(), 1);
        ex.clear();
        assert!(ex.is_empty());
    }
}
