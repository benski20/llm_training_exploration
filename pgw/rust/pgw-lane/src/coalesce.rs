//! Opportunistic multi-tenant coalescing of lane calls (scaling plan, provider target 3; opt-in).
//!
//! Concurrent calls for the same `(model, linear, ring)` that are waiting while another call of that
//! key runs are merged into **one** call of the inner executor (one weight read for all of them):
//! their packed bodies are concatenated row-wise, the inner executor computes every row, and the
//! reply is cut back into each call's own rows. Every output row of `W8 x_hat mod 2^K` depends only
//! on its own input row and the arithmetic is exact, so each call's bytes are identical to running it
//! alone.
//!
//! * **No added latency** (adaptive window = 0): a call that finds its key idle runs at once, alone,
//!   on the caller's thread, with no copy. Only calls that arrive while their key is busy wait, and
//!   they would have waited for the device anyway; the next of them to wake takes every queued call
//!   of the key (up to `max_rows` rows, FIFO) as one batch. Nothing ever waits for more work.
//! * **Privacy**: rows of different tenants share a GEMM (every row is uniform noise to the
//!   provider, as in the Python batcher), but each reply holds exactly the rows of its own call; no
//!   call's bytes reach another connection. Admission and per-tenant accounting happen per call
//!   before the executor (unchanged).
//! * **Failure isolation**: if the merged call fails (any status), every member is retried alone on
//!   the inner executor, with its own deadline, so one member cannot make another fail. A member
//!   whose deadline passed before its batch started is refused `deadline_exceeded` alone.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Instant;

use crate::admission::Refusal;
use crate::exec::{micros, Call, Done, Executor};
use crate::proto::Status;
use crate::sync::MutexExt;

type Key = (u32, u32, u16); // (model, linear, ring)
type SlotResult = Result<(Vec<u8>, Done), Refusal>;

/// A waiting call. The body is borrowed from the waiting connection thread, which does not return
/// before the slot is either removed from the queue by itself or completed by a leader (see
/// [`Coalescer::execute`]), so the pointer stays valid while a leader may read it.
struct Slot {
    body: *const u8,
    body_len: usize,
    rows: u16,
    out_len: usize,
    received: Instant,
    deadline: Instant,
    result: Mutex<Option<SlotResult>>,
}

// SAFETY: `body` is only read by a leader while the owning thread is blocked waiting on the slot
// (the slot is taken out of the queue under the queue lock before the read, and the owner waits for
// its result once it is taken).
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

impl Slot {
    fn body(&self) -> &[u8] {
        // SAFETY: see the type's invariant.
        unsafe { std::slice::from_raw_parts(self.body, self.body_len) }
    }
}

#[derive(Default)]
struct KeyState {
    busy: bool,
    pending: VecDeque<Arc<Slot>>,
}

/// Counters (tests, `/readyz`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoalesceStats {
    /// Inner executions of more than one call.
    pub merged_batches: u64,
    /// Calls served by a merged execution.
    pub merged_calls: u64,
    /// Calls run alone (idle key, or a batch of one).
    pub alone: u64,
    /// Members retried alone after their merged execution failed.
    pub retried: u64,
    /// Largest merged row count.
    pub max_rows: u64,
}

/// Executor wrapper that merges concurrent same-key calls (see the module docs).
pub struct Coalescer {
    inner: Arc<dyn Executor>,
    max_rows: u16,
    keys: Mutex<HashMap<Key, KeyState>>,
    cv: Condvar,
    merged_batches: AtomicU64,
    merged_calls: AtomicU64,
    alone: AtomicU64,
    retried: AtomicU64,
    max_seen: AtomicU64,
}

impl Coalescer {
    /// Wrap `inner`; merged calls hold at most `max_rows` rows (calls above it run alone).
    pub fn new(inner: Arc<dyn Executor>, max_rows: u16) -> Coalescer {
        Coalescer {
            inner,
            max_rows: max_rows.max(1),
            keys: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
            merged_batches: AtomicU64::new(0),
            merged_calls: AtomicU64::new(0),
            alone: AtomicU64::new(0),
            retried: AtomicU64::new(0),
            max_seen: AtomicU64::new(0),
        }
    }

    /// Counters since start.
    pub fn stats(&self) -> CoalesceStats {
        CoalesceStats {
            merged_batches: self.merged_batches.load(Ordering::Relaxed),
            merged_calls: self.merged_calls.load(Ordering::Relaxed),
            alone: self.alone.load(Ordering::Relaxed),
            retried: self.retried.load(Ordering::Relaxed),
            max_rows: self.max_seen.load(Ordering::Relaxed),
        }
    }

    fn finish_key(&self, key: Key) {
        let mut keys = self.keys.plock();
        if let Some(st) = keys.get_mut(&key) {
            st.busy = false;
            if st.pending.is_empty() {
                keys.remove(&key);
            }
        }
        drop(keys);
        self.cv.notify_all();
    }

    /// Run `slots` (taken from the queue, in order) as one inner call; fill every slot's result.
    fn run_batch(&self, call: &Call<'_>, slots: &[Arc<Slot>]) {
        let now = Instant::now();
        let mut live: Vec<&Arc<Slot>> = Vec::with_capacity(slots.len());
        for s in slots {
            if now > s.deadline {
                *s.result.plock() = Some(Err(Refusal::new(Status::DeadlineExceeded)));
            } else {
                live.push(s);
            }
        }
        if live.is_empty() {
            return;
        }
        if live.len() == 1 {
            self.alone.fetch_add(1, Ordering::Relaxed);
            let r = self.run_alone(call, live[0]);
            *live[0].result.plock() = Some(r);
            return;
        }
        let rows: u16 = live.iter().map(|s| s.rows).sum();
        let mut body = Vec::with_capacity(live.iter().map(|s| s.body_len).sum());
        for s in &live {
            body.extend_from_slice(s.body());
        }
        let out_len: usize = live.iter().map(|s| s.out_len).sum();
        let merged = Call {
            tenant: call.tenant, // unused by the device executors; accounting stays per call
            model: call.model,
            linear: call.linear,
            ring: call.ring,
            rows,
            body: &body,
            out_len,
            received: live.iter().map(|s| s.received).min().unwrap_or(now),
            deadline: live.iter().map(|s| s.deadline).min().unwrap_or(now),
        };
        let mut out = Vec::with_capacity(out_len);
        let r = self.inner.execute(&merged, &mut out);
        match r {
            Ok(d) if out.len() == out_len => {
                self.merged_batches.fetch_add(1, Ordering::Relaxed);
                self.merged_calls.fetch_add(live.len() as u64, Ordering::Relaxed);
                self.max_seen.fetch_max(rows as u64, Ordering::Relaxed);
                let end = Instant::now();
                let mut off = 0;
                for s in &live {
                    let part = out[off..off + s.out_len].to_vec();
                    off += s.out_len;
                    let queue_us = micros(s.received, end).saturating_sub(d.compute_us);
                    let done = Done { queue_us, compute_us: d.compute_us, backend: d.backend };
                    *s.result.plock() = Some(Ok((part, done)));
                }
            }
            _ => {
                // isolation: every member alone, with its own deadline
                for s in &live {
                    self.retried.fetch_add(1, Ordering::Relaxed);
                    let r = self.run_alone(call, s);
                    *s.result.plock() = Some(r);
                }
            }
        }
    }

    fn run_alone(&self, call: &Call<'_>, s: &Slot) -> Result<(Vec<u8>, Done), Refusal> {
        let c = Call {
            tenant: call.tenant,
            model: call.model,
            linear: call.linear,
            ring: call.ring,
            rows: s.rows,
            body: s.body(),
            out_len: s.out_len,
            received: s.received,
            deadline: s.deadline,
        };
        let mut out = Vec::with_capacity(s.out_len);
        self.inner.execute(&c, &mut out).map(|d| (out, d))
    }
}

impl Executor for Coalescer {
    fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
        if call.rows > self.max_rows {
            return self.inner.execute(call, out);
        }
        let key: Key = (call.model, call.linear, call.ring);
        let mut keys = self.keys.plock();
        let st = keys.entry(key).or_default();
        if !st.busy && st.pending.is_empty() {
            // idle key: run now, alone, no copy (the lone-trip path pays one uncontended lock)
            st.busy = true;
            drop(keys);
            self.alone.fetch_add(1, Ordering::Relaxed);
            let r = self.inner.execute(call, out);
            self.finish_key(key);
            return r;
        }
        let me = Arc::new(Slot {
            body: call.body.as_ptr(),
            body_len: call.body.len(),
            rows: call.rows,
            out_len: call.out_len,
            received: call.received,
            deadline: call.deadline,
            result: Mutex::new(None),
        });
        st.pending.push_back(me.clone());
        loop {
            if let Some(r) = me.result.plock().take() {
                drop(keys);
                return r.map(|(bytes, d)| {
                    out.clear();
                    out.extend_from_slice(&bytes);
                    d
                });
            }
            let st = keys.entry(key).or_default();
            let queued = st.pending.iter().any(|s| Arc::ptr_eq(s, &me));
            if queued && !st.busy {
                // lead: take every queued call of the key, FIFO, up to max_rows rows
                st.busy = true;
                let mut batch: Vec<Arc<Slot>> = Vec::new();
                let mut rows: u16 = 0;
                while let Some(s) = st.pending.front() {
                    if !batch.is_empty() && rows + s.rows > self.max_rows {
                        break;
                    }
                    rows += s.rows;
                    batch.push(st.pending.pop_front().expect("front exists"));
                }
                drop(keys);
                self.run_batch(call, &batch);
                self.finish_key(key);
                keys = self.keys.plock();
                continue; // own result is set (the batch held this call)
            }
            if queued && Instant::now() > call.deadline {
                st.pending.retain(|s| !Arc::ptr_eq(s, &me));
                if st.pending.is_empty() && !st.busy {
                    keys.remove(&key);
                }
                return Err(Refusal::new(Status::DeadlineExceeded));
            }
            keys = if queued {
                let wait = call.deadline.saturating_duration_since(Instant::now());
                self.cv.wait_timeout(keys, wait).unwrap_or_else(PoisonError::into_inner).0
            } else {
                // taken by a leader: wait for the result (the leader always sets it)
                self.cv.wait(keys).unwrap_or_else(PoisonError::into_inner)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// y[r, j] = sum_i x[r, i] * (j + 1 + linear) mod 2^32 (u32 LE); counts calls and rows; optional
    /// failure for merged calls or for a poisoned input row.
    struct Fake {
        calls: AtomicUsize,
        max_rows_seen: AtomicUsize,
        sleep: Duration,
        fail_merged: bool,
    }

    impl Fake {
        fn new(sleep_ms: u64, fail_merged: bool) -> Fake {
            Fake {
                calls: AtomicUsize::new(0),
                max_rows_seen: AtomicUsize::new(0),
                sleep: Duration::from_millis(sleep_ms),
                fail_merged,
            }
        }
    }

    fn reference(linear: u32, rows: u16, body: &[u8], m: usize) -> Vec<u8> {
        let n = body.len() / 4 / rows as usize;
        let mut out = Vec::new();
        for r in 0..rows as usize {
            let x = &body[r * n * 4..(r + 1) * n * 4];
            for j in 0..m {
                let c = j as u32 + 1 + linear;
                let s = x
                    .chunks(4)
                    .fold(0u32, |a, b| a.wrapping_add(u32::from_le_bytes([b[0], b[1], b[2], b[3]]).wrapping_mul(c)));
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }

    impl Executor for Fake {
        fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.max_rows_seen.fetch_max(call.rows as usize, Ordering::SeqCst);
            std::thread::sleep(self.sleep);
            if self.fail_merged && call.rows > 1 {
                out.push(0xEE);
                return Err(Refusal::new(Status::Fallback));
            }
            if call.body.chunks(4).any(|c| c == [0xDE, 0xAD, 0xBE, 0xEF]) {
                return Err(Refusal::new(Status::Internal));
            }
            let m = call.out_len / 4 / call.rows as usize;
            out.extend_from_slice(&reference(call.linear, call.rows, call.body, m));
            Ok(Done { queue_us: 0, compute_us: 7, backend: "fake" })
        }
    }

    const N: usize = 6;
    const M: usize = 5;

    fn body(seed: u32, rows: u16) -> Vec<u8> {
        (0..rows as u32 * N as u32)
            .flat_map(|i| (seed.wrapping_mul(2654435761) ^ i.wrapping_mul(40503)).to_le_bytes())
            .collect()
    }

    fn run(c: &dyn Executor, linear: u32, rows: u16, b: &[u8], deadline: Duration) -> Result<Vec<u8>, Status> {
        let now = Instant::now();
        let call = Call {
            tenant: "t",
            model: 0,
            linear,
            ring: 0,
            rows,
            body: b,
            out_len: rows as usize * M * 4,
            received: now,
            deadline: now + deadline,
        };
        let mut out = Vec::new();
        c.execute(&call, &mut out).map(|_| out).map_err(|r| r.status)
    }

    #[test]
    fn lone_call_runs_alone_at_once() {
        let f = Arc::new(Fake::new(0, false));
        let c = Coalescer::new(f.clone(), 8);
        let b = body(1, 2);
        let t0 = Instant::now();
        assert_eq!(run(&c, 0, 2, &b, Duration::from_secs(1)).unwrap(), reference(0, 2, &b, M));
        assert!(t0.elapsed() < Duration::from_millis(50));
        assert_eq!(c.stats().alone, 1);
        assert_eq!(c.stats().merged_batches, 0);
        assert!(c.keys.plock().is_empty());
    }

    #[test]
    fn concurrent_calls_merge_and_each_gets_its_own_bytes() {
        let f = Arc::new(Fake::new(20, false));
        let c = Arc::new(Coalescer::new(f.clone(), 8));
        let mut hs = Vec::new();
        for t in 0..12u32 {
            let c = c.clone();
            hs.push(std::thread::spawn(move || {
                let rows = 1 + (t % 2) as u16;
                let linear = t % 2; // two keys
                let b = body(100 + t, rows);
                std::thread::sleep(Duration::from_millis((t % 3) as u64));
                let got = run(&*c, linear, rows, &b, Duration::from_secs(5)).unwrap();
                assert_eq!(got, reference(linear, rows, &b, M), "call {t}");
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        let s = c.stats();
        assert!(s.merged_batches >= 1, "{s:?}");
        assert!(s.max_rows <= 8);
        assert!(f.max_rows_seen.load(Ordering::SeqCst) <= 8);
        assert_eq!(s.merged_calls + s.alone, 12);
        assert!(f.calls.load(Ordering::SeqCst) < 12);
        assert!(c.keys.plock().is_empty());
    }

    #[test]
    fn merged_failure_retries_members_alone() {
        let f = Arc::new(Fake::new(15, true));
        let c = Arc::new(Coalescer::new(f.clone(), 8));
        let mut hs = Vec::new();
        for t in 0..6u32 {
            let c = c.clone();
            hs.push(std::thread::spawn(move || {
                let b = body(7 + t, 1);
                assert_eq!(run(&*c, 0, 1, &b, Duration::from_secs(5)).unwrap(), reference(0, 1, &b, M));
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        assert!(c.stats().retried >= 2, "{:?}", c.stats());
    }

    #[test]
    fn one_bad_member_does_not_fail_the_others() {
        let f = Arc::new(Fake::new(15, false));
        let c = Arc::new(Coalescer::new(f.clone(), 8));
        let mut hs = Vec::new();
        for t in 0..6u32 {
            let c = c.clone();
            hs.push(std::thread::spawn(move || {
                let mut b = body(30 + t, 1);
                if t == 3 {
                    b[0..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
                }
                let r = run(&*c, 0, 1, &b, Duration::from_secs(5));
                if t == 3 {
                    assert_eq!(r.unwrap_err(), Status::Internal);
                } else {
                    assert_eq!(r.unwrap(), reference(0, 1, &b, M));
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
    }

    #[test]
    fn queued_call_times_out_alone_and_big_calls_bypass() {
        let f = Arc::new(Fake::new(80, false));
        let c = Arc::new(Coalescer::new(f.clone(), 4));
        let c2 = c.clone();
        let h = std::thread::spawn(move || {
            let b = body(1, 1);
            run(&*c2, 0, 1, &b, Duration::from_secs(5)).unwrap()
        });
        std::thread::sleep(Duration::from_millis(10));
        let b = body(2, 1);
        assert_eq!(run(&*c, 0, 1, &b, Duration::from_millis(20)).unwrap_err(), Status::DeadlineExceeded);
        h.join().unwrap();
        // rows above max_rows go straight to the inner executor
        let b = body(3, 6);
        assert_eq!(run(&*c, 0, 6, &b, Duration::from_secs(1)).unwrap(), reference(0, 6, &b, M));
        assert!(c.keys.plock().is_empty());
    }
}
