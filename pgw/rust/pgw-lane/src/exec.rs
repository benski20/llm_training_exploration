//! What the lane serves (model, linear and ring tables) and how a call is executed.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::admission::Refusal;
use crate::proto::Status;

/// One linear of a served model, in the order the ticket publishes (`lane_open` reply).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearEntry {
    /// Input features.
    pub n: u32,
    /// Output features.
    pub m: u32,
    /// Kind (`qkv`, `o`, `gate_up`, `down`, ...): metrics label.
    pub kind: String,
}

/// A ring the service serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingEntry {
    /// Name (`K32A12C27`).
    pub name: String,
    /// Packed bytes per element (`K / 8`).
    pub bytes_per_element: u32,
}

/// A model reachable through tickets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEntry {
    /// Linears in table order.
    pub linears: Vec<LinearEntry>,
    /// Rings in table order (the header's ring index points here).
    pub rings: Vec<RingEntry>,
}

/// One admitted call, validated against the tables.
pub struct Call<'a> {
    /// Tenant (from the ticket).
    pub tenant: &'a str,
    /// Model slot (from the ticket).
    pub model: u32,
    /// Linear index.
    pub linear: u32,
    /// Ring index (in the model's ring table).
    pub ring: u16,
    /// Rows.
    pub rows: u16,
    /// Packed `x_hat` (`rows * n * bytes_per_element` bytes).
    pub body: &'a [u8],
    /// Expected reply length (`rows * m * bytes_per_element`).
    pub out_len: usize,
    /// When the header arrived.
    pub received: Instant,
    /// Latest time compute may start.
    pub deadline: Instant,
}

/// Timing of an executed call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Done {
    /// Receipt to compute start (device lock wait), microseconds.
    pub queue_us: u32,
    /// Compute, microseconds.
    pub compute_us: u32,
    /// Backend label for metrics.
    pub backend: &'static str,
}

/// Executes calls. `out` is empty on entry and must hold exactly `call.out_len` bytes on success.
pub trait Executor: Send + Sync {
    /// Run one call.
    fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal>;
}

/// Saturating microseconds between two instants.
pub fn micros(from: Instant, to: Instant) -> u32 {
    to.saturating_duration_since(from).as_micros().min(u32::MAX as u128) as u32
}

/// Serves a call with the first executor that accepts it: `primary` may answer
/// [`Status::Fallback`] to pass the call to `secondary`; without an executor the call is refused
/// with [`Status::Fallback`] (the client resends it on the stream channel).
pub struct Chain {
    /// Tried first (e.g. CUDA-graph replay).
    pub primary: Option<Arc<dyn Executor>>,
    /// Tried when the primary has nothing for the call (e.g. the Python kernel path).
    pub secondary: Option<Arc<dyn Executor>>,
}

impl Executor for Chain {
    fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
        if let Some(p) = &self.primary {
            match p.execute(call, out) {
                Err(r) if r.status == Status::Fallback => out.clear(),
                other => return other,
            }
        }
        match &self.secondary {
            Some(s) => s.execute(call, out),
            None => Err(Refusal::new(Status::Fallback)),
        }
    }
}

/// A mutex that can be shared with Python (`DeviceExecutor.lock`) and held across calls: at most
/// one call per device runs at a time, on either path (the one-call-per-device invariant).
#[derive(Default)]
pub struct DeviceLock {
    held: Mutex<bool>,
    cv: Condvar,
}

impl DeviceLock {
    /// Acquire, waiting until `deadline` (`None` = forever). False on timeout.
    pub fn acquire(&self, deadline: Option<Instant>) -> bool {
        let mut held = self.held.lock().unwrap();
        while *held {
            match deadline {
                None => held = self.cv.wait(held).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return false;
                    }
                    held = self.cv.wait_timeout(held, d - now).unwrap().0;
                }
            }
        }
        *held = true;
        true
    }

    /// Acquire without waiting.
    pub fn try_acquire(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        if *held {
            return false;
        }
        *held = true;
        true
    }

    /// Release. Returns false if it was not held.
    pub fn release(&self) -> bool {
        let mut held = self.held.lock().unwrap();
        let was = *held;
        *held = false;
        drop(held);
        self.cv.notify_one();
        was
    }

    /// True while held.
    pub fn locked(&self) -> bool {
        *self.held.lock().unwrap()
    }
}

/// RAII holder of a [`DeviceLock`].
pub struct DeviceGuard<'a>(&'a DeviceLock);

impl<'a> DeviceGuard<'a> {
    /// Acquire `lock` until `deadline`; `None` on timeout.
    pub fn acquire(lock: &'a DeviceLock, deadline: Option<Instant>) -> Option<DeviceGuard<'a>> {
        lock.acquire(deadline).then_some(DeviceGuard(lock))
    }
}

impl Drop for DeviceGuard<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct Fixed(Result<u8, Status>);

    impl Executor for Fixed {
        fn execute(&self, call: &Call<'_>, out: &mut Vec<u8>) -> Result<Done, Refusal> {
            match self.0 {
                Ok(v) => {
                    out.resize(call.out_len, v);
                    Ok(Done { queue_us: 0, compute_us: 0, backend: "fixed" })
                }
                Err(s) => {
                    out.push(0xEE); // partial output must be discarded by the chain
                    Err(Refusal::new(s))
                }
            }
        }
    }

    fn call(body: &[u8]) -> Call<'_> {
        let now = Instant::now();
        Call {
            tenant: "t",
            model: 0,
            linear: 0,
            ring: 0,
            rows: 1,
            body,
            out_len: 4,
            received: now,
            deadline: now + Duration::from_secs(1),
        }
    }

    #[test]
    fn chain_order() {
        let body = [0u8; 4];
        let c = call(&body);
        let mut out = Vec::new();
        let ch =
            Chain { primary: Some(Arc::new(Fixed(Err(Status::Fallback)))), secondary: Some(Arc::new(Fixed(Ok(3)))) };
        assert_eq!(ch.execute(&c, &mut out).unwrap().backend, "fixed");
        assert_eq!(out, vec![3; 4]);
        let ch = Chain {
            primary: Some(Arc::new(Fixed(Err(Status::DeadlineExceeded)))),
            secondary: Some(Arc::new(Fixed(Ok(3)))),
        };
        assert_eq!(ch.execute(&c, &mut Vec::new()).unwrap_err().status, Status::DeadlineExceeded);
        let ch = Chain { primary: None, secondary: None };
        assert_eq!(ch.execute(&c, &mut Vec::new()).unwrap_err().status, Status::Fallback);
    }

    #[test]
    fn device_lock_excludes_and_times_out() {
        let l = Arc::new(DeviceLock::default());
        assert!(l.acquire(None));
        assert!(!l.try_acquire());
        assert!(!l.acquire(Some(Instant::now() + Duration::from_millis(20))));
        let l2 = l.clone();
        let h = std::thread::spawn(move || l2.acquire(Some(Instant::now() + Duration::from_secs(5))));
        std::thread::sleep(Duration::from_millis(20));
        assert!(l.release());
        assert!(h.join().unwrap());
        assert!(l.locked());
        {
            assert!(DeviceGuard::acquire(&l, Some(Instant::now())).is_none());
        }
        l.release();
        {
            let _g = DeviceGuard::acquire(&l, None).unwrap();
            assert!(l.locked());
        }
        assert!(!l.locked());
    }
}
