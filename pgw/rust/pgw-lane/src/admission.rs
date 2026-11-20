//! Admission shared by the lane and the Python stream channel.
//!
//! When a lane runs, the service's admission state lives here so both paths draw on one budget:
//! the draining flag, the global in-flight cap (`max_inflight_requests`), the per-tenant in-flight
//! cap and the per-tenant row-rate token bucket (`tenant_rows_per_s` / `tenant_burst_rows`). The
//! Python handler calls [`Admission::admit`] / [`Admission::exit`] instead of its own counters
//! (`pgw.service.handler.ServiceHandler`), and the lane calls the same functions per call.
//!
//! The token bucket has the Python `TokenBucket` semantics exactly: a request larger than the burst
//! is admitted once the bucket is full and then charged in full (the balance goes negative).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::proto::Status;
use crate::sync::MutexExt;

/// Admission limits (from the service settings).
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// Global in-flight cap across both paths.
    pub max_inflight: u64,
    /// Per-tenant in-flight cap (`None` = none).
    pub tenant_max_inflight: Option<u64>,
    /// Per-tenant rows per second (`None` = no rate limit).
    pub tenant_rows_per_s: Option<f64>,
    /// Bucket size (`None` = 2 s of rate, at least 1).
    pub tenant_burst_rows: Option<f64>,
    /// Retry hint for `Unavailable` / the concurrency cap, milliseconds.
    pub retry_after_ms: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_inflight: 256,
            tenant_max_inflight: None,
            tenant_rows_per_s: None,
            tenant_burst_rows: None,
            retry_after_ms: 1000,
        }
    }
}

/// A refusal: status and retry hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// Wire status.
    pub status: Status,
    /// Retry hint, milliseconds.
    pub retry_after_ms: u32,
}

impl Refusal {
    /// A refusal without a retry hint.
    pub fn new(status: Status) -> Refusal {
        Refusal { status, retry_after_ms: 0 }
    }
}

/// Classic token bucket over a caller-supplied clock (seconds), Python `TokenBucket` semantics.
#[derive(Clone, Debug)]
pub struct TokenBucket {
    rate: f64,
    burst: f64,
    tokens: f64,
    t: f64,
}

impl TokenBucket {
    /// A full bucket at time `now`.
    pub fn new(rate: f64, burst: f64, now: f64) -> TokenBucket {
        TokenBucket { rate, burst, tokens: burst, t: now }
    }

    /// Charge `n` tokens at `now`: 0 on success, else seconds to wait.
    pub fn take(&mut self, n: f64, now: f64) -> f64 {
        self.tokens = self.burst.min(self.tokens + (now - self.t) * self.rate);
        self.t = now;
        let need = n.min(self.burst);
        if self.tokens >= need {
            self.tokens -= n;
            return 0.0;
        }
        (need - self.tokens) / self.rate
    }
}

#[derive(Default)]
struct Tenant {
    inflight: u64,
    bucket: Option<TokenBucket>,
}

/// Shared admission state (see the module docs).
pub struct Admission {
    limits: Mutex<Limits>,
    draining: AtomicBool,
    inflight: AtomicU64,
    tenants: Mutex<HashMap<String, Tenant>>,
    epoch: Instant,
}

impl Admission {
    /// New state with `limits`.
    pub fn new(limits: Limits) -> Admission {
        Admission {
            limits: Mutex::new(limits),
            draining: AtomicBool::new(false),
            inflight: AtomicU64::new(0),
            tenants: Mutex::new(HashMap::new()),
            epoch: Instant::now(),
        }
    }

    /// Replace the limits (buckets are rebuilt lazily with the new rate).
    pub fn set_limits(&self, limits: Limits) {
        *self.limits.plock() = limits;
        for t in self.tenants.plock().values_mut() {
            t.bucket = None;
        }
    }

    /// Current limits.
    pub fn limits(&self) -> Limits {
        self.limits.plock().clone()
    }

    /// Stop admitting (shutdown).
    pub fn set_draining(&self, on: bool) {
        self.draining.store(on, Ordering::SeqCst);
    }

    /// True once draining.
    pub fn draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// In-flight requests across both paths.
    pub fn inflight(&self) -> u64 {
        self.inflight.load(Ordering::SeqCst)
    }

    /// In-flight requests of one tenant.
    pub fn tenant_inflight(&self, tenant: &str) -> u64 {
        self.tenants.plock().get(tenant).map(|t| t.inflight).unwrap_or(0)
    }

    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    /// Admit `rows` rows for `tenant`: draining, global cap, tenant cap, then the row bucket (in
    /// that order, like the Python handler). With `enter`, the request is counted in flight on
    /// success and the caller must call [`Admission::exit`] once it finishes.
    pub fn admit(&self, tenant: &str, rows: u64, enter: bool) -> Result<(), Refusal> {
        let lim = self.limits.plock().clone();
        let busy = Refusal { status: Status::Unavailable, retry_after_ms: lim.retry_after_ms };
        if self.draining() {
            return Err(busy);
        }
        let mut tenants = self.tenants.plock();
        if self.inflight.load(Ordering::SeqCst) >= lim.max_inflight {
            return Err(busy);
        }
        let now = self.now();
        let st = tenants.entry(tenant.to_owned()).or_default();
        if let Some(cap) = lim.tenant_max_inflight {
            if st.inflight >= cap {
                return Err(Refusal { status: Status::RateLimited, retry_after_ms: lim.retry_after_ms });
            }
        }
        if let Some(rate) = lim.tenant_rows_per_s {
            let burst = lim.tenant_burst_rows.unwrap_or_else(|| (2.0 * rate).ceil().max(1.0));
            let b = st.bucket.get_or_insert_with(|| TokenBucket::new(rate, burst, now));
            let wait = b.take(rows as f64, now);
            if wait > 0.0 {
                let ms = (wait * 1000.0).ceil().min(u32::MAX as f64) as u32;
                return Err(Refusal { status: Status::RateLimited, retry_after_ms: ms.max(1) });
            }
        }
        if enter {
            st.inflight += 1;
            self.inflight.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    /// A request admitted with `enter` finished.
    pub fn exit(&self, tenant: &str) {
        let mut tenants = self.tenants.plock();
        if let Some(st) = tenants.get_mut(tenant) {
            if st.inflight > 0 {
                st.inflight -= 1;
                self.inflight.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_matches_python_semantics() {
        let mut b = TokenBucket::new(10.0, 20.0, 0.0);
        assert_eq!(b.take(15.0, 0.0), 0.0); // 5 left
        assert!((b.take(10.0, 0.0) - 0.5).abs() < 1e-12); // need 10, have 5
        assert_eq!(b.take(10.0, 0.5), 0.0); // refilled to 10 -> 0
                                            // oversized request: admitted once full, then debt
        let mut b = TokenBucket::new(10.0, 20.0, 0.0);
        assert_eq!(b.take(50.0, 0.0), 0.0); // balance -30
        assert!((b.take(1.0, 0.0) - 3.1).abs() < 1e-9);
    }

    #[test]
    fn caps_and_drain() {
        let a = Admission::new(Limits { max_inflight: 2, tenant_max_inflight: Some(1), ..Limits::default() });
        a.admit("t1", 1, true).unwrap();
        assert_eq!(a.admit("t1", 1, true).unwrap_err().status, Status::RateLimited);
        a.admit("t2", 1, true).unwrap();
        assert_eq!(a.admit("t3", 1, true).unwrap_err().status, Status::Unavailable);
        assert_eq!(a.inflight(), 2);
        a.exit("t1");
        a.exit("t1"); // extra exit is ignored
        assert_eq!(a.inflight(), 1);
        a.admit("t3", 1, false).unwrap();
        assert_eq!(a.inflight(), 1);
        a.set_draining(true);
        let r = a.admit("t3", 1, true).unwrap_err();
        assert_eq!((r.status, r.retry_after_ms), (Status::Unavailable, 1000));
    }

    #[test]
    fn row_rate_refusal_has_retry_hint() {
        let a = Admission::new(Limits {
            tenant_rows_per_s: Some(100.0),
            tenant_burst_rows: Some(10.0),
            ..Limits::default()
        });
        a.admit("t", 10, false).unwrap();
        let r = a.admit("t", 10, false).unwrap_err();
        assert_eq!(r.status, Status::RateLimited);
        assert!(r.retry_after_ms >= 1 && r.retry_after_ms <= 100);
        a.admit("u", 10, false).unwrap(); // other tenants are independent
    }
}
