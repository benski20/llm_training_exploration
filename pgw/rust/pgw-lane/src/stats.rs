//! Lane counters, aggregated in Rust and drained by the Python service into its Prometheus
//! collectors (`pgw_service_requests_total`, `pgw_service_request_seconds`, `pgw_service_rows_total`,
//! `pgw_service_bytes_*_total`, `pgw_service_kernel_seconds`). Labels are tenant ids, linear kinds and
//! status names only; nothing about payloads.

use std::collections::HashMap;
use std::sync::Mutex;

/// Upper bounds of the request-latency histogram (seconds); same as the Python service's buckets.
pub const LATENCY_BOUNDS: [f64; 15] =
    [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

/// A latency histogram: per-bucket (non-cumulative) counts, the `+Inf` bucket last, and the sum.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Histogram {
    /// Counts per bucket (`LATENCY_BOUNDS.len() + 1` entries, non-cumulative).
    pub buckets: Vec<u64>,
    /// Sum of observations, seconds.
    pub sum: f64,
    /// Number of observations.
    pub count: u64,
}

impl Histogram {
    fn observe(&mut self, secs: f64) {
        if self.buckets.is_empty() {
            self.buckets = vec![0; LATENCY_BOUNDS.len() + 1];
        }
        let i = LATENCY_BOUNDS.iter().position(|&b| secs <= b).unwrap_or(LATENCY_BOUNDS.len());
        self.buckets[i] += 1;
        self.sum += secs;
        self.count += 1;
    }
}

/// Everything counted since the last drain.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// `(tenant, status name) -> latency histogram` (its count is the request count).
    pub requests: HashMap<(String, &'static str), Histogram>,
    /// `(tenant, kind) -> rows`.
    pub rows: HashMap<(String, String), u64>,
    /// `tenant -> (bytes in, bytes out)`.
    pub bytes: HashMap<String, (u64, u64)>,
    /// `(kind, backend) -> (calls, rows, compute seconds)` of executed calls.
    pub kernel: HashMap<(String, &'static str), (u64, u64, f64)>,
    /// Connections opened / refused at open since the last drain.
    pub connections: (u64, u64),
}

/// Thread-safe accumulator.
#[derive(Default)]
pub struct Stats {
    inner: Mutex<Snapshot>,
}

impl Stats {
    /// One finished call (any status).
    pub fn request(&self, tenant: &str, status: &'static str, secs: f64) {
        let mut s = self.inner.lock().unwrap();
        s.requests.entry((tenant.to_owned(), status)).or_default().observe(secs.max(0.0));
    }

    /// One computed call.
    #[allow(clippy::too_many_arguments)]
    pub fn computed(
        &self,
        tenant: &str,
        kind: &str,
        backend: &'static str,
        rows: u64,
        bytes_in: u64,
        bytes_out: u64,
        compute_secs: f64,
    ) {
        let mut s = self.inner.lock().unwrap();
        *s.rows.entry((tenant.to_owned(), kind.to_owned())).or_default() += rows;
        let b = s.bytes.entry(tenant.to_owned()).or_default();
        b.0 += bytes_in;
        b.1 += bytes_out;
        let k = s.kernel.entry((kind.to_owned(), backend)).or_default();
        k.0 += 1;
        k.1 += rows;
        k.2 += compute_secs;
    }

    /// A connection was opened (`ok`) or refused at open.
    pub fn connection(&self, ok: bool) {
        let mut s = self.inner.lock().unwrap();
        if ok {
            s.connections.0 += 1;
        } else {
            s.connections.1 += 1;
        }
    }

    /// Take everything counted so far.
    pub fn drain(&self) -> Snapshot {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_and_drain() {
        let st = Stats::default();
        st.request("t", "ok", 0.0004);
        st.request("t", "ok", 0.002);
        st.request("t", "ok", 99.0);
        st.computed("t", "qkv", "cuda_graph", 2, 10, 20, 0.001);
        st.computed("t", "qkv", "cuda_graph", 1, 5, 10, 0.001);
        let s = st.drain();
        let h = &s.requests[&("t".to_owned(), "ok")];
        assert_eq!(h.count, 3);
        assert_eq!(h.buckets[0], 1);
        assert_eq!(h.buckets[2], 1);
        assert_eq!(h.buckets[15], 1);
        assert_eq!(s.rows[&("t".to_owned(), "qkv".to_owned())], 3);
        assert_eq!(s.bytes["t"], (15, 30));
        assert_eq!(s.kernel[&("qkv".to_owned(), "cuda_graph")].0, 2);
        assert_eq!(st.drain(), Snapshot::default());
    }
}
