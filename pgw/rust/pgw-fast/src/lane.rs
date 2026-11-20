//! Trip lane client (chat-speed plan WP-A3; stub created by WP-A0, owned by WP-A3).
//!
//! Will hold `TripChannel`: one masked round trip per call with the GIL released (`writev` of the
//! binary header and the `x_hat` body, hybrid spin-then-block wait, reply read into a caller-owned
//! buffer), the `lane_open` ticket, and TLS on the TCP lane. The wire format is the service lane of
//! `rust/pgw-lane` (WP-A1).
//!
//! Until WP-A3 lands this module registers **no** Python functions, so `pgw_fast` exposes no lane
//! API and Python keeps using the WebSocket stream channel (`pgw.wire.client`).

use pyo3::prelude::*;

/// Register this module's functions in `pgw_fast` (none yet).
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
