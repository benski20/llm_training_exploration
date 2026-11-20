//! # pgw-lane
//!
//! Service side of the PGW trip lane (chat-speed plan WP-A1; this crate was created as a stub by
//! WP-A0 and is owned by WP-A1: WP-A1 takes it over and overwrites these files rather than creating
//! them; the workspace member line in `rust/Cargo.toml` already exists). Planned scope:
//!
//! * a listener on a Unix domain socket (SO_PEERCRED checked) or TCP with TLS, opened by a ticket the
//!   regular stream channel issues (`lane_open`);
//! * a fixed binary header of public fields only (linear index, rows, ring, random request id,
//!   deadline from config, body length), with shape/size/deadline enforcement and refusal codes that
//!   send the caller back to the regular path;
//! * per-(linear, rows 1..8) CUDA graphs `[H2D, memset, GEMV, D2H]` over static pinned buffers,
//!   launched through a `dlopen`ed `cuGraphLaunch`, with a hybrid spin-then-block wait;
//! * admission, quota and metrics drained to the Python service; the one-call-per-device invariant
//!   shared with the Python path.
//!
//! The provider only ever sees one-time-padded `x_hat` and the resulting `y_hat`, exactly as on the
//! WebSocket path: the lane changes framing and latency, never content.
//!
//! Nothing is implemented yet, so nothing can enable the lane: the service setting `lane` is refused
//! for any value other than `"off"` until WP-A1 lands (`pgw.service.config.PENDING_SPEED_SETTINGS`).
#![forbid(unsafe_code)]

/// Version of the lane wire format this crate implements. `0` = no lane (stub).
pub const LANE_PROTOCOL_VERSION: u16 = 0;

#[cfg(test)]
mod tests {
    #[test]
    fn stub_has_no_protocol() {
        assert_eq!(super::LANE_PROTOCOL_VERSION, 0);
    }
}
