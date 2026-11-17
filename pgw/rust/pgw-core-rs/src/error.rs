//! Error type with the stable PGW error codes (SPEC.md 6.6, `pgw.core.ErrorCode`).

use std::fmt;

/// Stable error codes shared with the Python core and the wire protocol.
///
/// Only the codes that this crate can raise are listed. [`ErrorCode::as_str`] returns the exact
/// string used by `pgw.core.ErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Invalid configuration or argument (`config`).
    Config,
    /// Tensor shape / length mismatch (`bad_shape`).
    BadShape,
    /// Value not representable in the ring (`ring_overflow`).
    RingOverflow,
    /// Freivalds or range check failed (`integrity_check_failed`).
    Integrity,
    /// Not enough mask slots (`mask_exhausted`).
    MaskExhausted,
    /// Attempted mask reuse or rollback detected (`mask_reuse`).
    MaskReuse,
    /// Mask store bookkeeping inconsistency (`mask_store`).
    MaskStore,
    /// Malformed payload (`protocol`).
    Protocol,
}

impl ErrorCode {
    /// The wire / Python string for this code.
    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Config => "config",
            ErrorCode::BadShape => "bad_shape",
            ErrorCode::RingOverflow => "ring_overflow",
            ErrorCode::Integrity => "integrity_check_failed",
            ErrorCode::MaskExhausted => "mask_exhausted",
            ErrorCode::MaskReuse => "mask_reuse",
            ErrorCode::MaskStore => "mask_store",
            ErrorCode::Protocol => "protocol",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error raised by this crate. Messages never contain key material, masks or activations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PgwError {
    code: ErrorCode,
    message: String,
}

impl PgwError {
    /// Create an error with a code and a human-readable message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    /// The stable error code.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// The message (safe to log).
    pub fn message(&self) -> &str {
        &self.message
    }

    pub(crate) fn config(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::Config, msg)
    }

    pub(crate) fn shape(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::BadShape, msg)
    }

    pub(crate) fn overflow(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::RingOverflow, msg)
    }
}

impl fmt::Display for PgwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for PgwError {}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, PgwError>;
