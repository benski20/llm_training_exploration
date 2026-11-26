//! Lane wire format, version 1 (all integers little-endian).
//!
//! Every client frame is a fixed 48-byte [`RequestHeader`] followed by `body_len` bytes; every
//! service frame is a fixed 40-byte [`ReplyHeader`] followed by `body_len` bytes. The header
//! carries **public fields only**: the linear's index in the table the ticket published, the row
//! count, the ring's index, a random request id, a deadline taken from configuration and the body
//! length. The body is the same packed `x_hat` / `y_hat` bytes the WebSocket stream channel carries
//! (`pgw.core.pack_ring`), so the provider learns nothing it did not see before.
//!
//! ```text
//! request (48 B)                                reply (40 B)
//!  0  [u8;4]  magic  b"PGWL"                     0  [u8;4]  magic  b"PGWR"
//!  4  u16     version (1)                        4  u16     version (1)
//!  6  u16     kind (1 open, 2 call, 3 shm)       6  u16     status (Status)
//!  8  [u8;16] request id (random; open: 0)       8  [u8;16] request id (echo)
//! 24  u32     linear index                      24  u32     queue_us
//! 28  u16     rows (1..=max_rows)               28  u32     compute_us
//! 30  u16     ring index                        32  u32     retry_after_ms
//! 32  u32     deadline_ms (0 = service cap)     36  u32     body_len
//! 36  u32     body_len
//! 40  u64     reserved (0)
//! ```
//!
//! An `open` frame (kind 1) must be the first frame of a connection: its body is the 32-byte ticket
//! the service issued through `POST /pgw/v1/lane/open`; linear, rows, ring and deadline are 0. A
//! `call` frame (kind 2) carries `rows * n(linear) * bytes_per_element(ring)` bytes of `x_hat`
//! exactly; an OK reply carries `rows * m(linear) * bytes_per_element(ring)` bytes of `y_hat`.
//!
//! **Version 2** adds one frame kind and changes nothing else: a `multi` frame (kind 4, header
//! version 2) carries the rows of several answers of **one** batched trip, the same items the
//! WebSocket stream channel's compute request carries (`ComputeRequest.items`, in order). Its
//! header's `linear` field holds the item count (1..=[`MAX_ITEMS`]) and `rows` the total row count
//! (1..=[`MAX_MULTI_ROWS`]); the body is an item table followed by the items' `x_hat`, in order. An
//! item may have more rows than one `call` frame serves (a prompt's rows): the service runs them in
//! calls its executors serve (rows are independent) and answers with the same bytes:
//!
//! ```text
//! multi body: count x { u32 linear index, u16 rows (>= 1), u16 reserved (0) }  (8 B each)
//!             then x_hat of item 0, item 1, ...   (rows_i * n(linear_i) * bpe bytes each)
//! OK reply:   the 40-byte reply header (version 1, unchanged) and y_hat of item 0, item 1, ...
//! ```
//!
//! A service offers version 2 in its ticket (`lane/open`: `"versions": [1, 2]`, `"max_items"`);
//! a client sends a `multi` frame only to a service that offered it, so a version-1 service never
//! sees one (it would answer [`Status::Version`], fatal). Version-1 frames are unchanged and served
//! as before. A `multi` frame is all or nothing: one status for the whole frame, and a refusal
//! means "send the whole request on the stream channel" as for a single call. The connection's
//! tenant (its ticket) owns every item: a frame never mixes tenants.
//!
//! Statuses below [`FATAL_MIN`] leave the connection usable; statuses at or above it are followed
//! by the service closing the connection. [`Status::is_fallback`] statuses mean "send the
//! byte-identical body on the regular stream channel under the same request id".

/// Request magic (`b"PGWL"`).
pub const REQUEST_MAGIC: [u8; 4] = *b"PGWL";
/// Reply magic (`b"PGWR"`).
pub const REPLY_MAGIC: [u8; 4] = *b"PGWR";
/// Wire format version of the `open`, `call` and `shm` frames and of every reply.
pub const VERSION: u16 = 1;
/// Wire format version of the `multi` frame (the highest version this crate implements).
pub const VERSION_MULTI: u16 = 2;
/// Most items in one `multi` frame.
pub const MAX_ITEMS: u16 = 64;
/// Most rows (all items) in one `multi` frame.
pub const MAX_MULTI_ROWS: u16 = 256;
/// Size of one entry of a `multi` frame's item table.
pub const ITEM_LEN: usize = 8;
/// Size of a [`RequestHeader`] on the wire.
pub const REQUEST_HEADER_LEN: usize = 48;
/// Size of a [`ReplyHeader`] on the wire.
pub const REPLY_HEADER_LEN: usize = 40;
/// Length of a lane ticket (the body of an `open` frame).
pub const TICKET_LEN: usize = 32;
/// Largest rows per call any lane serves (graphs exist for rows 1..=8).
pub const MAX_ROWS: u16 = 8;
/// First status code after which the service closes the connection.
pub const FATAL_MIN: u16 = 16;

/// Frame kind of a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// First frame: present the ticket.
    Open,
    /// One masked product.
    Call,
    /// Opt-in (Unix domain socket only, after a successful open): move this connection's calls to a
    /// shared-memory segment the service creates and passes back with `SCM_RIGHTS` ([`crate::shm`]).
    /// All fields 0, no body. A service without it answers [`Status::Fallback`] (non-fatal) and the
    /// connection stays on the socket.
    Shm,
    /// Version 2: the items of one batched trip (see the module docs). `linear` = item count, `rows`
    /// = total rows, body = item table + the items' `x_hat`.
    Multi,
}

impl Kind {
    fn from_u16(v: u16) -> Option<Kind> {
        match v {
            1 => Some(Kind::Open),
            2 => Some(Kind::Call),
            3 => Some(Kind::Shm),
            4 => Some(Kind::Multi),
            _ => None,
        }
    }

    fn as_u16(self) -> u16 {
        match self {
            Kind::Open => 1,
            Kind::Call => 2,
            Kind::Shm => 3,
            Kind::Multi => 4,
        }
    }

    /// The header version a frame of this kind carries.
    pub fn version(self) -> u16 {
        match self {
            Kind::Multi => VERSION_MULTI,
            _ => VERSION,
        }
    }
}

/// One entry of a `multi` frame's item table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    /// Linear index in the ticket's table.
    pub linear: u32,
    /// Rows of this item (>= 1; all items at most MAX_MULTI_ROWS).
    pub rows: u16,
}

/// Encode an item table (clients and tests).
pub fn encode_items(items: &[Item], out: &mut Vec<u8>) {
    for it in items {
        out.extend_from_slice(&it.linear.to_le_bytes());
        out.extend_from_slice(&it.rows.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
    }
}

/// Parse the item table of a `multi` frame whose header named `count` items and `total` rows: every
/// entry with at least one row and a zero reserved field, the rows summing to `total`. Table indices
/// are checked by the server (it knows the tables). `Err(BadHeader)` otherwise.
pub fn parse_items(b: &[u8], count: usize, total: u16) -> Result<Vec<Item>, Status> {
    if b.len() != count * ITEM_LEN || count == 0 || count > MAX_ITEMS as usize {
        return Err(Status::BadHeader);
    }
    let mut items = Vec::with_capacity(count);
    let mut sum: u32 = 0;
    for e in b.chunks_exact(ITEM_LEN) {
        let it = Item { linear: u32_at(e, 0), rows: u16_at(e, 4) };
        if it.rows == 0 || it.rows > MAX_MULTI_ROWS || u16_at(e, 6) != 0 {
            return Err(Status::BadHeader);
        }
        sum += u32::from(it.rows);
        items.push(it);
    }
    if sum != u32::from(total) {
        return Err(Status::BadHeader);
    }
    Ok(items)
}

/// Reply status codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Status {
    /// Success; the body is `y_hat`.
    Ok = 0,
    /// Not served on the lane (rows above the lane's maximum, a ring or linear without a lane
    /// executor): resend on the stream channel.
    Fallback = 1,
    /// Draining or overloaded: resend on the stream channel (which applies the same admission).
    Unavailable = 2,
    /// Tenant quota (shared with the stream channel); `retry_after_ms` is set.
    RateLimited = 3,
    /// The call's deadline passed before it could run.
    DeadlineExceeded = 4,
    /// The executor failed: resend on the stream channel.
    Internal = 5,
    /// Malformed header (magic, kind, reserved bits, field ranges). Fatal.
    BadHeader = 16,
    /// `body_len` does not match the shape the header names. Fatal.
    BadLength = 17,
    /// Linear or ring index outside the ticket's tables. Fatal.
    UnknownLinear = 18,
    /// Unknown, expired or already used ticket, or the tenant was revoked. Fatal.
    Unauthenticated = 19,
    /// The connecting process is not an allowed peer (SO_PEERCRED). Fatal.
    PeerRefused = 20,
    /// Unsupported wire format version. Fatal.
    Version = 21,
    /// A call before a successful open, or a second open. Fatal.
    NotOpen = 22,
    /// The service has `max_connections` lane connections open: it closes this one before reading
    /// anything. Fatal; resend the byte-identical body on the stream channel (which applies the
    /// same admission) and retry the lane after `retry_after_ms`.
    Busy = 23,
}

impl Status {
    /// Every status, in code order (tests, metrics tables).
    pub const ALL: [Status; 14] = [
        Status::Ok,
        Status::Fallback,
        Status::Unavailable,
        Status::RateLimited,
        Status::DeadlineExceeded,
        Status::Internal,
        Status::BadHeader,
        Status::BadLength,
        Status::UnknownLinear,
        Status::Unauthenticated,
        Status::PeerRefused,
        Status::Version,
        Status::NotOpen,
        Status::Busy,
    ];

    /// Numeric code on the wire.
    pub fn code(self) -> u16 {
        self as u16
    }

    /// Status for a wire code (`None` for an unknown code).
    pub fn from_code(code: u16) -> Option<Status> {
        Status::ALL.iter().copied().find(|s| s.code() == code)
    }

    /// The service closes the connection after this status.
    pub fn is_fatal(self) -> bool {
        self.code() >= FATAL_MIN
    }

    /// The client should resend the byte-identical body on the stream channel.
    pub fn is_fallback(self) -> bool {
        matches!(self, Status::Fallback | Status::Unavailable | Status::Internal | Status::Busy)
    }

    /// Short snake-case name (metrics label, Python mirror).
    pub fn name(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Fallback => "fallback",
            Status::Unavailable => "unavailable",
            Status::RateLimited => "rate_limited",
            Status::DeadlineExceeded => "deadline_exceeded",
            Status::Internal => "internal",
            Status::BadHeader => "bad_header",
            Status::BadLength => "bad_length",
            Status::UnknownLinear => "unknown_linear",
            Status::Unauthenticated => "unauthenticated",
            Status::PeerRefused => "peer_refused",
            Status::Version => "version",
            Status::NotOpen => "not_open",
            Status::Busy => "busy",
        }
    }
}

/// A parsed, structurally valid request header (field ranges against the ticket's tables are
/// checked by the server, which knows them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestHeader {
    /// Frame kind.
    pub kind: Kind,
    /// Random request id chosen by the client (echoed).
    pub request_id: [u8; 16],
    /// Linear index in the ticket's table.
    pub linear: u32,
    /// Rows (1..=MAX_ROWS for a call, 0 for open).
    pub rows: u16,
    /// Ring index in the ticket's table.
    pub ring: u16,
    /// Deadline in milliseconds from receipt (0 = the service's cap).
    pub deadline_ms: u32,
    /// Body length in bytes.
    pub body_len: u32,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

impl RequestHeader {
    /// Parse 48 header bytes. Returns the status to refuse with on a malformed header: the
    /// request id is echoed when the magic was right (`(status, id)`).
    pub fn parse(b: &[u8; REQUEST_HEADER_LEN]) -> Result<RequestHeader, (Status, [u8; 16])> {
        let mut id = [0u8; 16];
        if b[0..4] != REQUEST_MAGIC {
            return Err((Status::BadHeader, id));
        }
        id.copy_from_slice(&b[8..24]);
        let version = u16_at(b, 4);
        if version != VERSION && version != VERSION_MULTI {
            return Err((Status::Version, id));
        }
        let kind = Kind::from_u16(u16_at(b, 6)).ok_or((Status::BadHeader, id))?;
        if kind.version() != version {
            // version 2 is the multi frame only, and a multi frame is version 2 only
            return Err((Status::BadHeader, id));
        }
        if b[40..48].iter().any(|&x| x != 0) {
            return Err((Status::BadHeader, id));
        }
        let h = RequestHeader {
            kind,
            request_id: id,
            linear: u32_at(b, 24),
            rows: u16_at(b, 28),
            ring: u16_at(b, 30),
            deadline_ms: u32_at(b, 32),
            body_len: u32_at(b, 36),
        };
        match kind {
            Kind::Open => {
                if h.linear != 0
                    || h.rows != 0
                    || h.ring != 0
                    || h.deadline_ms != 0
                    || h.body_len as usize != TICKET_LEN
                {
                    return Err((Status::BadHeader, id));
                }
            }
            Kind::Call => {
                if h.rows == 0 {
                    return Err((Status::BadHeader, id));
                }
            }
            Kind::Shm => {
                if h.linear != 0 || h.rows != 0 || h.ring != 0 || h.deadline_ms != 0 || h.body_len != 0 {
                    return Err((Status::BadHeader, id));
                }
            }
            Kind::Multi => {
                if h.linear == 0
                    || h.linear > u32::from(MAX_ITEMS)
                    || h.rows == 0
                    || h.rows > MAX_MULTI_ROWS
                    || (h.body_len as u64) < h.linear as u64 * ITEM_LEN as u64
                {
                    return Err((Status::BadHeader, id));
                }
            }
        }
        Ok(h)
    }

    /// Encode (clients and tests).
    pub fn encode(&self) -> [u8; REQUEST_HEADER_LEN] {
        let mut b = [0u8; REQUEST_HEADER_LEN];
        b[0..4].copy_from_slice(&REQUEST_MAGIC);
        b[4..6].copy_from_slice(&self.kind.version().to_le_bytes());
        b[6..8].copy_from_slice(&self.kind.as_u16().to_le_bytes());
        b[8..24].copy_from_slice(&self.request_id);
        b[24..28].copy_from_slice(&self.linear.to_le_bytes());
        b[28..30].copy_from_slice(&self.rows.to_le_bytes());
        b[30..32].copy_from_slice(&self.ring.to_le_bytes());
        b[32..36].copy_from_slice(&self.deadline_ms.to_le_bytes());
        b[36..40].copy_from_slice(&self.body_len.to_le_bytes());
        b
    }
}

/// A reply header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplyHeader {
    /// Status.
    pub status: Status,
    /// Echoed request id.
    pub request_id: [u8; 16],
    /// Time between receipt and the start of compute (device lock wait), microseconds.
    pub queue_us: u32,
    /// Compute time, microseconds.
    pub compute_us: u32,
    /// Retry hint for `Unavailable` / `RateLimited`, milliseconds.
    pub retry_after_ms: u32,
    /// Body length.
    pub body_len: u32,
}

impl ReplyHeader {
    /// A body-less reply with `status`.
    pub fn refusal(status: Status, request_id: [u8; 16], retry_after_ms: u32) -> ReplyHeader {
        ReplyHeader { status, request_id, queue_us: 0, compute_us: 0, retry_after_ms, body_len: 0 }
    }

    /// Encode.
    pub fn encode(&self) -> [u8; REPLY_HEADER_LEN] {
        let mut b = [0u8; REPLY_HEADER_LEN];
        b[0..4].copy_from_slice(&REPLY_MAGIC);
        b[4..6].copy_from_slice(&VERSION.to_le_bytes());
        b[6..8].copy_from_slice(&self.status.code().to_le_bytes());
        b[8..24].copy_from_slice(&self.request_id);
        b[24..28].copy_from_slice(&self.queue_us.to_le_bytes());
        b[28..32].copy_from_slice(&self.compute_us.to_le_bytes());
        b[32..36].copy_from_slice(&self.retry_after_ms.to_le_bytes());
        b[36..40].copy_from_slice(&self.body_len.to_le_bytes());
        b
    }

    /// Parse (clients and tests). `None` on a bad magic, version or status code.
    pub fn parse(b: &[u8; REPLY_HEADER_LEN]) -> Option<ReplyHeader> {
        if b[0..4] != REPLY_MAGIC || u16_at(b, 4) != VERSION {
            return None;
        }
        let status = Status::from_code(u16_at(b, 6))?;
        let mut id = [0u8; 16];
        id.copy_from_slice(&b[8..24]);
        Some(ReplyHeader {
            status,
            request_id: id,
            queue_us: u32_at(b, 24),
            compute_us: u32_at(b, 28),
            retry_after_ms: u32_at(b, 32),
            body_len: u32_at(b, 36),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(rows: u16, body_len: u32) -> RequestHeader {
        RequestHeader { kind: Kind::Call, request_id: [7u8; 16], linear: 3, rows, ring: 1, deadline_ms: 250, body_len }
    }

    #[test]
    fn request_roundtrip_and_layout() {
        let h = call(2, 2 * 896 * 4);
        let b = h.encode();
        assert_eq!(&b[0..4], b"PGWL");
        assert_eq!(u16_at(&b, 4), 1);
        assert_eq!(u16_at(&b, 6), 2);
        assert_eq!(&b[8..24], &[7u8; 16]);
        assert_eq!(u32_at(&b, 24), 3);
        assert_eq!(u16_at(&b, 28), 2);
        assert_eq!(u16_at(&b, 30), 1);
        assert_eq!(u32_at(&b, 32), 250);
        assert_eq!(u32_at(&b, 36), 2 * 896 * 4);
        assert!(b[40..48].iter().all(|&x| x == 0));
        assert_eq!(RequestHeader::parse(&b).unwrap(), h);
    }

    #[test]
    fn reply_roundtrip() {
        let r = ReplyHeader {
            status: Status::RateLimited,
            request_id: [9u8; 16],
            queue_us: 1,
            compute_us: 2,
            retry_after_ms: 3,
            body_len: 0,
        };
        assert_eq!(ReplyHeader::parse(&r.encode()).unwrap(), r);
    }

    #[test]
    fn malformed_headers_are_refused() {
        let good = call(1, 16).encode();
        let mut b = good;
        b[0] = b'X';
        assert_eq!(RequestHeader::parse(&b).unwrap_err().0, Status::BadHeader);
        let mut b = good;
        b[4] = 3; // 2 is the multi frame's version (BadHeader on a call, multi_frame_layout_and_versions)
        assert_eq!(RequestHeader::parse(&b).unwrap_err(), (Status::Version, [7u8; 16]));
        let mut b = good;
        b[6] = 9;
        assert_eq!(RequestHeader::parse(&b).unwrap_err().0, Status::BadHeader);
        let mut b = good;
        b[47] = 1;
        assert_eq!(RequestHeader::parse(&b).unwrap_err().0, Status::BadHeader);
        let b = call(0, 16).encode();
        assert_eq!(RequestHeader::parse(&b).unwrap_err().0, Status::BadHeader);
        // an open frame with anything but the ticket length or with call fields set
        let mut o = RequestHeader {
            kind: Kind::Open,
            request_id: [0; 16],
            linear: 0,
            rows: 0,
            ring: 0,
            deadline_ms: 0,
            body_len: TICKET_LEN as u32,
        };
        assert!(RequestHeader::parse(&o.encode()).is_ok());
        o.rows = 1;
        assert_eq!(RequestHeader::parse(&o.encode()).unwrap_err().0, Status::BadHeader);
        o.rows = 0;
        o.body_len = 31;
        assert_eq!(RequestHeader::parse(&o.encode()).unwrap_err().0, Status::BadHeader);
    }

    fn multi(count: u32, rows: u16, body_len: u32) -> RequestHeader {
        RequestHeader {
            kind: Kind::Multi,
            request_id: [5u8; 16],
            linear: count,
            rows,
            ring: 0,
            deadline_ms: 0,
            body_len,
        }
    }

    #[test]
    fn multi_frame_layout_and_versions() {
        let h = multi(3, 4, 3 * 8 + 4 * 16);
        let b = h.encode();
        assert_eq!(&b[0..4], b"PGWL");
        assert_eq!(u16_at(&b, 4), 2, "a multi frame is version 2");
        assert_eq!(u16_at(&b, 6), 4);
        assert_eq!(u32_at(&b, 24), 3);
        assert_eq!(u16_at(&b, 28), 4);
        assert_eq!(RequestHeader::parse(&b).unwrap(), h);
        // version 1 frames keep version 1
        assert_eq!(u16_at(&call(1, 4).encode(), 4), 1);
        // a multi frame claiming version 1, or a call claiming version 2: refused
        let mut v1 = b;
        v1[4] = 1;
        assert_eq!(RequestHeader::parse(&v1).unwrap_err().0, Status::BadHeader);
        let mut c2 = call(1, 4).encode();
        c2[4] = 2;
        assert_eq!(RequestHeader::parse(&c2).unwrap_err().0, Status::BadHeader);
        let mut v3 = b;
        v3[4] = 3;
        assert_eq!(RequestHeader::parse(&v3).unwrap_err(), (Status::Version, [5u8; 16]));
        // field ranges
        for bad in [multi(0, 1, 0), multi(65, 65, 65 * 8 + 65), multi(2, 0, 16), multi(2, 257, 16), multi(2, 2, 15)] {
            assert_eq!(RequestHeader::parse(&bad.encode()).unwrap_err().0, Status::BadHeader, "{bad:?}");
        }
        assert!(RequestHeader::parse(&multi(64, 256, 64 * 8).encode()).is_ok());
    }

    #[test]
    fn item_table() {
        let items = [Item { linear: 7, rows: 1 }, Item { linear: 7, rows: 3 }, Item { linear: 2, rows: 8 }];
        let mut b = Vec::new();
        encode_items(&items, &mut b);
        assert_eq!(b.len(), 24);
        assert_eq!(&b[0..8], &[7, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(parse_items(&b, 3, 12).unwrap(), items.to_vec());
        assert_eq!(parse_items(&b, 3, 11).unwrap_err(), Status::BadHeader, "rows must sum to the header's");
        assert_eq!(parse_items(&b[..16], 3, 12).unwrap_err(), Status::BadHeader);
        let mut r = b.clone();
        r[6] = 1; // reserved
        assert_eq!(parse_items(&r, 3, 12).unwrap_err(), Status::BadHeader);
        let mut z = b.clone();
        z[4] = 0; // zero rows
        assert_eq!(parse_items(&z, 3, 11).unwrap_err(), Status::BadHeader);
        let mut big = b.clone();
        big[4] = 9; // above one call's MAX_ROWS: allowed (the service cuts it into calls)
        assert!(parse_items(&big, 3, 20).is_ok());
        big[4] = 0x01;
        big[5] = 0x01; // 257 rows: above MAX_MULTI_ROWS
        assert_eq!(parse_items(&big, 3, 268).unwrap_err(), Status::BadHeader);
    }

    #[test]
    fn status_classes() {
        for s in Status::ALL {
            assert_eq!(Status::from_code(s.code()), Some(s));
            assert_eq!(s.is_fatal(), s.code() >= 16);
        }
        assert!(Status::Fallback.is_fallback() && Status::Unavailable.is_fallback() && Status::Internal.is_fallback());
        assert!(Status::Busy.is_fallback() && Status::Busy.is_fatal());
        assert!(!Status::RateLimited.is_fallback() && !Status::Ok.is_fallback());
        assert_eq!(Status::from_code(6), None);
    }

    /// Deterministic fuzz: random and mutated headers never panic, and anything accepted
    /// re-encodes to the same bytes.
    #[test]
    fn parser_fuzz() {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let seeds = [call(1, 4).encode(), call(8, 1 << 20).encode(), multi(4, 4, 1 << 12).encode()];
        for i in 0..200_000 {
            let mut b = if i % 2 == 0 { seeds[(i / 2) % 3] } else { [0u8; REQUEST_HEADER_LEN] };
            let flips = 1 + (next() % 4) as usize;
            for _ in 0..flips {
                let r = next();
                b[(r % 48) as usize] = (r >> 8) as u8;
            }
            if i % 7 == 0 {
                for x in b.iter_mut() {
                    *x = next() as u8;
                }
            }
            if let Ok(h) = RequestHeader::parse(&b) {
                assert_eq!(h.encode(), b);
            }
        }
    }
}
