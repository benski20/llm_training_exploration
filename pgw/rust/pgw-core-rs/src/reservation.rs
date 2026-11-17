//! Single-use mask accounting (SPEC.md 7.5 and 7.6; mirrors `pgw.maskstore.store`).
//!
//! This module holds the pure bookkeeping rules, independent of storage, so they can be audited
//! and property-tested in isolation:
//!
//! * [`Ledger`]: per-segment consumption cursors (`next_slot`), the global `generation` counter and
//!   `consumed_total`. [`Ledger::reserve`] is all-or-nothing, takes the oldest segments first and
//!   only ever moves cursors forward. A durable store must persist the new ledger state **before**
//!   any mask bytes are released (crash loses masks, never reuses them).
//! * [`Ledger::check_successor`] and [`Ledger::check_witness`]: rollback detection between two
//!   observed states, and against a monotonic witness counter.
//! * [`ReservationCursor`]: hands out rows of one reservation strictly in order, one cursor per
//!   linear, never across a segment-extent boundary (Freivalds vectors differ per segment), and
//!   burns everything that was not taken when closed.

use std::collections::BTreeMap;

use crate::error::{ErrorCode, PgwError, Result};

/// Lifecycle of a segment (same strings as `pgw.maskstore.state.SegmentStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentStatus {
    /// Has unconsumed slots (or none consumed yet).
    Active,
    /// Every slot handed out; the file awaits shredding.
    Consumed,
    /// File securely deleted.
    Shredded,
    /// Moved out of this store.
    Exported,
    /// File vanished; remaining slots written off.
    Missing,
    /// File failed validation; remaining slots written off.
    Quarantined,
}

impl SegmentStatus {
    /// The state-database string.
    pub const fn as_str(self) -> &'static str {
        match self {
            SegmentStatus::Active => "active",
            SegmentStatus::Consumed => "consumed",
            SegmentStatus::Shredded => "shredded",
            SegmentStatus::Exported => "exported",
            SegmentStatus::Missing => "missing",
            SegmentStatus::Quarantined => "quarantined",
        }
    }

    /// Parse a state-database string.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "active" => SegmentStatus::Active,
            "consumed" => SegmentStatus::Consumed,
            "shredded" => SegmentStatus::Shredded,
            "exported" => SegmentStatus::Exported,
            "missing" => SegmentStatus::Missing,
            "quarantined" => SegmentStatus::Quarantined,
            other => return Err(PgwError::new(ErrorCode::MaskStore, format!("unknown segment status {other:?}"))),
        })
    }
}

/// Consumption state of one segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentState {
    /// Unique segment id (never reused).
    pub segment_id: String,
    /// Total slots in the segment.
    pub slots: u64,
    /// Slots `[0, next_slot)` are consumed forever.
    pub next_slot: u64,
    /// Lifecycle status.
    pub status: SegmentStatus,
}

impl SegmentState {
    /// Unconsumed slots (0 unless active, and 0 for an impossible cursor `next_slot > slots`).
    pub fn available(&self) -> u64 {
        if self.status == SegmentStatus::Active {
            self.slots.saturating_sub(self.next_slot)
        } else {
            0
        }
    }
}

/// A contiguous run of slots `[start, start + count)` inside one segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotExtent {
    /// Segment holding the slots.
    pub segment_id: String,
    /// First slot.
    pub start: u64,
    /// Number of slots (> 0).
    pub count: u64,
}

/// Result of a successful [`Ledger::reserve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    /// Generation of the consuming transaction (strictly increasing).
    pub generation: u64,
    /// Slots reserved.
    pub slots: u64,
    /// Where the slots live, oldest segment first.
    pub extents: Vec<SlotExtent>,
}

/// How a witness counter relates to the ledger generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessStatus {
    /// Witness equals the ledger generation.
    InSync,
    /// Witness lags (a crash between commit and witness update); the caller advances it.
    Behind(u64),
}

/// In-memory model of one `(fingerprint, ring)` pool of a mask store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ledger {
    generation: u64,
    consumed_total: u64,
    segments: Vec<SegmentState>,
}

fn store_err(msg: impl Into<String>) -> PgwError {
    PgwError::new(ErrorCode::MaskStore, msg)
}

fn reuse_err(msg: impl Into<String>) -> PgwError {
    PgwError::new(ErrorCode::MaskReuse, msg)
}

impl Ledger {
    /// Empty ledger at generation 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild a ledger from persisted state and validate its invariants.
    pub fn from_parts(generation: u64, consumed_total: u64, segments: Vec<SegmentState>) -> Result<Self> {
        let l = Self { generation, consumed_total, segments };
        l.verify()?;
        Ok(l)
    }

    /// Current generation (one increment per consuming transaction).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Total slots consumed or written off, ever.
    pub fn consumed_total(&self) -> u64 {
        self.consumed_total
    }

    /// Segments in import order.
    pub fn segments(&self) -> &[SegmentState] {
        &self.segments
    }

    /// Slots available for reservation (saturates at `u64::MAX`).
    pub fn available(&self) -> u64 {
        self.segments.iter().map(SegmentState::available).fold(0u64, u64::saturating_add)
    }

    fn find_mut(&mut self, segment_id: &str) -> Result<&mut SegmentState> {
        self.segments
            .iter_mut()
            .find(|s| s.segment_id == segment_id)
            .ok_or_else(|| store_err(format!("unknown segment {segment_id}")))
    }

    /// Register a new, fully unconsumed segment. Ids are unique forever.
    pub fn add_segment(&mut self, segment_id: &str, slots: u64) -> Result<()> {
        if slots == 0 {
            return Err(PgwError::config("a segment must have at least one slot"));
        }
        if self.segments.iter().any(|s| s.segment_id == segment_id) {
            return Err(reuse_err(format!("segment {segment_id} was already imported; refusing to import it twice")));
        }
        self.segments.push(SegmentState {
            segment_id: segment_id.to_owned(),
            slots,
            next_slot: 0,
            status: SegmentStatus::Active,
        });
        Ok(())
    }

    /// Atomically consume `slots` slots, oldest segments first.
    ///
    /// On `mask_exhausted` the ledger is unchanged. On success cursors advance, segments that
    /// become full are marked consumed, and the generation increases by one.
    pub fn reserve(&mut self, slots: u64) -> Result<Reservation> {
        if slots == 0 {
            return Err(PgwError::config("slots must be a positive integer"));
        }
        // Exact (non-saturating) total so the plan below always covers `slots`.
        let available: u128 = self.segments.iter().map(|s| s.available() as u128).sum();
        if available < slots as u128 {
            return Err(PgwError::new(
                ErrorCode::MaskExhausted,
                format!("not enough mask slots in the store: needed {slots}, available {available}"),
            ));
        }
        let generation = self.generation.checked_add(1).ok_or_else(|| store_err("generation counter overflow"))?;
        let consumed_total =
            self.consumed_total.checked_add(slots).ok_or_else(|| store_err("consumed total overflow"))?;
        // Plan first, then apply: the ledger is either fully updated or untouched.
        let mut need = slots;
        let mut plan: Vec<(usize, u64)> = Vec::new();
        for (i, seg) in self.segments.iter().enumerate() {
            if need == 0 {
                break;
            }
            let take = need.min(seg.available());
            if take > 0 {
                plan.push((i, take));
                need -= take;
            }
        }
        if need != 0 {
            return Err(store_err("internal: reservation plan does not cover the request"));
        }
        let mut extents = Vec::with_capacity(plan.len());
        for (i, take) in plan {
            let seg = &mut self.segments[i];
            extents.push(SlotExtent { segment_id: seg.segment_id.clone(), start: seg.next_slot, count: take });
            seg.next_slot += take; // take <= slots - next_slot: cannot overflow
            if seg.next_slot == seg.slots {
                seg.status = SegmentStatus::Consumed;
            }
        }
        self.generation = generation;
        self.consumed_total = consumed_total;
        Ok(Reservation { generation, slots, extents })
    }

    /// Write off a segment's remaining slots (`Missing`, `Quarantined` or `Exported`).
    ///
    /// The remaining slots count as consumed: they can never be handed out again. The rules follow
    /// `pgw.maskstore.store`:
    ///
    /// * `Exported`: only a segment that is `Active` with **no** slot consumed may leave the store
    ///   (`mask_reuse` otherwise), because the receiving store would hand out its consumed slots again.
    /// * `Missing` / `Quarantined`: only a segment whose file is still expected on disk
    ///   (`Active` or `Consumed`); anything else is a `mask_store` error.
    ///
    /// On error the ledger is unchanged. Returns the number of slots written off.
    pub fn write_off(&mut self, segment_id: &str, status: SegmentStatus) -> Result<u64> {
        if !matches!(status, SegmentStatus::Missing | SegmentStatus::Quarantined | SegmentStatus::Exported) {
            return Err(PgwError::config(format!("cannot write off a segment as {}", status.as_str())));
        }
        let generation = self.generation.checked_add(1).ok_or_else(|| store_err("generation counter overflow"))?;
        let consumed_total = self.consumed_total;
        let seg = self.find_mut(segment_id)?;
        if status == SegmentStatus::Exported {
            if seg.status != SegmentStatus::Active || seg.next_slot != 0 {
                return Err(reuse_err(format!(
                    "segment {segment_id} is {} with {} slots consumed; only an unused segment can be exported",
                    seg.status.as_str(),
                    seg.next_slot
                )));
            }
        } else if !matches!(seg.status, SegmentStatus::Active | SegmentStatus::Consumed) {
            return Err(store_err(format!(
                "segment {segment_id} is {}; only active or consumed segments can be written off",
                seg.status.as_str()
            )));
        }
        let lost = seg.slots.checked_sub(seg.next_slot).ok_or_else(|| store_err("impossible segment cursor"))?;
        let consumed_total = consumed_total.checked_add(lost).ok_or_else(|| store_err("consumed total overflow"))?;
        seg.next_slot = seg.slots;
        seg.status = status;
        self.consumed_total = consumed_total;
        self.generation = generation;
        Ok(lost)
    }

    /// Mark a fully consumed segment as shredded (file and wrapped DEK destroyed).
    pub fn mark_shredded(&mut self, segment_id: &str) -> Result<()> {
        let seg = self.find_mut(segment_id)?;
        if seg.status != SegmentStatus::Consumed {
            return Err(store_err(format!(
                "segment {segment_id} is {}; only consumed segments are shredded",
                seg.status.as_str()
            )));
        }
        seg.status = SegmentStatus::Shredded;
        Ok(())
    }

    /// Check the ledger invariants (`verify_integrity` in the Python store).
    pub fn verify(&self) -> Result<()> {
        let mut sum: u64 = 0;
        for (i, s) in self.segments.iter().enumerate() {
            if s.slots == 0 || s.next_slot > s.slots {
                return Err(store_err(format!("segment {} has an impossible cursor", s.segment_id)));
            }
            if s.status != SegmentStatus::Active && s.next_slot != s.slots {
                return Err(store_err(format!(
                    "segment {} is {} but has unconsumed slots",
                    s.segment_id,
                    s.status.as_str()
                )));
            }
            if self.segments[..i].iter().any(|o| o.segment_id == s.segment_id) {
                return Err(store_err(format!("duplicate segment id {}", s.segment_id)));
            }
            sum = sum.checked_add(s.next_slot).ok_or_else(|| store_err("consumed total overflow"))?;
        }
        if sum != self.consumed_total {
            return Err(reuse_err(format!(
                "consumed total {} does not match the segment cursors ({sum}); possible rollback",
                self.consumed_total
            )));
        }
        Ok(())
    }

    /// Check that `self` can legitimately follow `earlier`: no segment vanished, no cursor moved
    /// back, no consumed segment became active again, and generation/totals did not decrease.
    /// A violation means state was rolled back and masks could be reused.
    pub fn check_successor(&self, earlier: &Ledger) -> Result<()> {
        if self.generation < earlier.generation || self.consumed_total < earlier.consumed_total {
            return Err(reuse_err("mask store generation or consumed total went backwards (rollback)"));
        }
        for old in &earlier.segments {
            let Some(new) = self.segments.iter().find(|s| s.segment_id == old.segment_id) else {
                return Err(reuse_err(format!("segment {} disappeared from the state (rollback)", old.segment_id)));
            };
            if new.slots != old.slots || new.next_slot < old.next_slot {
                return Err(reuse_err(format!("segment {} cursor moved backwards (rollback)", old.segment_id)));
            }
            if old.status != SegmentStatus::Active && new.status == SegmentStatus::Active {
                return Err(reuse_err(format!("segment {} became active again (rollback)", old.segment_id)));
            }
        }
        Ok(())
    }

    /// Compare with a monotonic witness (HSM/TPM counter or HMAC journal).
    ///
    /// A witness **ahead** of the ledger means the state was restored from an older copy: refuse
    /// to serve (`mask_reuse`). A witness behind is a benign crash window and is reported so the
    /// caller can advance it.
    pub fn check_witness(&self, witness_generation: u64) -> Result<WitnessStatus> {
        match witness_generation.cmp(&self.generation) {
            std::cmp::Ordering::Greater => Err(reuse_err(format!(
                "witness generation {witness_generation} is ahead of the state generation {} (rollback)",
                self.generation
            ))),
            std::cmp::Ordering::Equal => Ok(WitnessStatus::InSync),
            std::cmp::Ordering::Less => Ok(WitnessStatus::Behind(self.generation - witness_generation)),
        }
    }
}

/// Location of rows handed out by [`ReservationCursor::take`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TakePlan {
    /// Index of the extent within the reservation.
    pub extent_index: usize,
    /// Segment holding the rows (selects seed, `u` and `W r` chunks).
    pub segment_id: String,
    /// Slot index inside the segment of the first row: `r = ring_uniform_rows(n, K, slot0, rows, "r", linear)`.
    pub slot0: u64,
    /// Number of rows.
    pub rows: u64,
}

/// Row cursors of one reservation (one per linear). Rows are never handed out twice.
///
/// Deliberately **not** `Clone`: a copy would carry the same cursors and could hand out the same
/// mask rows a second time.
#[derive(Debug)]
pub struct ReservationCursor {
    extents: Vec<(SlotExtent, u64)>, // (extent, first reservation row)
    slots: u64,
    cursors: BTreeMap<String, u64>,
    closed: bool,
}

impl ReservationCursor {
    /// Create cursors for `linear_ids` over the reservation's extents.
    ///
    /// Extents must be non-empty and must not overlap within a segment (overlapping extents would
    /// map two reservation rows to the same mask slot): a violation is a `mask_reuse` error.
    pub fn new<S: AsRef<str>>(extents: &[SlotExtent], linear_ids: &[S]) -> Result<Self> {
        if extents.is_empty() {
            return Err(store_err("reservation without extents"));
        }
        if linear_ids.is_empty() {
            return Err(PgwError::config("reservation without linears"));
        }
        let mut row = 0u64;
        let mut ext: Vec<(SlotExtent, u64)> = Vec::with_capacity(extents.len());
        for e in extents {
            if e.count == 0 {
                return Err(store_err("empty reservation extent"));
            }
            let end = e.start.checked_add(e.count).ok_or_else(|| store_err("reservation extent overflows"))?;
            for (o, _) in &ext {
                if o.segment_id == e.segment_id && e.start < o.start + o.count && o.start < end {
                    return Err(reuse_err(format!(
                        "reservation extents overlap in segment {}; refusing to hand out a slot twice",
                        e.segment_id
                    )));
                }
            }
            ext.push((e.clone(), row));
            row = row.checked_add(e.count).ok_or_else(|| store_err("reservation size overflow"))?;
        }
        let mut cursors = BTreeMap::new();
        for id in linear_ids {
            if cursors.insert(id.as_ref().to_owned(), 0).is_some() {
                return Err(PgwError::config(format!("duplicate linear id {}", id.as_ref())));
            }
        }
        Ok(Self { extents: ext, slots: row, cursors, closed: false })
    }

    /// Total slots reserved.
    pub fn slots(&self) -> u64 {
        self.slots
    }

    /// `true` after [`ReservationCursor::close`].
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn cursor(&self, lid: &str) -> Result<u64> {
        self.cursors
            .get(lid)
            .copied()
            .ok_or_else(|| PgwError::config(format!("linear {lid} is not covered by this reservation")))
    }

    fn extent_index(&self, row: u64) -> Option<usize> {
        self.extents.iter().position(|(e, first)| *first <= row && row < first + e.count)
    }

    /// Rows already taken for `lid`.
    pub fn taken(&self, lid: &str) -> Result<u64> {
        self.cursor(lid)
    }

    /// Rows not yet taken for `lid`.
    pub fn remaining(&self, lid: &str) -> Result<u64> {
        Ok(self.slots - self.cursor(lid)?)
    }

    /// Rows of `lid` left in the current extent (0 when exhausted).
    pub fn rows_until_boundary(&self, lid: &str) -> Result<u64> {
        let cur = self.cursor(lid)?;
        Ok(match self.extent_index(cur) {
            Some(i) => self.extents[i].1 + self.extents[i].0.count - cur,
            None => 0,
        })
    }

    /// Take the next `rows` rows of `lid`. The cursor advances before the plan is returned, so the
    /// rows are never handed out again even if the caller fails to use them.
    ///
    /// Errors: `mask_reuse` after [`ReservationCursor::close`], `config` for an unknown linear,
    /// `rows == 0` or a take that crosses an extent boundary, `mask_exhausted` past the end. A
    /// refused take consumes nothing.
    pub fn take(&mut self, lid: &str, rows: u64) -> Result<TakePlan> {
        if self.closed {
            return Err(reuse_err("mask reservation is closed; its unused rows were burned"));
        }
        let start = self.cursor(lid)?;
        if rows == 0 {
            return Err(PgwError::config("rows must be a positive integer"));
        }
        let end = start.checked_add(rows).filter(|&e| e <= self.slots).ok_or_else(|| {
            PgwError::new(
                ErrorCode::MaskExhausted,
                format!("mask reservation exhausted: linear {lid} needs {rows}, {} remaining", self.slots - start),
            )
        })?;
        let ei = self.extent_index(start).expect("start < slots lies in some extent");
        let (ext, first) = &self.extents[ei];
        if end > first + ext.count {
            return Err(PgwError::config(format!(
                "take({lid}, {rows}) crosses a segment extent boundary; split at rows_until_boundary() = {}",
                first + ext.count - start
            )));
        }
        let plan =
            TakePlan { extent_index: ei, segment_id: ext.segment_id.clone(), slot0: ext.start + (start - first), rows };
        self.cursors.insert(lid.to_owned(), end);
        Ok(plan)
    }

    /// Close the reservation and return the number of burned slots (`slots - max cursor`).
    /// Returns `None` if it was already closed (idempotent; burned slots are counted once).
    pub fn close(&mut self) -> Option<u64> {
        if self.closed {
            return None;
        }
        self.closed = true;
        let used = self.cursors.values().copied().max().unwrap_or(0);
        Some(self.slots - used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger(segs: &[(&str, u64)]) -> Ledger {
        let mut l = Ledger::new();
        for (id, n) in segs {
            l.add_segment(id, *n).unwrap();
        }
        l
    }

    #[test]
    fn reserve_spans_segments_in_order() {
        let mut l = ledger(&[("a", 5), ("b", 3), ("c", 10)]);
        assert_eq!(l.available(), 18);
        let r1 = l.reserve(4).unwrap();
        assert_eq!(r1.generation, 1);
        assert_eq!(r1.extents, vec![SlotExtent { segment_id: "a".into(), start: 0, count: 4 }]);
        let r2 = l.reserve(6).unwrap();
        assert_eq!(r2.generation, 2);
        assert_eq!(
            r2.extents,
            vec![
                SlotExtent { segment_id: "a".into(), start: 4, count: 1 },
                SlotExtent { segment_id: "b".into(), start: 0, count: 3 },
                SlotExtent { segment_id: "c".into(), start: 0, count: 2 },
            ]
        );
        assert_eq!(l.segments()[0].status, SegmentStatus::Consumed);
        assert_eq!(l.segments()[1].status, SegmentStatus::Consumed);
        assert_eq!(l.segments()[2].status, SegmentStatus::Active);
        assert_eq!(l.consumed_total(), 10);
        assert_eq!(l.available(), 8);
        l.verify().unwrap();
    }

    #[test]
    fn exhaustion_is_atomic() {
        let mut l = ledger(&[("a", 2), ("b", 2)]);
        let before = l.clone();
        let e = l.reserve(5).unwrap_err();
        assert_eq!(e.code(), ErrorCode::MaskExhausted);
        assert_eq!(l, before);
        assert_eq!(l.reserve(0).unwrap_err().code(), ErrorCode::Config);
        l.reserve(4).unwrap();
        assert_eq!(l.reserve(1).unwrap_err().code(), ErrorCode::MaskExhausted);
    }

    #[test]
    fn no_slot_is_ever_handed_out_twice() {
        let mut l = ledger(&[("a", 7), ("b", 1), ("c", 13), ("d", 4)]);
        let mut seen = std::collections::HashSet::new();
        let mut gen = 0;
        for n in [1u64, 3, 5, 2, 8, 1, 5] {
            let r = l.reserve(n).unwrap();
            assert_eq!(r.generation, gen + 1);
            gen = r.generation;
            assert_eq!(r.extents.iter().map(|e| e.count).sum::<u64>(), n);
            for e in &r.extents {
                for s in e.start..e.start + e.count {
                    assert!(seen.insert((e.segment_id.clone(), s)), "slot reused");
                }
            }
            l.verify().unwrap();
        }
        assert_eq!(l.available(), 0);
        assert_eq!(seen.len(), 25);
    }

    #[test]
    fn duplicate_import_and_write_off() {
        let mut l = ledger(&[("a", 4), ("b", 4)]);
        assert_eq!(l.add_segment("a", 4).unwrap_err().code(), ErrorCode::MaskReuse);
        assert_eq!(l.add_segment("z", 0).unwrap_err().code(), ErrorCode::Config);
        l.reserve(1).unwrap();
        assert_eq!(l.write_off("a", SegmentStatus::Missing).unwrap(), 3);
        assert_eq!(l.consumed_total(), 4);
        assert_eq!(l.generation(), 2);
        assert!(l.write_off("a", SegmentStatus::Active).is_err());
        assert!(l.write_off("nope", SegmentStatus::Missing).is_err());
        let r = l.reserve(2).unwrap();
        assert_eq!(r.extents[0].segment_id, "b");
        l.verify().unwrap();
        assert!(l.mark_shredded("b").is_err()); // still active
        l.reserve(2).unwrap();
        l.mark_shredded("b").unwrap();
        assert!(l.write_off("b", SegmentStatus::Quarantined).is_err());
        l.verify().unwrap();
    }

    #[test]
    fn verify_detects_inconsistency() {
        let segs = vec![SegmentState { segment_id: "a".into(), slots: 4, next_slot: 2, status: SegmentStatus::Active }];
        assert!(Ledger::from_parts(1, 2, segs.clone()).is_ok());
        assert_eq!(Ledger::from_parts(1, 1, segs.clone()).unwrap_err().code(), ErrorCode::MaskReuse);
        let mut bad = segs.clone();
        bad[0].status = SegmentStatus::Consumed;
        assert!(Ledger::from_parts(1, 2, bad).is_err());
        let mut bad = segs.clone();
        bad[0].next_slot = 5;
        assert!(Ledger::from_parts(1, 5, bad).is_err());
        let mut dup = segs.clone();
        dup.push(dup[0].clone());
        assert!(Ledger::from_parts(1, 4, dup).is_err());
        for s in ["active", "consumed", "shredded", "exported", "missing", "quarantined"] {
            assert_eq!(SegmentStatus::parse(s).unwrap().as_str(), s);
        }
        assert!(SegmentStatus::parse("bogus").is_err());
    }

    #[test]
    fn rollback_detection() {
        let mut l = ledger(&[("a", 4), ("b", 4)]);
        let snap = l.clone();
        l.reserve(5).unwrap();
        l.check_successor(&snap).unwrap();
        assert_eq!(snap.check_successor(&l).unwrap_err().code(), ErrorCode::MaskReuse);
        let mut forged = l.clone();
        forged.segments[1].next_slot = 0;
        forged.segments[1].status = SegmentStatus::Active;
        assert!(forged.check_successor(&l).is_err());
        let mut dropped = l.clone();
        dropped.segments.remove(0);
        assert!(dropped.check_successor(&l).is_err());
        assert_eq!(l.check_witness(1).unwrap(), WitnessStatus::InSync);
        assert_eq!(l.check_witness(0).unwrap(), WitnessStatus::Behind(1));
        assert_eq!(l.check_witness(2).unwrap_err().code(), ErrorCode::MaskReuse);
    }

    #[test]
    fn cursor_takes_in_order_per_linear() {
        let ext = vec![
            SlotExtent { segment_id: "a".into(), start: 4, count: 1 },
            SlotExtent { segment_id: "b".into(), start: 0, count: 3 },
        ];
        let mut c = ReservationCursor::new(&ext, &["0.qkv", "0.o"]).unwrap();
        assert_eq!(c.slots(), 4);
        assert_eq!(c.rows_until_boundary("0.qkv").unwrap(), 1);
        let e = c.take("0.qkv", 2).unwrap_err();
        assert_eq!(e.code(), ErrorCode::Config); // crosses boundary
        assert_eq!(c.taken("0.qkv").unwrap(), 0); // a refused take does not consume
        let p = c.take("0.qkv", 1).unwrap();
        assert_eq!(p, TakePlan { extent_index: 0, segment_id: "a".into(), slot0: 4, rows: 1 });
        let p = c.take("0.qkv", 2).unwrap();
        assert_eq!(p, TakePlan { extent_index: 1, segment_id: "b".into(), slot0: 0, rows: 2 });
        assert_eq!(c.remaining("0.qkv").unwrap(), 1);
        assert_eq!(c.remaining("0.o").unwrap(), 4); // independent cursor
        assert_eq!(c.take("0.qkv", 2).unwrap_err().code(), ErrorCode::MaskExhausted);
        assert_eq!(c.take("1.qkv", 1).unwrap_err().code(), ErrorCode::Config);
        assert_eq!(c.take("0.o", 0).unwrap_err().code(), ErrorCode::Config);
        c.take("0.qkv", 1).unwrap();
        assert_eq!(c.rows_until_boundary("0.qkv").unwrap(), 0);
        assert_eq!(c.close(), Some(0));
        assert_eq!(c.close(), None);
        assert_eq!(c.take("0.o", 1).unwrap_err().code(), ErrorCode::MaskReuse);
    }

    #[test]
    fn export_and_write_off_rules() {
        let mut l = ledger(&[("a", 4), ("b", 4), ("c", 2)]);
        // a fresh segment can be exported; it counts as consumed
        assert_eq!(l.write_off("b", SegmentStatus::Exported).unwrap(), 4);
        assert_eq!(l.consumed_total(), 4);
        l.reserve(1).unwrap(); // takes a[0]
        let before = l.clone();
        // a partly used segment must never leave the store (the receiver would reuse slot 0)
        assert_eq!(l.write_off("a", SegmentStatus::Exported).unwrap_err().code(), ErrorCode::MaskReuse);
        // an exported / written-off segment cannot be written off again
        assert_eq!(l.write_off("b", SegmentStatus::Missing).unwrap_err().code(), ErrorCode::MaskStore);
        assert_eq!(l.write_off("b", SegmentStatus::Exported).unwrap_err().code(), ErrorCode::MaskReuse);
        assert_eq!(l, before, "refused write-offs leave the ledger unchanged");
        // a consumed segment whose file vanished: nothing lost, status recorded
        l.reserve(5).unwrap();
        assert_eq!(l.segments()[2].status, SegmentStatus::Consumed);
        assert_eq!(l.write_off("c", SegmentStatus::Missing).unwrap(), 0);
        assert_eq!(l.write_off("c", SegmentStatus::Quarantined).unwrap_err().code(), ErrorCode::MaskStore);
        l.verify().unwrap();
    }

    #[test]
    fn overflow_is_an_error_not_a_panic() {
        let segs = vec![
            SegmentState { segment_id: "a".into(), slots: u64::MAX, next_slot: 0, status: SegmentStatus::Active },
            SegmentState { segment_id: "b".into(), slots: 5, next_slot: 0, status: SegmentStatus::Active },
        ];
        let mut l = Ledger::from_parts(0, 0, segs).unwrap();
        assert_eq!(l.available(), u64::MAX);
        l.reserve(3).unwrap();
        let before = l.clone();
        // enough slots exist (2^64 + 1) but the consumed counter would overflow: refused, unchanged
        assert_eq!(l.reserve(u64::MAX).unwrap_err().code(), ErrorCode::MaskStore);
        assert_eq!(l, before);
        let mut g = Ledger::from_parts(u64::MAX, 0, vec![]).unwrap();
        g.add_segment("x", 1).unwrap();
        assert_eq!(g.reserve(1).unwrap_err().code(), ErrorCode::MaskStore);
    }

    #[test]
    fn overlapping_extents_are_refused() {
        let ov = vec![
            SlotExtent { segment_id: "a".into(), start: 0, count: 4 },
            SlotExtent { segment_id: "b".into(), start: 0, count: 4 },
            SlotExtent { segment_id: "a".into(), start: 3, count: 2 },
        ];
        assert_eq!(ReservationCursor::new(&ov, &["x"]).unwrap_err().code(), ErrorCode::MaskReuse);
        let adjacent = vec![
            SlotExtent { segment_id: "a".into(), start: 0, count: 4 },
            SlotExtent { segment_id: "a".into(), start: 4, count: 2 },
        ];
        assert_eq!(ReservationCursor::new(&adjacent, &["x"]).unwrap().slots(), 6);
        let wrap = vec![SlotExtent { segment_id: "a".into(), start: u64::MAX, count: 2 }];
        assert_eq!(ReservationCursor::new(&wrap, &["x"]).unwrap_err().code(), ErrorCode::MaskStore);
        let empty = vec![SlotExtent { segment_id: "a".into(), start: 0, count: 0 }];
        assert!(ReservationCursor::new(&empty, &["x"]).is_err());
    }

    #[test]
    fn closed_cursor_refuses_everything_with_mask_reuse() {
        let ext = vec![SlotExtent { segment_id: "a".into(), start: 0, count: 3 }];
        let mut c = ReservationCursor::new(&ext, &["x"]).unwrap();
        assert_eq!(c.close(), Some(3));
        for (lid, rows) in [("x", 1u64), ("unknown", 1), ("x", 0)] {
            assert_eq!(c.take(lid, rows).unwrap_err().code(), ErrorCode::MaskReuse, "{lid} {rows}");
        }
    }

    /// Exhaustive small-model check: for random interleavings of reserve / write-off, no slot is
    /// handed out twice, verify() holds after every step and each step is a valid successor.
    #[test]
    fn randomized_ledger_invariants() {
        let mut state = 0x5eed_u64;
        let mut rnd = |m: u64| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        for _round in 0..50 {
            let mut l = Ledger::new();
            let mut seen = std::collections::HashSet::new();
            for i in 0..8 {
                l.add_segment(&format!("s{i}"), 1 + rnd(9)).unwrap();
            }
            for _ in 0..30 {
                let prev = l.clone();
                match rnd(4) {
                    0 => {
                        let id = format!("s{}", rnd(8));
                        let _ = l.write_off(&id, SegmentStatus::Missing);
                    }
                    _ => {
                        if let Ok(r) = l.reserve(1 + rnd(6)) {
                            for e in &r.extents {
                                for s in e.start..e.start + e.count {
                                    assert!(seen.insert((e.segment_id.clone(), s)), "slot reused");
                                }
                            }
                        }
                    }
                }
                l.verify().unwrap();
                l.check_successor(&prev).unwrap();
            }
        }
    }

    #[test]
    fn cursor_burns_unused() {
        let ext = vec![SlotExtent { segment_id: "a".into(), start: 0, count: 10 }];
        let mut c = ReservationCursor::new(&ext, &["x", "y"]).unwrap();
        c.take("x", 3).unwrap();
        c.take("y", 6).unwrap();
        assert_eq!(c.close(), Some(4));
        assert!(c.is_closed());
        assert!(ReservationCursor::new(&[], &["x"]).is_err());
        assert!(ReservationCursor::new(&ext, &["x", "x"]).is_err());
        assert!(ReservationCursor::new::<&str>(&ext, &[]).is_err());
    }
}
