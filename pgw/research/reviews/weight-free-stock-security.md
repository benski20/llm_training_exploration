# Weight-free factory (option C): is a pre-drawn check stock as secret and as sound as on-demand draws?

Date: 2026-11-26. Scope: `factory_weights = "stream"` (opt-in; default `"pack"` unchanged) for the SDK client's
encrypted factory (`mask_scheme = "pks-v3"`, `check_field = "auto"`, masked head). Inputs: option C of
`weight-free-client-checks.md` (section 5), the stock argument of `pks-v3-client-disk.md` (section 3), SPEC 5.1 /
7.2 / 7.3 / 7.6, `factory/generator.py` (`generate`, `_private_check`, `_wipe_private`), `maskstore/format.py`,
`maskstore/keys.py`, `maskstore/state.py`, `maskstore/counters.py`. Written before the implementation; the
implementation follows it (`pgw.factory.checkstock`).

## 0. Verdict

**No property weakens beyond what the earlier analysis stated.** Every pre-drawn item is produced by the same code,
from the same CSPRNG, with the same derivation as an on-demand draw, and is used under the same single-use rules;
the bytes a segment carries are identical for the same seed. Two things change, and both were stated before
(`pks-v3-client-disk.md` section 3, `weight-free-client-checks.md` section 5):

1. **More secret material at rest.** Up to `K` (default 64) future segments' seeds and `uW`, and `P` private
   `v W8` sets, sit on disk under the store's KEK envelope. A snapshot of the disk *and* the KEK exposes them,
   where today it exposes the segments in the mask store. The attacker who can do that can already decrypt the
   seeds of every stored segment, and so its masks `r`: privacy is gone in that event today. The stock widens the
   window of that same event by `K` segments; it adds no new attacker.
2. **Private `v W8` on disk for the first time** (section 3): an integrity-only exposure under a condition that
   already implies a total privacy break.

Recovery after a suspected compromise differs operationally (redrawing needs a `W8` pass instead of the local
pack), not in kind: rotate the KEK and discard the stock (`pgw factory check-stock --discard`), as for the store.

## 1. What an item is, and why it equals an on-demand draw

| item | today (pack mode) | pre-drawn (stream mode) |
|---|---|---|
| segment set | `seed = os.urandom(32)` in `generate_into_store`; per linear `u = prime_check_vectors(seed, checks, m, key)` (p31, stream `"u-p31"`) or `SeedExpander(seed).ring_uniform((checks, m), K, "u", key)` (ring); `uW = u W8` (`prime_vecmat` / `ring_vecmat`) | the same calls, at pass time: `new_seed()` (`os.urandom`), the same `u` functions per linear, `uW` from the same kernels on `W8` streamed and SHA-256-checked against the pinned manifest. Several seeds' `u` are stacked into one matrix: the kernels are exact integer sums per row, so row `i` of the batched product equals the single-vector product bit for bit (tested) |
| private set | `v = SeedExpander(new_seed()).ring_uniform((c, m), K, "v")`, `vW = ring_vecmat(v, W8)` per linear, at worker start and after every failed or aborted segment | the same `v` from a fresh `new_seed()` kept as that 32-byte seed, `vW` from the same kernel; `c` from the same `_private_vectors` rule (40 bits against chosen errors). The generator rebuilds `v` from its seed when it takes the set |

* **Distribution.** Uniform 32-byte seeds from the OS CSPRNG in both cases; `u`, `r` and `v` are deterministic
  functions of them (SPEC 7.3). Drawing earlier changes no distribution.
* **Independence from the provider's view** (what the p31 soundness argument of SPEC 5.1 and the factory's
  self-check argument need). The provider never sees a seed, `u`, `v` or a product with them; it sees `W8` being
  downloaded (public data) at pass time. A pre-drawn value is independent of everything the provider has seen
  until it is used, exactly like a value drawn at the moment of use. The order "draw, then the provider acts" is
  preserved: the provider's products for a segment are requested after its seed exists in both modes.
* **Bound to the true `W8`.** A pass hashes every tensor against the manifest that the pinned fingerprint covers
  (provider endpoint: `_Remote.object_bytes`; public source: the locally packed checkpoint must reproduce the pinned
  fingerprint and each linear is loaded with `check_hash=True`). The masked head's `W8` is derived from the
  gateway pack's hash-checked plain head, as today. Nothing from a pass is committed until every linear passed:
  a wrong stream is a denial of service, never a wrong check vector.

## 2. Single use, crash safety, tamper and rollback

* **Single use.** A take records the consumption durably (index rewritten, fsync of the file and the directory)
  *before* the item is read; the item file is then removed. After a crash, an item no longer listed in the index is
  deleted at the next open, never handed out. A pass writes items under temporary names and lists them in the
  index only after all are synced, so a crash mid-pass leaves orphans that are deleted.
* **Same lifetime rules as today.** A segment item is used for exactly one segment; if that segment aborts, its
  seed is gone (today too: an aborted segment's seed is never reused). A private set is taken at worker start and
  after every `_wipe_private` (any failed or aborted segment), so every private set is used until its first
  failure or abort and never after, as today. A worker that crashes loses its set (it was recorded as taken).
  Distinct workers take distinct sets (today each draws its own).
* **Second guard for segment seeds.** The mask store keeps `seed_hash` UNIQUE forever (`maskstore/state.py`), so a
  segment seed that somehow came back is refused at commit.
* **Tamper.** Each item is AES-256-GCM under a fresh 32-byte DEK wrapped by the store's KEK provider
  (`KeyProvider.wrap`, the same envelope as segments), with the item header (kind, sequence number, fingerprint,
  ring, check field, linear list) as AAD and the header's SHA-256 recorded in the index. The index is
  `KeyProvider.mac`'d (HMAC-SHA256 under the KEK-derived key) over its canonical JSON with its own domain label
  `pgw/v1/check-stock-index`. A flipped bit, a swapped or renamed item, an item of another stock or kind, or an
  edited index fails authentication: the item is burned and the take fails closed; an index that does not verify
  makes the whole stock unusable (discarded, refilled by a new pass). A provider without `mac()` is refused.
* **Rollback** (restoring an older index and its item files, which needs no key). The index carries a generation
  number, mirrored after every change into a MAC'd witness file outside the stock directory; an index behind its
  witness is refused and the stock discarded. This is the mask store's own level (`counters.py`: the journal
  detects restoring the database alone; restoring both from one snapshot needs a monotonic counter). For segment
  seeds the store's permanent `seed_hash` refusal still holds after a full snapshot restore. For private sets a
  full snapshot restore could serve a set that an earlier, aborted segment used: the same class of event as
  restoring the store's database and journal together, which the store also leaves to a monotonic counter.

## 3. The one new exposure: private `v W8` on disk

Today the private vectors live only in the factory's RAM. Under stream mode they live in the envelope until taken.

* **What knowing `v` gives.** A provider that knows a worker's `v` can add an error `e` with `v e = 0 (mod 2^32)`
  for every vector, which the factory self-check then misses (integrity of masks). The p31 online check is
  untouched by `v` (it uses the segment's `u`), and since F4 it catches a wrong mask online with the same reaction
  for every input; so even then the error is caught at the gateway, not silently used.
* **Who can learn it.** Reading an item needs the KEK. The same KEK unwraps every segment in the mask store, whose
  chunk 0 holds the segment seed, from which `r` (the masks) follows. Anyone who can read `v` can therefore already
  unmask every stored activation. `v W8` adds nothing beyond `v` (`W8` is public), and `v` is stored as its seed.
* **Conclusion.** An integrity exposure under a condition that already breaks privacy completely. It is not a new
  attacker and not a weaker bound for any attacker without the KEK. Mitigations kept: the KEK envelope, 0600 files
  in a 0700 directory, a small `P` (default `max(8, 2 x workers + 2)`), the take-before-use rule.

## 4. What does not change

Soundness per check (p31 `2^-31`, private self-check `2^-40` against chosen errors), lattice parameters and noise,
the hint, the wire, the segment format and SPEC 7.3's derivation, what the provider sees per segment. The provider
additionally sees when a pass downloads `W8` (stream from its endpoint), which follows the factory's segment rate,
already visible to it through maskgen requests.

## 5. Costs (stated, not security)

Availability: no check material, no segment, no masks, no request (fail closed, clear error). Network: a full
`W8` per refill pass (`K - low water` segments per pass). Disk: the stock instead of `W8` (measured in STATUS.md).
