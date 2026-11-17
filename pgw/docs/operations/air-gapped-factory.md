# Operator guide: mask factory air-gapped from us

Privacy against the compute service depends on one organisational fact: **nobody who sees `x̂`
(us) may ever obtain the mask seeds** (whitepaper section 8). This guide shows how to run the mask
factory so that it has no network path to us at all. There are two variants:

* **4.1 Customer-run, offline.** Your factory host has no network connection. It writes directly
  into the mask store, or into files carried to the gateway host.
* **4.2 Independent partner.** A third party that is contractually independent of us generates
  masks on its own GPUs and delivers them sealed (HPKE) to your public key. The partner never holds
  your store KEK, and we never hold anything.

Both variants use the same software (`pgw factory`) and the same model pack, bound by the pack's
SHA-256 fingerprint.

## 1. What the factory needs, and what it must never have

| needs | must never have |
|---|---|
| The model pack, `linears` shards only (`pack.export(out, ["linears"])` or the full pack) | network routes to the compute service or to us |
| The mask store root and KEK (variant 4.1), or the customer's X25519 **public** key (variant 4.2) | the customer's store KEK (variant 4.2) |
| A GPU (or CPU) and local disk | credentials of the compute service |

## 2. Host preparation (both variants)

1. Install from verified media: the signed `pgw-factory` image (or `factory-cuda` for GPUs), or a
   wheel built from reviewed source. Record the image digest (audit checklist S-03).
2. Disable all network interfaces except, optionally, a management interface that cannot route to
   the internet. For containers, use `--network none` (the compose demo runs the factory this way).
3. Check arithmetic and the CSPRNG: `pgw doctor --role factory --pack /data/packs/llama8b`.
4. Measure your rate: `pgw factory benchmark --pack /data/packs/llama8b --ring K40A12`.

Expected rates (from [measurements.json](../measurements.json)):

| hardware | 8B, K48A20 | 70B, K48A20 | basis |
|---|---|---|---|
| NVIDIA T4 | 0.0066 s/slot (550k slots/h) | 0.047 s/slot (77k slots/h, weights streamed from host RAM) | measured |
| NVIDIA T4, K32A8 | 0.0043 s/slot | 0.032 s/slot | measured |
| 4 vCPU Xeon, fbgemm int8 at 0.30 TOPS | 0.28 s/slot (GEMM time only) | 2.8 s/slot | derived from measured TOPS |
| Laptop NPU, 40 TOPS | 3-7 ms/slot | — | estimate, untested |

## 3. Variant 4.1: customer-run offline factory

### 3.1 Same host as the store (simplest)

```bash
pgw factory run --pack /data/packs/llama8b --ring K40A12 \
    --store /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek --nice 10
```

The gateway host is networked and the factory process is not. Run the factory in a separate container
or VM with `--network none` that shares only the store volume.

### 3.2 Separate offline host (sneakernet)

The offline factory exports sealed segment files, which are carried to the gateway host on
removable media. Use the HPKE mechanism from section 4 in-house as well. Create the recipient key
pair on the gateway host (section 4.1), copy only the **public** key to the offline host, then:

```bash
# offline host: no store KEK needed, segments are sealed to the gateway's public key, signed by the
# factory's own key (its public key is in the gateway's PGW_MASKSTORE_TRUSTED_PARTNER_KEYS) and bound
# to the gateway's store (store_id from `pgw maskstore status` on the gateway host)
pgw factory generate --pack /data/packs/llama8b --ring K40A12 --slots 4096 --segments 32 \
    --export-dir /media/outbox --recipient-key /data/keys/gateway.pub.hex \
    --partner-key /data/keys/factory-sign.key --destination-store <gateway store_id>
# gateway host
pgw maskstore import --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek /media/outbox/*.pgwm
```

The media then carry only sealed segments. Losing them reveals nothing to anyone without the
gateway's recipient private key.

Media handling:

* Each segment file is encrypted and AEAD-protected. Tampering is detected on import.
* Use write-once or freshly wiped media. Log each transfer (who, when, segment ids) in the audit
  evidence (audit checklist M-04).
* After a successful `pgw maskstore import` (it removes each imported file and its `.sig`), wipe the media.

## 4. Partner delivery (HPKE)

### 4.1 Customer: create the recipient key pair (once)

```bash
pgw maskstore recipient-key create /etc/pgw/secrets/recipient.key --public-out /etc/pgw/recipient.pub.hex
# prints key_id, kek_id ("hpke:...") and public_key_hex; the private key is written with mode 0600
```

Set `PGW_MASKSTORE_RECIPIENT_KEY_FILE=/etc/pgw/secrets/recipient.key` on the gateway host. Give
`recipient.pub.hex` to the partner over an authenticated channel, and confirm the key id out of band
(for example by phone). The public key is not secret, but a substituted one would let someone else
read the masks.

### 4.2 Partner: signing key, generate and export

HPKE seals each segment to the customer but does not say who produced it: anyone who knows the
customer's public key could craft a segment from a seed they know. The partner therefore signs every
segment with an Ed25519 key, and the customer accepts only signatures from trusted partner keys.

```bash
# once: the partner's signing key; give partner.pub.hex to the customer over an authenticated channel
pgw maskstore partner-key create /keys/partner-sign.key --public-out /keys/partner.pub.hex
pgw factory run --pack /data/packs/llama8b --ring K40A12 \
    --export-dir /export/customer-a --recipient-key /keys/customer-a.pub.hex \
    --partner-key /keys/partner-sign.key --destination-store <customer store_id>
```

Each segment `<id>.pgwm` in the outbox gets a sidecar `<id>.pgwm.sig`, which the factory writes
before the segment appears. Deliver both files. The signed header names the customer's mask store
(`--destination-store`, the `store_id` the customer reads from `pgw maskstore status` and sends with
the public key); no other store will import it. One recipient key pair per store: never share it.

Each exported segment's data key is sealed with HPKE (RFC 9180: DHKEM X25519, HKDF-SHA-256,
AES-256-GCM) to the customer's key, with `kek_id = "hpke:<key id>"`. Contractually, the partner
deletes exported files after confirmed delivery and keeps no seeds. Seeds exist only inside the
sealed segment.

Partner independence requirements (put them in the contract; audit annually):

- [ ] separate legal entity, ownership and management from the service operator
- [ ] no shared staff, infrastructure, cloud accounts or administrative credentials with the service operator
- [ ] no copies of seeds or segments after delivery; deletion logs available to the customer
- [ ] factory hosts without network routes to the service operator
- [ ] notification duty for any approach by the service operator or third parties asking for masks

### 4.3 Customer: import

```bash
pgw maskstore import --root /var/lib/pgw/masks --key-file /etc/pgw/secrets/kek /srv/incoming/*.pgwm
```

Before the first import, set `PGW_MASKSTORE_TRUSTED_PARTNER_KEYS` to the partner's public key
(`partner.pub.hex`, confirmed out of band like the recipient key). A segment without a valid
signature from one of them is refused, and without trusted keys every partner segment is refused
(`PGW_MASKSTORE_ALLOW_UNSIGNED_PARTNER_SEGMENTS` exists for development only). Segments not bound to
this store (`destination`) and segments with fewer than 20 bits of Freivalds soundness are refused
too. Keep each `.sig` next to its segment when importing; after a successful import both files are
removed (`--keep-source` keeps them).

The import verifies the header, the fingerprint, the ring and **every chunk's** AEAD tag. It then
re-wraps the data key under your local KEK. A segment sealed to another recipient, or with any
corrupted byte, is rejected and reported per file.

## 5. Capacity planning for an offline supply

The offline supply must cover the longest gap between deliveries plus a safety margin:

```
store_slots ≥ tokens_per_day × (delivery_interval_days + safety_days)
store_bytes ≈ store_slots × mask_bytes_per_slot     (6.9 MB for 8B K40A12, 8.3 MB at K48A20)
```

Example (*analytic*): 200,000 tokens/day of an 8B model at K40A12 with weekly delivery and a 3-day
margin needs 2 million slots, about 13.8 TB. Mask volume is large. A customer-side GPU factory
on the gateway site usually beats shipping media at these volumes. Use
`docs/tools/pgw_sizing.py` for your numbers.

## 6. Failure modes

| symptom | cause | action |
|---|---|---|
| `fingerprint_mismatch` on import | partner used a different pack | compare `pgw inspect --verify full` on both sides |
| import fails for one file | corruption or wrong recipient | re-deliver that segment; others are unaffected |
| `mask_exhausted` at the gateway | supply below demand | emergency in-house generation on any GPU or CPU, or shorten the delivery interval |
| factory host reachable from the internet | configuration drift | treat as an incident; rotate the recipient key if the partner side was affected |
