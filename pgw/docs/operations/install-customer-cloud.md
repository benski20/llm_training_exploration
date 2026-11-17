# Operator guide: gateway in your cloud account (topology B)

This guide is for customers who run the gateway inside their own AWS, GCP or Azure account, in the
same region as our compute service, connected over a private link. This is the lowest-latency
topology. With about 1 ms RTT, the round-trip floor is about 0.13 s/token for an 8B model
(*estimate*: 128 round trips × RTT). Read [install-on-prem.md](install-on-prem.md) first: the
software, configuration and operations are the same. This page covers what differs.

```
 your cloud account (VPC)                                  our account, same region
 ┌──────────────────────────────────────────────┐          ┌──────────────────────────┐
 │ apps ─▶ pgw-gateway (Kubernetes / VM)         │ Private- │ pgw compute service      │
 │            ▲            │ x̂ (mTLS)           ├─ Link ──▶│ (behind an endpoint      │
 │   encrypted volume       └───────────────────┼─ / PSC ─▶│  service)                │
 │   (mask store, KMS key)                      │          └──────────────────────────┘
 │   GPU node pool: pgw factory (optional)      │
 └──────────────────────────────────────────────┘
```

## 1. Network

| cloud | mechanism | notes |
|---|---|---|
| AWS | Interface VPC endpoint to our endpoint service (PrivateLink) | We allow-list your AWS account id. Private DNS name supplied by us |
| GCP | Private Service Connect endpoint to our service attachment | We allow-list your project |
| Azure | Private Endpoint to our Private Link Service | We approve the connection request |

* No internet egress is needed for the gateway. Deny egress by default and allow only the private
  endpoint (security group, firewall rule or NSG). This is your independent control against a
  malicious build (whitepaper section 9).
* TLS still applies over the private link: set `service_ca_file` to our CA bundle and keep
  `require_service_tls` on.
* Bandwidth: an 8B model at K48A20 moves 13.4 MB per token, both directions together (*analytic*;
  11.1 MB at K40A12). Check endpoint and NAT throughput limits and data-processing charges
  against the sizing figures.

## 2. Kubernetes (Helm)

Charts live in `deploy/helm/`: `pgw-gateway` and `pgw-factory`. (`pgw-service` is ours.)

```bash
kubectl create namespace pgw
kubectl -n pgw create secret generic pgw-gateway-secrets \
  --from-file=admin-token=./admin-token --from-file=service-token=./service-token \
  --from-file=service-ca.pem=./service-ca.pem
helm install pgw-gateway deploy/helm/pgw-gateway -n pgw -f my-values.yaml
```

Rules the chart follows. Check them in the rendered manifests (`helm template`) and do not override them:

* `replicaCount: 1` with the `Recreate` strategy. Each replica owns its mask store, because masks
  are single use. Scale out with one release per replica, each with its own volume.
* The pod runs as a non-root user with a read-only root filesystem and all capabilities dropped.
* Secrets are referenced from existing Kubernetes Secrets (or an external secrets operator), never
  written into values.
* The NetworkPolicy allows egress only to the service endpoint and DNS, and limits `/metrics` to
  the monitoring namespace.

## 3. Mask store on cloud storage

* Use a dedicated block volume (EBS gp3/io2, GCP PD-SSD/Hyperdisk, Azure Premium SSD) encrypted with
  **your** customer-managed key. Do not use shared file systems: the store relies on POSIX locking
  and fsync.
* Key provider: `PGW_MASKSTORE_KEY_PROVIDER=kms` with `PGW_MASKSTORE_KMS_KEY_URI` (for example
  `aws-kms://arn:aws:kms:...`). Grant the gateway's workload identity (IRSA, GKE Workload Identity
  or Azure workload identity) only `Encrypt`/`Decrypt` on that key. In v1 the KMS provider is a stub
  (SPEC 7.6): until your build lists `kms` as supported in `pgw doctor`, use `pkcs11` (CloudHSM,
  Cloud HSM, Managed HSM) or a `file` KEK mounted from your secret manager.
* Rollback protection: set `PGW_MASKSTORE_KMS_MAC_KEY` to enable the HMAC journal **and** configure
  a monotonic counter that does not live on the mask volume: `PGW_MASKSTORE_COUNTER=provider` (HSM)
  or `PGW_MASKSTORE_COUNTER=file` with `PGW_MASKSTORE_COUNTER_FILE` on a second volume that is never
  snapshotted together with the mask volume. The journal alone catches a restored `state.sqlite`,
  but **not** a revert of the whole volume: the database and the journal come back together and
  look consistent, and slots would be handed out a second time (threat model T-22). With the `kms`
  provider and no MAC key, the journal is off too. Disable snapshot restore for the mask volume in
  your runbooks and policies. A restore detected by a counter causes downtime, never reuse; recovery
  is a new, empty store.
* Size the volume from [sizing.md](sizing.md). Disk throughput matters: each 8B token reads about
  8.3 MB of masks (K48A20), so 100 tokens/s needs about 830 MB/s of sustained reads.

## 4. Mask factory in your account

A GPU node pool (one T4 or better) in the same account runs `pgw-factory` with access to the store
volume (same node, or the factory exports into a staging directory that the gateway imports). The
factory pod needs no network egress at all. A measured T4 rate of 550k slots/h for 8B at K48A20
covers about 13 million tokens/day per GPU at full duty.
Use spot instances: the factory is a batch job that keeps the inventory between low water and target,
and loses only the segment in progress if interrupted (an aborted segment is never committed).

Customers without GPUs use a partner factory instead ([air-gapped-factory.md](air-gapped-factory.md#4-partner-delivery-hpke)).

## 5. Marketplace images

Marketplace listings for AWS and Azure (in preparation, `deploy/marketplace/`) will bundle the same `pgw-gateway` image and charts. Verify the image
digest against the release notes before use.

## 6. Checklist

- [ ] Private endpoint up; `pgw doctor --role gateway --service-url ...` passes
- [ ] Egress denied except the endpoint; DNS resolves the private name
- [ ] Mask volume encrypted with a customer-managed key; snapshot restore disabled in runbooks
- [ ] Key provider, HMAC journal and a monotonic counter off the mask volume configured; `pgw maskstore status` clean
- [ ] Factory (or partner import) keeps `pgw_masks_available_slots` above low water
- [ ] `/admin/status` shows `production_policy: true`
- [ ] Alerts wired: low water, integrity failures, noise alarm, readiness
