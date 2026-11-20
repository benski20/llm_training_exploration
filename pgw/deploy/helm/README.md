# Helm charts

| chart | runs at | contents |
|---|---|---|
| `pgw-gateway` | customer | Deployment (1 replica, Recreate: one pod per mask store), Service, state PVC (mask store, API keys, quotas, audit log), models ConfigMap, NetworkPolicy, optional Ingress / PDB / ServiceMonitor |
| `pgw-factory` | customer or partner | Deployment running `pgw factory run` (CPU or GPU), NetworkPolicy denying all traffic |
| `pgw-service` | us | Deployment (stateless, GPU), compute Service + separate metrics Service, PDB, optional HPA / ServiceMonitor, NetworkPolicy (no egress) |

Values never contain secrets. Every chart references **existing Secrets** (`existingSecret` values)
created with your secret manager (External Secrets, Secrets Store CSI, Vault, sealed secrets) or kubectl.
`<chart>/ci/ci-values.yaml` holds the minimal values used by CI.

## Customer side

```bash
# secrets: admin token, mask-store KEK (base64 of 32 random bytes), compute-service tenant token
kubectl create secret generic pgw-gateway-secrets \
  --from-literal=admin-token="$(openssl rand -hex 32)" \
  --from-literal=kek="$(openssl rand -base64 32)" \
  --from-file=service-token=./service-token          # issued by the PGW operator

helm install gw deploy/helm/pgw-gateway \
  --set secrets.existingSecret=pgw-gateway-secrets \
  --set modelPacks.existingClaim=pgw-packs \
  --set 'networkPolicy.egress.computeService.cidrs={10.0.12.34/32}' \
  -f my-models.yaml                                  # models: name, packDir, fingerprint, serviceUrl

helm install factory deploy/helm/pgw-factory \
  --set modelPacks.existingClaim=pgw-packs \
  --set store.existingClaim=gw-pgw-gateway-state \
  --set secrets.existingSecret=pgw-gateway-secrets \
  --set gpu.enabled=true --set image.repository=registry.example.com/pgw/pgw-factory-cuda
```

* The **KEK** is shared by the gateway and the factory (same store). Keep a backup of it: without it
  the store's masks are unusable (but never reusable: losing masks is safe, reusing them is not).
* The state PVC is annotated `helm.sh/resource-policy: keep`. **Never restore it from a snapshot**:
  an old store would allow mask reuse; the store's rollback protection refuses to serve.
* The factory (output=store) is always scheduled onto the gateway's node
  (`store.colocateWithGateway`). Never share a store across nodes with ReadWriteMany/NFS storage
  (SQLite WAL and flock are unreliable there; the store refuses to open on a network filesystem).
  A factory elsewhere uses output=export, and the gateway imports with `pgw maskstore import`.
* Rollback protection: `maskstore.counter=file` (default) keeps the monotonic counter on a separate
  `<release>-counter` PVC labelled `velero.io/exclude-from-backup`. Do not snapshot or back it up
  with the state volume. If the state volume is ever restored, burn every ACTIVE segment and rebind
  witnesses before serving again. `counter=provider` (KMS/HSM) or `tpm` are alternatives; `none` is
  refused unless `maskstore.allowNoExternalCounter=true` (dev only).
* Model packs: a PVC with pack directories (`pgw pack` output, verified with `pgw doctor --pack`), or
  an OCI image with the packs under `/packs` (`modelPacks.image`), copied by an init container.
* The gateway renders `/run/pgw/models.json` (tmpfs, 0600) from the ConfigMap template at start-up;
  the compute-service token is read from the mounted Secret file. `pgw doctor --role gateway` inside
  the pod checks the configuration.
* `networkPolicy.egress.computeService` is the gateway's only allowed destination besides DNS:
  the PrivateLink / Private Endpoint addresses or the service's public range.
* Operator console (`/console/`, on by default with `console.enabled`): it shares the API listener
  with `/admin/*` and signs in with the admin token. Read-only viewer tokens come from
  `console.viewerTokensKey` (a key of `secrets.existingSecret`, mounted as a file and re-read on
  change). Keep `/console` and `/admin` off the application ingress: route only `/v1` through
  `ingress` and publish the console with `consoleIngress` on an internal ingress class (TLS
  required, own host name, no path rewriting). Without an ingress, use
  `kubectl port-forward svc/<release>-pgw-gateway 8080:8080` and `http://localhost:8080/console/`.
  See [docs/operations/console.md](../../docs/operations/console.md).

## Operator side (compute service)

```bash
pgw service token create --tenant acme --tokens-file tokens   # prints the token once; give it to acme
kubectl create secret generic pgw-tenants --from-file=tokens=./tokens
kubectl create secret generic pgw-service-tls \
  --from-file=tls.crt --from-file=tls.key --from-file=ca.crt=customer-client-ca.crt

helm install compute deploy/helm/pgw-service \
  --set modelPacks.existingClaim=pgw-packs --set tenants.existingSecret=pgw-tenants \
  --set tls.existingSecret=pgw-service-tls \
  --set 'networkPolicy.computeFrom[0].ipBlock.cidr=0.0.0.0/0'   # restrict to the LB / customer ranges
```

* With mTLS (`tls.requireClientCert`, default) kubelet probes use TCP checks because they cannot
  present client certificates.
* The service has **no egress** at all; packs are mounted read-only and verified (`verifyPacks: full`).
* `terminationGracePeriodSeconds` exceeds the drain timeout, so rolling updates finish in-flight work.

## Placement and CPU (latency)

Every generated token makes 112 round trips to the compute service one after another, with gateway
masking work between them, so the gateway is latency-bound on a few cores:

* **Gateway: Guaranteed QoS.** The default `resources` have integer CPU requests equal to the limits
  and memory requests equal to the limits. Keep it that way. On nodes whose kubelet runs with
  `cpuManagerPolicy: static` (kubelet config `cpuManagerPolicy: static` plus a `reservedSystemCPUs` or
  `kubeReserved` CPU reservation; on EKS/GKE/AKS set it in the node pool's kubelet configuration), a
  Guaranteed pod with integer CPUs gets exclusive cores, so the masking work is not preempted by the
  factory or other pods. Set `priorityClassName` to a PriorityClass you create for the gateway.
* **Factory: hard CPU limit.** The factory's `resources.limits.cpu` equals its request. With
  `output: store` it must run on the gateway's node (the store volume is ReadWriteOnce); the CPU
  limit and the gateway's exclusive cores keep them apart. With `output: export` it can run elsewhere:
  `separateFromGateway: preferred` (or `required`) adds a pod anti-affinity to gateway nodes.
* **Zones.** Run the gateway in the zone of the compute-service endpoint it calls (same-zone round
  trips add about 0.01-0.035 s per token, cross-zone about 0.05-0.11 s; estimates). The gateway's
  `nodeSelector` comment in values.yaml shows `topology.kubernetes.io/zone`; the service chart's
  `topologySpreadConstraints` comment shows a zone spread for its replicas.
* **Client IPs behind a proxy.** `trustedProxyCidrs` (gateway and service) sets
  `PGW_TRUSTED_PROXY_CIDRS`: only peers in these ranges may set the client IP with X-Forwarded-For.
  Empty (default) trusts nobody.

## Alerts

With the chart's ServiceMonitor enabled (`serviceMonitor.enabled` for the gateway,
`metrics.serviceMonitor.enabled` for the factory and the service) each chart also renders a
`PrometheusRule` (turn off with `alerts.enabled: false`; add `alerts.labels` to match your Prometheus
operator's `ruleSelector`). The same rules for a plain Prometheus are in
[deploy/monitoring/pgw-alerts.yml](../monitoring/pgw-alerts.yml):

| alert | chart | severity | fires when |
|---|---|---|---|
| PGWMaskRunwayLow | gateway | warning | masks left / consumption rate < `alerts.runwayMinSeconds` (default 7200 s) |
| PGWMaskStoreEmpty | gateway | page | no masks left for 5 m |
| PGWMaskStoreBelowLowWater | gateway | warning | below low water for 30 m |
| PGWMaskBurnRatioHigh | gateway | warning | burned / consumed slots > `alerts.burnRatioMax` over 1 h |
| PGWIntegrityCheckFailed | gateway | page | any Freivalds or range check failure |
| PGWNoiseAlarm | gateway | page | the outbound-noise monitor alarms |
| PGWModelNotReady | gateway | warning | a model not ready for 10 m |
| PGWFactoryStalled | factory | warning | no plan or segment for `alerts.factoryStallSeconds` |
| PGWFactoryNotRefilling | factory | warning | below low water and nothing generated in 1 h |
| PGWFactoryErrors | factory | warning | more than 2 failed runs in 30 m |
| PGWServiceErrorRatioHigh | service | warning | error share > `alerts.errorRatioMax` over 10 m |

Mask-supply alerts aggregate by `(fingerprint, ring)`. Set `alerts.runwayMinSeconds` above the time
one segment takes to arrive (`pgw doctor` prints the measured mask-supply rate), with margin.

## Testing the charts

`tests/cli/test_deploy_helm.py` renders every chart in many value combinations (with a small Go-template
renderer when `helm` is not installed), validates the manifests, checks the security context, probes,
selectors and network policies, and checks that every `PGW_*` variable is a real setting that parses.
CI additionally runs `helm lint --strict`, `helm template` and kubeconform. `tests/deploy/test_alert_rules.py`
checks the alert rules (every metric exists in `METRIC_NAMES`, the charts match the standalone file, and
`promtool check rules` when promtool is installed) and the placement values above.
