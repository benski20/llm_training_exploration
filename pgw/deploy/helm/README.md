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
* With a ReadWriteOnce state volume the factory is scheduled onto the gateway's node
  (`store.colocateWithGateway`); with ReadWriteMany storage it can run anywhere.
* Model packs: a PVC with pack directories (`pgw pack` output, verified with `pgw doctor --pack`), or
  an OCI image with the packs under `/packs` (`modelPacks.image`), copied by an init container.
* The gateway renders `/run/pgw/models.json` (tmpfs, 0600) from the ConfigMap template at start-up;
  the compute-service token is read from the mounted Secret file. `pgw doctor --role gateway` inside
  the pod checks the configuration.
* `networkPolicy.egress.computeService` is the gateway's only allowed destination besides DNS:
  the PrivateLink / Private Endpoint addresses or the service's public range.

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

## Testing the charts

`tests/cli/test_deploy_helm.py` renders every chart in many value combinations (with a small Go-template
renderer when `helm` is not installed), validates the manifests, checks the security context, probes,
selectors and network policies, and checks that every `PGW_*` variable is a real setting that parses.
CI additionally runs `helm lint --strict`, `helm template` and kubeconform.
