# PGW deployment

Everything needed to run PGW outside a developer checkout. All artifacts use the same `pgw` CLI
(`pgw --help`) and the same `PGW_*` settings (`pgw config show <role>`), and every one of them keeps
the trust boundaries of SPEC section 3: the compute service never gets masks, seeds or keys, the
factory needs no network, and the gateway may talk only to the compute service.

| directory | what | for |
|---|---|---|
| [docker/](docker/) | Dockerfiles: `gateway` (CPU), `factory` (CPU), `factory-cuda`, `service-cuda`, `service-cpu` (demo/CI), `provider` (CUDA base, service + SDK-key accounts + native extension + pack tool + RunPod boot) and `provider-cpu` (tests, client test pod); `pgw-entrypoint`, `pgw-healthcheck`, `provider-smoke.sh` | all container deployments |
| [runpod/](runpod/README.md) | provider on RunPod pods: launcher with cost cap, TTL self-stop + watchdog, SDK key bootstrap, client pod in the same data center, `rtt_probe.py` | our hosted provider |
| [compose/](compose/README.md) | docker-compose demo: pack -> factory -> service -> gateway on one machine | evaluation, development |
| [helm/](helm/README.md) | charts `pgw-gateway`, `pgw-factory`, `pgw-service` | Kubernetes (customer cluster, our cluster) |
| [systemd/](systemd/) | hardened units, env examples, `install.sh` | VMs and bare metal; base of the OVA/AMI images |
| [vmware/](vmware/README.md) | Packer template, OVF properties, first-boot script, OVA notes | on-prem appliance (Topology A) |
| [marketplace/aws](marketplace/aws/README.md), [marketplace/azure](marketplace/azure/README.md) | listing and packaging notes | cloud marketplaces (Topology B) |

CI (`../.github/workflows/ci.yml`) lints, type-checks and tests the code, lints and renders the
charts, validates the compose file, lints Dockerfiles and shell scripts, and builds every image.
`../ci/check.sh` runs the same checks locally.

## Images

Build from the `pgw/` directory (the build context); each Dockerfile has an allowlist
`<name>.Dockerfile.dockerignore`:

```bash
docker build -f deploy/docker/gateway.Dockerfile      -t pgw-gateway .
docker build -f deploy/docker/factory.Dockerfile      -t pgw-factory .
docker build -f deploy/docker/factory-cuda.Dockerfile -t pgw-factory-cuda .
docker build -f deploy/docker/service-cuda.Dockerfile -t pgw-service-cuda .
docker build -f deploy/docker/provider.Dockerfile     -t pgw-provider .       # hosted provider (RunPod)
docker build -f deploy/docker/provider-cpu.Dockerfile -t pgw-provider-cpu .   # + deploy/docker/provider-smoke.sh
```

The `provider` images (`.github/workflows/provider-image.yml` pushes them to GHCR on a `pgw-provider-v*` tag) build
the native extension `pgw_fast` in a builder stage with a pinned Rust toolchain, install the hash-locked
`requirements/provider-*.txt` (service + pack tool, the same versions as the service and gateway locks), and start
as root only to hand the data directory (or, under `pgw-runpod-boot`, the RunPod volume) to uid 10001; every PGW
process runs as 10001 with `no_new_privs`.

All images: multi-stage, Python 3.12 slim, non-root uid 10001, read-only-root compatible, JSON logs,
no secrets. `pgw-entrypoint` renders config templates listed in `PGW_RENDER=src:dst[,src:dst]`
(`${env:NAME}` / `${file:PATH}` references, output 0600) and then execs the command, so secrets live
in Secret objects or files and never in ConfigMaps, images or command lines. The CUDA images use the
CUDA wheels of torch (no CUDA base image needed); hosts need the NVIDIA driver and container toolkit.

## Network policy summary

| component | ingress | egress |
|---|---|---|
| gateway | API clients (configurable), Prometheus | DNS + the compute service only (`networkPolicy.egress.computeService`) |
| factory | none | none (optionally DNS / KMS endpoint) |
| service | gateways on the compute port, Prometheus on the metrics port | none |

Kubernetes NetworkPolicy objects implement this in the charts; the compose demo uses an internal
network and `network_mode: none`; the VM image uses nftables and systemd `PrivateNetwork=`.

## Choosing a topology (SPEC section 13)

* **A, on-prem gateway**: OVA (`vmware/`) or the gateway chart in the customer's cluster; masks from a
  local factory (`pgw-factory` chart or `pgw-factory.service`) or a partner.
* **B, customer cloud account**: gateway chart / marketplace listing in the same region as our service,
  connected over PrivateLink / Private Link.
* **C, partner factory**: `pgw-factory` chart with `output: export` at an independent partner; the
  customer imports segments with `pgw maskstore import`.
