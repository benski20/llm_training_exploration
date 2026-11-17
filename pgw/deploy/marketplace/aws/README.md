# AWS Marketplace packaging notes

Two listings cover the customer side; the compute service stays in our AWS accounts.

| listing | delivery | contents |
|---|---|---|
| **PGW Gateway (Kubernetes)** | Container product (Helm, EKS) | `pgw-gateway` and `pgw-factory` images in the Marketplace ECR, charts from `deploy/helm` |
| **PGW Gateway (AMI)** | AMI product (single instance) | Ubuntu 24.04 AMI built with `deploy/systemd/install.sh` (gateway + factory units) |
| PGW Mask Factory GPU (optional) | AMI product (g4dn / g5 / g6) | `install.sh --role factory --cuda cu124`, for customers running their own factory |

## Compute service connectivity (Topology B)

* We publish the compute service as a **VPC endpoint service (PrivateLink)** behind an NLB in each
  supported region. Customers create an interface endpoint in their VPC, in the same region (RTT
  about 1 ms, so roughly 0.13 s per token for 8B at 128 round trips).
* The NLB passes TCP through; the service terminates TLS with mutual authentication (customer client
  certificates from our CA) plus a per-tenant bearer token.
* Gateway Helm values: `models[].serviceUrl` = the endpoint's private DNS name;
  `networkPolicy.egress.computeService.cidrs` = the endpoint ENI addresses (/32 each).
  The gateway then has no other egress path. On the AMI, the instance security group allows
  outbound 443/8443 to the endpoint's security group only (plus DNS via the VPC resolver).

## Keys

* Mask store KEK: v1 supports `file`, `env` and `pkcs11` key providers; the `kms` provider (AWS KMS
  envelope encryption with an HMAC journal key) is a stub in v1 (SPEC 7.6). Until it ships, store the
  KEK in AWS Secrets Manager and deliver it as a Kubernetes Secret through the Secrets Store CSI driver
  or External Secrets (chart: `maskstore.keyProvider: env`, `secrets.existingSecret`). On the AMI use
  a root-only file delivered by SSM Parameter Store (SecureString) at first boot.
* Use an encrypted gp3 EBS volume (customer-managed KMS key) for `/var/lib/pgw`. Do not restore the
  mask-store volume from snapshots (mask reuse); `pgw doctor --maskstore` detects rollback.
* IRSA (EKS) / instance profile: no AWS permissions are needed except for KMS/Secrets Manager reads.

## Metering and licensing

* Container product: hourly or per-request pricing through the AWS Marketplace Metering Service
  (`RegisterUsage` at gateway start-up, requires the IRSA role from the listing). Implement as an
  optional `PGW_GATEWAY_*` integration before listing (tracked in INTERFACE_NOTES).
* AMI product: hourly pricing by instance type, no in-product metering.
* Usage of the compute service is billed by us per tenant (`pgw_service_*` tenant counters).

## Listing requirements checklist

* Images: non-root (uid 10001), read-only root filesystem, no secrets or default passwords, no SSH
  password login on AMIs, IMDSv2 required, latest OS patches at submission, CVE scan clean
  (Marketplace scans every version).
* Helm: charts must install with Marketplace ECR image URIs via values (`image.repository`), no
  external image pulls; provide `values.yaml` documentation and a usage guide (deploy/helm READMEs).
* AMI: `cloud-init` first-boot configuration from user data or SSM (mirror of
  `deploy/vmware/pgw-ovf-config.sh`), EBS-backed, ENA enabled, single root volume plus data volume.
* Support and EULA documents; the EULA must state the threat model (SPEC section 3): the service
  operator never receives masks; the partner factory must be independent of the operator.

## Build pipeline

1. CI builds and scans images (`.github/workflows/ci.yml`), pushes to our ECR, then to the
   Marketplace ECR repositories (`aws marketplace-catalog start-change-set` with `AddDeliveryOptions`).
2. AMI: Packer `amazon-ebs` builder running the same provisioning steps as
   `deploy/vmware/pgw-gateway.pkr.hcl`, then share with the Marketplace scanning account.
3. Release notes include image digests, chart versions and model-pack fingerprints of the supported
   models.
