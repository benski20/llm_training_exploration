# Azure Marketplace packaging notes

| offer type | contents |
|---|---|
| **Azure Container (Kubernetes application)** | CNAB bundle with the `pgw-gateway` and `pgw-factory` Helm charts and images (AKS) |
| **Azure Virtual Machine** | Ubuntu 24.04 image with the gateway + factory units (`deploy/systemd/install.sh`) |
| Azure Managed Application (optional) | ARM template that deploys the VM image or the AKS app plus a Private Endpoint, with `createUiDefinition.json` |

The compute service runs in our Azure subscriptions.

## Connectivity

* We expose the compute service through **Azure Private Link Service** behind a Standard Load Balancer
  per region. Customers create a Private Endpoint in their VNet; set `models[].serviceUrl` to its
  private DNS name and `networkPolicy.egress.computeService.cidrs` to the endpoint's private IP (/32).
* On AKS, network policies require Azure CNI with Azure Network Policy Manager or Cilium; Calico also
  works. Without a policy engine the gateway's in-process egress guard still applies, but add an NSG
  rule limiting the node pool's outbound traffic.
* mTLS client certificates for the gateway are issued by our CA and delivered as a Key Vault
  certificate, mounted with the Secrets Store CSI driver into the secret referenced by
  `serviceClientTls.existingSecret`.

## Keys

* The `kms` key provider (Azure Key Vault / Managed HSM) is a stub in v1 (SPEC 7.6). Until it ships,
  keep the 32-byte KEK as a Key Vault secret, sync it into the Kubernetes Secret named in
  `secrets.existingSecret` (Secrets Store CSI driver with `secretObjects`), and use
  `maskstore.keyProvider: env`. For HSM-backed keys use `pkcs11` with the Managed HSM PKCS#11 library.
* Use Azure Disk with customer-managed keys (disk encryption set) for the state volume. Do not restore
  the mask store from disk snapshots (mask reuse); the store's generation journal detects rollback.
* Workload Identity (AKS) for Key Vault access; no other Azure permissions are needed.

## Kubernetes application offer

* Package: `cpa` (container package app) tool builds the CNAB from a `manifest.yaml` that references
  `deploy/helm/pgw-gateway` and `deploy/helm/pgw-factory`, plus `createUiDefinition.json` and
  `mainTemplate.json` (cluster extension resource). Image references must use the `global.azure.images`
  value convention, so add an overlay values file that maps `image.repository`/`image.tag` to it.
* Billing: flat monthly or per-core through the cluster extension; usage of the compute service is
  billed by us per tenant.
* Validation: `cpa verify`; images must be non-root, scanned, and pulled only from the publisher ACR.

## VM offer

* Build with Packer `azure-arm` using the provisioning steps in `deploy/vmware/pgw-gateway.pkr.hcl`;
  first-boot configuration from custom data / cloud-init (equivalent of `pgw-ovf-config.sh`), with
  secrets fetched from Key Vault through the VM's managed identity.
* Generalize with `waagent -deprovision+user`; no default passwords or SSH keys in the image.
* Data disk for `/var/lib/pgw` (packs and mask store) sized from `pgw inspect` (bytes per token slot).

## Certification checklist

* Security: non-root containers, read-only root filesystems, no secrets in images or templates, CVE
  scan clean, TLS 1.2+ only, no outbound internet requirement beyond the Private Endpoint.
* Documentation: deployment guide, threat model (SPEC section 3), support contacts, EULA stating that
  the service operator never receives masks and that partner factories are independent.
