# PGW in the customer's own cloud account (standard deployment)

The standard deployment runs everything that touches plaintext or mask secrets **in the customer's own cloud
account**, in the same region (and zone) as the provider's compute service:

| VM | runs | sees | needs |
|---|---|---|---|
| **Gateway** (CPU only) | `pgw gateway serve` (OpenAI-compatible client API, TLS), the encrypted mask store, for <= 14B also the CPU encrypted factory | prompts, outputs, the gateway's KV cache, masks | CPU cores and memory bandwidth, an encrypted volume |
| **GPU mask service** (optional; standard from 32B) | `pgw factory mask-service`: one-time masks `W8 r mod 2^K` computed on the customer's GPU from the customer's own copy of the public model pack | the public weights, the per-segment seeds and masks it produces | one small GPU (L4 or A10G), the public pack |
| premises | a thin client (any OpenAI client) pointed at the gateway's client API | its own prompts and outputs | network path to the VPC (VPN / Direct Connect) |

The provider runs only the compute service and sees masked activations, which are uniformly random to it.

## What the customer controls, and why privacy holds

* **The provider never gets `r`, the segment seeds or the check vectors.** Seeds are drawn from `os.urandom` inside
  the mask-service process, `r` and `u` are expanded from them on the customer's GPU, and every segment is
  encrypted (AES-256-GCM) before it is written. A segment sent to a gateway on another VM is HPKE-sealed to the
  gateway store's recipient key, signed by the mask service's Ed25519 key and bound to that one store
  (SPEC 7.7); the gateway refuses anything else.
* **The weights the mask service uses are public.** `W8` is the provider's model already (the enrolled pack
  fingerprint pins it); computing `W8 r` needs no help from the provider: no hint, no ciphertexts, no provider
  round trips. Nothing the mask service does depends on the provider's behaviour.
* **Every VM, key and volume is in the customer's account**: KMS keys, EBS volumes, the SSM parameters, the
  security groups and the instance roles. The mask service has no route to the provider (its security group
  allows inbound 8470 from the gateway only; its outbound is setup HTTPS for the registry and Hugging Face).
* **Exactness is checked, not assumed.** Before producing anything, the mask service generates a throw-away
  segment on its GPU and compares sampled rows of `W8 r` and the full `u W8` of sampled linears with an
  independent CPU reference, bit for bit; any difference stops it. Every production chunk then passes the
  Freivalds self-check, the gateway's import verifies every chunk, and the gateway checks every service result.
* GPU temporaries are zeroed after use, and the GPU memory the allocator caches is scrubbed after each segment.

What the provider can still observe is unchanged from every other topology: timing, sizes and the model in use
(SPEC 3.4).

## Sizing

Per gateway serving one decode stream at about 30 tokens/s (`research/plans/scaling-32b-70b.md`). Mask demand is
one slot per token position (prompt rows included). Prices: AWS on-demand list prices, us-east-1, Linux, as
assumed in Nov 2026; check your region and discounts.

| model | gateway VM | mask supply | GPU mask service VM | slot (K32) | mask demand (decode / with prompts) | est. per hour |
|---|---|---|---|---|---|---|
| qwen2.5-7b | c7i.4xlarge ($0.71) | CPU encrypted factory on the gateway (no GPU) | - | 6.2 MB | 33-40 / ~90 slots/s | ~$0.85 |
| qwen2.5-14b | c7i.8xlarge ($1.43) | CPU encrypted factory on the gateway | (optional g6.4xlarge) | 9.3 MB | 33-40 / ~90 slots/s | ~$1.55 |
| qwen2.5-32b | c7i.8xlarge ($1.43) | **GPU mask service** | g6.4xlarge, 1x L4, 64 GiB ($1.32); or g5.4xlarge, 1x A10G ($1.62) | 19.2 MB | 33-40 / ~90 slots/s | ~$2.95 (L4) |
| qwen2.5-72b | c7i.16xlarge ($2.86) | **GPU mask service** | g6.8xlarge, 1x L4, 128 GiB ($2.01); or g5.8xlarge, 1x A10G ($2.45) | 28.1 MB | 33-40 / ~90 slots/s | ~$5.10 (L4) |
| llama-3.1-70b | c7i.16xlarge ($2.86) | **GPU mask service** | g6.8xlarge ($2.01) / g5.8xlarge ($2.45) | 27.4 MB | 33-40 / ~90 slots/s | ~$5.10 (L4) |

"est. per hour" adds the NAT gateway ($0.045), the provider endpoint ($0.01 per AZ) and the gp3 volumes at full
throughput from 32B on (~$0.15). Not included: PrivateLink data processing ($0.01/GB, on the customer's side of
the endpoint), NAT data processing during setup (the public weights are downloaded once per VM: ~$1.5 at 32B,
~$3.5 at 72B), the provider's compute charges.

**Mask service rate.** Measured on a T4 (Modal, `benchmarks/mask_service_bench.py`: synthetic layers of the
public shapes, K32, extrapolated by layer count, 1024-slot segments into an encrypted store, start-up check
passed first): **32B 55-58 slots/s, 72B 30-40 slots/s** (47-50 slots/s with 4096-slot segments, generation
only); 7B 164-174. A T4's int8 GEMM ceiling at 72B is about 62 slots/s. Estimated for an L4 or A10G (about
1.5-1.9x a T4's int8 throughput): **32B about 85-110 slots/s, 72B about 60-90 slots/s**, i.e. at or above the
decode demand (33-40 slots/s) and around the with-prompts demand (~90 slots/s) at 72B. Above roughly 40 slots/s at
72B the volumes (gp3 tops out at 1 GB/s; a 72B slot is 28 MB) bind before the GPU:
at 72B put the mask store and the outbox on instance-store NVMe or io2, keep the conversation's KV cache across
turns (only new tokens need masks), or use a second GPU VM. Measure on your instance with
`pgw factory mask-service --check-only` (start-up check) and `pgw factory benchmark --pack ... --device cuda`.

**Choice of GPU.** The pack must be readable at GPU speed: weights that do not fit the GPU (24 GB) are streamed
from host RAM, so the VM needs RAM >= 1.5x the pack (64 GiB at 32B, 128 GiB at 72B); disk-streamed weights would
cap the rate at the volume's 1 GB/s. L4 (g6) is the cheapest per slot; A10G (g5) is available in more regions.

## Onboarding in about an hour

Prerequisites (do these first; the quota can take a day):

1. **GPU quota** (32B and larger): Service Quotas -> EC2 -> "Running On-Demand G and VT instances" in the target
   region must allow the GPU VM's vCPUs (16 for g6.4xlarge, 32 for g6.8xlarge). New accounts often have 0.
2. **Region co-location**: deploy into the **same region** as the provider's compute service, and into the zone
   of its zonal endpoint (the onboarding e-mail names both). Every token makes 100-320 sequential round trips;
   another region adds 10-70 ms each and makes the gateway unusably slow.
3. From the provider: the **enrollment code** (`PGW1-...`, single use, expires), the **PrivateLink endpoint service
   name**, the **image registry** (Marketplace ECR) and tag.
4. Tools: Terraform >= 1.5, AWS credentials for the target account (an admin role for the first apply).

Steps:

```bash
cd pgw/deploy/customer-cloud/aws
cp examples/qwen32b-separate-vm.tfvars my.tfvars      # edit: name, region, zone, endpoint, client CIDRs, registry
export TF_VAR_enrollment_code='PGW1-....'             # never put it in a file under version control
terraform init
terraform plan  -var-file=my.tfvars                    # ~35 resources (separate_vm), ~25 (cpu_encrypted)
terraform apply -var-file=my.tfvars                    # ~3 min for AWS; first boot continues on the VMs
```

First boot (both VMs in parallel; follow with `aws ssm start-session --target <instance id>` then
`sudo tail -f /var/log/pgw-bootstrap.log`):

| minute | gateway VM | mask service VM |
|---|---|---|
| 0-5 | Docker, images, enrollment with the code (CA pinned from the code), mask store + KEK | images, waits for the gateway's public values |
| 5-30 | public pack from Hugging Face (32B: ~65 GB download, ~10-25 min) | the same public pack, fingerprint checked against the enrolled one |
| 30-35 | recipient key + store id published; waits for the mask service | signing key + outbox certificate published; **start-up check** (bit-identical to the CPU reference) |
| 35-45 | pulls the first segments, starts the gateway on :8443 | first segments (2048 slots each) |

Then, from a machine in `client_api_cidrs`:

```bash
aws ssm get-parameter --name /pgw/<name>/gateway/client-tls-cert --query Parameter.Value --output text > pgw-ca.pem
aws ssm start-session --target <gateway id>     # then: sudo docker exec pgw-gateway pgw gateway keys create --name app
curl --cacert pgw-ca.pem https://<gateway ip>:8443/v1/chat/completions -H "Authorization: Bearer <key>" \
     -H 'content-type: application/json' -d '{"model":"<name>","messages":[{"role":"user","content":"Hello"}]}'
```

To use your own certificate for the client API, put PEM text into the SSM parameters
`/pgw/<name>/client-tls/cert` (String) and `/pgw/<name>/client-tls/key` (SecureString) before `apply`.

Health: `docker logs pgw-mask-service` shows `mask service start-up check passed` and one line per segment with
its slots/s; `docker logs pgw-mask-pull` lists imported segments; `docker exec pgw-gateway pgw doctor --role
gateway` checks the whole gateway, including mask runway.

## Variables

Required: `region`, `enrollment_code` (sensitive), `client_api_cidrs`, `image_registry`. Important optional ones:
`model` (sizing preset), `mask_supply` (`auto` | `cpu_encrypted` | `separate_vm` | `same_host`),
`gateway_instance_type`, `mask_service_instance_type`, `create_vpc` / `vpc_id` / `subnet_id`,
`provider_endpoint_service_name` (or `provider_cidrs`), `availability_zone`, `check_field` (`ring` | `p31`),
`kms_key_arn`, `expose_client_api_publicly` (off), `allow_setup_egress`. See `aws/variables.tf`.

Network rules created: gateway inbound only TCP 8443 from `client_api_cidrs` (or from the optional NLB);
gateway outbound only to the provider endpoint (API and trip lane ports), the mask service's 8470, DNS, and HTTPS
for setup; mask service inbound only 8470 from the gateway; no SSH ingress (Session Manager instead); IMDSv2
only; every volume encrypted.

`same_host` puts the GPU on the gateway VM (one g6.8xlarge / g6.16xlarge) and writes masks straight into the
gateway's store: one VM fewer and no transfer, but the gateway's CPU and the GPU are billed together around the
clock. `cpu_encrypted` (default up to 14B) needs no GPU at all: the CPU factory gets `W8 r` from the provider on
MLWE ciphertexts (pgw.hemask) and builds a public hint once (about 20 min at 7B).

## Teardown

```bash
terraform destroy -var-file=my.tfvars
```

This deletes both VMs and their volumes (the mask store, the counter, the public packs, any undelivered segments
in the outbox), the SSM parameters, the roles, the endpoint and, with `create_vpc`, the network. Unused masks are
destroyed with the volumes and can never be reused. Also delete the gateway in the provider's console (or ask the
provider) so its client certificate and token are revoked. Terraform state keeps the (consumed) enrollment code
and the courier token: delete the state or keep it encrypted.

## Files

* `aws/`: the Terraform module (`variables.tf`, `main.tf` network and security groups, `instances.tf` VMs, roles,
  SSM, `outputs.tf`, `cloud-init.yaml.tftpl`, `examples/*.tfvars`). Validated with `terraform fmt` and
  `terraform validate` (Terraform 1.9.8, hashicorp/aws 6.73, cloudinit 2.4, random 3.9); `tests/deploy/test_customer_cloud_template.py`
  checks the contract statically. Not applied against a real account in this repository.
* `scripts/pgw-gateway-bootstrap.sh`, `scripts/pgw-mask-service-bootstrap.sh`: first-boot scripts (idempotent),
  usable on other clouds with the SSM calls replaced.
