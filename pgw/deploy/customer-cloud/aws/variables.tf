# Inputs of the PGW customer-cloud deployment (gateway VM + optional GPU mask service VM) in the customer's own
# AWS account, in the provider's region. See ../README.md ("Onboarding in about an hour") and the sizing table.

variable "name" {
  description = "Deployment name: prefix of every resource and of the SSM parameters (/pgw/<name>/...)."
  type        = string
  default     = "pgw"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}$", var.name))
    error_message = "name: 2-31 lower-case letters, digits or '-', starting with a letter."
  }
}

variable "region" {
  description = "AWS region. Must be the region of the provider's compute service (the enrollment e-mail names it): every token makes 100-320 sequential round trips to it."
  type        = string
}

variable "availability_zone" {
  description = "Availability zone for both VMs: the zone of the provider's zonal endpoint (same-AZ RTT). Empty = the first zone of the region."
  type        = string
  default     = ""
}

variable "model" {
  description = "Model the gateway serves (sizing preset). The pack and its fingerprint come from the enrollment; this only picks defaults and the mask supply."
  type        = string
  default     = "qwen2.5-32b"
  validation {
    condition     = contains(["qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b", "llama-3.1-70b"], var.model)
    error_message = "model: one of qwen2.5-7b, qwen2.5-14b, qwen2.5-32b, qwen2.5-72b, llama-3.1-70b."
  }
}

variable "enrollment_code" {
  description = "One-time enrollment code (PGW1-...) from the provider. Stored as an SSM SecureString, read once by the gateway at first boot and then overwritten with 'consumed'. It is single-use and expires; Terraform state still holds it, so keep state encrypted."
  type        = string
  sensitive   = true
  validation {
    condition     = can(regex("^PGW1-", var.enrollment_code))
    error_message = "enrollment_code must start with PGW1-."
  }
}

# ---- network ---------------------------------------------------------------------------------------------------
variable "create_vpc" {
  description = "Create a VPC (private subnet for the VMs, public subnet with a NAT gateway for setup egress). false = use vpc_id and subnet_id."
  type        = bool
  default     = true
}

variable "vpc_cidr" {
  description = "CIDR of the created VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "vpc_id" {
  description = "Existing VPC (create_vpc = false)."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Existing private subnet for both VMs (create_vpc = false), in the provider endpoint's zone, with outbound HTTPS through a NAT or proxy for setup."
  type        = string
  default     = ""
}

variable "provider_endpoint_service_name" {
  description = "PrivateLink endpoint service of the provider's compute service in this region (com.amazonaws.vpce.<region>.vpce-svc-...), from the onboarding e-mail. Empty = reach the provider over provider_cidrs instead."
  type        = string
  default     = ""
}

variable "provider_cidrs" {
  description = "Provider service addresses the gateway may reach when no PrivateLink endpoint is used."
  type        = list(string)
  default     = []
}

variable "provider_ports" {
  description = "Provider ports: the HTTPS API and the native trip lane (TLS)."
  type        = list(number)
  default     = [443, 8443, 9443]
}

variable "client_api_cidrs" {
  description = "Who may call the gateway's client API (TLS on 8443): the premises' thin-client ranges over VPN / Direct Connect, or a VPC CIDR. Never 0.0.0.0/0 unless expose_client_api_publicly is set deliberately."
  type        = list(string)
}

variable "expose_client_api_publicly" {
  description = "Put a network load balancer in the public subnet forwarding TCP 8443 (TLS passthrough to the gateway) for client_api_cidrs. Off by default: the client API stays private."
  type        = bool
  default     = false
}

variable "allow_setup_egress" {
  description = "Allow outbound HTTPS to the internet (container registry, Hugging Face for the public model weights, AWS APIs). Needed at first boot; can be turned off afterwards if VPC endpoints for SSM/ECR/S3 exist."
  type        = bool
  default     = true
}

# ---- gateway VM -----------------------------------------------------------------------------------------------
variable "gateway_instance_type" {
  description = "Gateway VM (CPU only). Empty = the sizing preset for the model."
  type        = string
  default     = ""
}

variable "gateway_data_volume_gb" {
  description = "Encrypted gp3 volume for /var/lib/pgw (public pack, mask store). Empty (0) = the sizing preset."
  type        = number
  default     = 0
}

variable "kms_key_arn" {
  description = "Customer-managed KMS key for the EBS volumes and SSM parameters. Empty = the AWS-managed keys."
  type        = string
  default     = ""
}

# ---- mask supply ----------------------------------------------------------------------------------------------
variable "mask_supply" {
  description = "Where the one-time masks come from: 'auto' (preset: cpu_encrypted up to 14B, separate_vm from 32B), 'cpu_encrypted' (CPU-only encrypted factory on the gateway VM), 'separate_vm' (GPU mask service VM, segments delivered over the signed HPKE export channel), 'same_host' (GPU on the gateway VM itself, shared mask store)."
  type        = string
  default     = "auto"
  validation {
    condition     = contains(["auto", "cpu_encrypted", "separate_vm", "same_host"], var.mask_supply)
    error_message = "mask_supply: auto, cpu_encrypted, separate_vm or same_host."
  }
}

variable "mask_service_instance_type" {
  description = "GPU mask service VM (separate_vm), or the GPU gateway VM (same_host). Empty = the sizing preset (an L4 instance)."
  type        = string
  default     = ""
}

variable "mask_service_data_volume_gb" {
  description = "Encrypted gp3 volume of the mask service (public pack + export outbox). 0 = the sizing preset."
  type        = number
  default     = 0
}

variable "check_field" {
  description = "Segment check format: 'auto' (p31 for the CPU encrypted factory, whose masks come from the provider: the gateway then catches a wrong mask online with the same reaction for every input; ring for masks computed by the customer's own GPU mask service), 'ring' or 'p31' (the gateway accepts p31 by default where its native trip kernels exist)."
  type        = string
  default     = "auto"
  validation {
    condition     = contains(["auto", "ring", "p31"], var.check_field)
    error_message = "check_field: auto, ring or p31."
  }
}

variable "he_scheme" {
  description = "Encrypted mask scheme of the CPU encrypted factory: 'auto' (pks-v3 for cpu_encrypted: compact ~2 GB hint at 7B instead of ~30 GB; PENDING EXTERNAL REVIEW, lattice-estimator run and cryptographer sign-off, research/reviews/pks-v3-review-packet.md; this preset passes the library's acknowledgement), 'mlwe' (wire v2, the reviewed scheme) or 'pks-v3'. pks-v3 needs the provider's PGW_SERVICE_MASKGEN_V3=true."
  type        = string
  default     = "auto"
  validation {
    condition     = contains(["auto", "mlwe", "pks-v3"], var.he_scheme)
    error_message = "he_scheme: auto, mlwe or pks-v3."
  }
}

variable "mask_target_seconds" {
  description = "Masks kept ready at the gateway, in seconds of decoding at 30 tok/s (sets low water / target slots)."
  type        = number
  default     = 120
}

# ---- images ---------------------------------------------------------------------------------------------------
variable "image_registry" {
  description = "Registry path holding pgw-gateway, pgw-factory and pgw-factory-cuda (the Marketplace ECR or the provider's registry), e.g. 123456789012.dkr.ecr.eu-west-1.amazonaws.com/pgw."
  type        = string
}

variable "image_tag" {
  description = "Image tag (pin a released version)."
  type        = string
  default     = "0.1.0"
}

variable "ssh_key_name" {
  description = "Optional EC2 key pair for break-glass SSH (no SSH ingress rule is created; use SSM Session Manager)."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Extra tags for every resource."
  type        = map(string)
  default     = {}
}
