# Qwen2.5-32B: CPU-only gateway + GPU mask service VM (L4) in the provider's region and zone.
name                           = "acme-pgw"
region                         = "eu-west-1"
availability_zone              = "eu-west-1a"
model                          = "qwen2.5-32b"
mask_supply                    = "auto" # -> separate_vm
provider_endpoint_service_name = "com.amazonaws.vpce.eu-west-1.vpce-svc-0123456789abcdef0"
client_api_cidrs               = ["10.200.0.0/16"] # premises thin clients over VPN / Direct Connect
image_registry                 = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/pgw"
image_tag                      = "0.1.0"
# enrollment_code: pass with TF_VAR_enrollment_code (never commit it)
