# Qwen2.5-7B: one CPU-only VM (gateway + CPU encrypted factory); no GPU anywhere at the customer.
name                           = "acme-pgw-7b"
region                         = "eu-west-1"
model                          = "qwen2.5-7b"
mask_supply                    = "auto" # -> cpu_encrypted
provider_endpoint_service_name = "com.amazonaws.vpce.eu-west-1.vpce-svc-0123456789abcdef0"
client_api_cidrs               = ["10.200.0.0/16"]
image_registry                 = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/pgw"
