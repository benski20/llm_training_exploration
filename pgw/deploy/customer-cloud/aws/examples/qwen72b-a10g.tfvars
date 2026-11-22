# Qwen2.5-72B with an A10G mask service instead of the L4 default, in an existing VPC, p31 segments.
name                           = "acme-pgw-72b"
region                         = "us-east-1"
availability_zone              = "us-east-1b"
model                          = "qwen2.5-72b"
mask_service_instance_type     = "g5.8xlarge"
create_vpc                     = false
vpc_id                         = "vpc-0123456789abcdef0"
subnet_id                      = "subnet-0123456789abcdef0"
provider_endpoint_service_name = "com.amazonaws.vpce.us-east-1.vpce-svc-0123456789abcdef0"
client_api_cidrs               = ["10.0.0.0/8"]
image_registry                 = "123456789012.dkr.ecr.us-east-1.amazonaws.com/pgw"
check_field                    = "p31"
