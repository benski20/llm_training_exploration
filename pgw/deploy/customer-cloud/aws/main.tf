# PGW customer-cloud deployment on AWS: a CPU-only gateway VM and, for 32B+, a small GPU mask service VM, both in
# the customer's own account, in the provider's region and zone. The provider never gets r, the segment seeds or
# the check vectors; the mask service computes W8 r from the public weights, which the provider has already.

locals {
  # Sizing presets (README "Sizing"). slot_mb: W r bytes per token position at K32; pack_gb: public int8 pack;
  # hint_gb: one-time public hint of the CPU encrypted factory (cpu_encrypted only).
  presets = {
    "qwen2.5-7b"    = { gateway = "c7i.4xlarge", supply = "cpu_encrypted", gpu = "g6.4xlarge", gpu_same_host = "g6.8xlarge", slot_mb = 6.2, pack_gb = 10, hint_gb = 30 }
    "qwen2.5-14b"   = { gateway = "c7i.8xlarge", supply = "cpu_encrypted", gpu = "g6.4xlarge", gpu_same_host = "g6.8xlarge", slot_mb = 9.3, pack_gb = 17, hint_gb = 55 }
    "qwen2.5-32b"   = { gateway = "c7i.8xlarge", supply = "separate_vm", gpu = "g6.4xlarge", gpu_same_host = "g6.8xlarge", slot_mb = 19.2, pack_gb = 36, hint_gb = 0 }
    "qwen2.5-72b"   = { gateway = "c7i.16xlarge", supply = "separate_vm", gpu = "g6.8xlarge", gpu_same_host = "g6.16xlarge", slot_mb = 28.1, pack_gb = 78, hint_gb = 0 }
    "llama-3.1-70b" = { gateway = "c7i.16xlarge", supply = "separate_vm", gpu = "g6.8xlarge", gpu_same_host = "g6.16xlarge", slot_mb = 27.4, pack_gb = 76, hint_gb = 0 }
  }
  p      = local.presets[var.model]
  supply = var.mask_supply == "auto" ? local.p.supply : var.mask_supply

  separate_vm = local.supply == "separate_vm"
  same_host   = local.supply == "same_host"

  gateway_instance_type = var.gateway_instance_type != "" ? var.gateway_instance_type : (
  local.same_host ? (var.mask_service_instance_type != "" ? var.mask_service_instance_type : local.p.gpu_same_host) : local.p.gateway)
  mask_service_instance_type = var.mask_service_instance_type != "" ? var.mask_service_instance_type : local.p.gpu

  # Masks kept ready at the gateway: decoding at 30 tok/s plus prompt rows (factor 2.5, scaling plan section 6).
  target_slots    = ceil(var.mask_target_seconds * 30 * 2.5)
  low_water_slots = ceil(local.target_slots / 3)
  segment_slots   = local.p.slot_mb > 15 ? 512 : 2048
  store_gb        = ceil(local.target_slots * local.p.slot_mb / 1000 * 1.3)
  gateway_data_gb = var.gateway_data_volume_gb > 0 ? var.gateway_data_volume_gb : (
  local.p.pack_gb * (local.same_host ? 2 : 1) + (local.supply == "cpu_encrypted" ? 2 * local.p.hint_gb : 0) + local.store_gb + 40)
  outbox_slots         = max(local.segment_slots * 2, ceil(local.target_slots / 4))
  mask_service_data_gb = var.mask_service_data_volume_gb > 0 ? var.mask_service_data_volume_gb : ceil(2 * local.p.pack_gb + 2 * local.outbox_slots * local.p.slot_mb / 1000 + 40)
  # gp3 at full throughput from 32B on: the mask store ingests and serves 0.5-2.6 GB/s at 32B-72B.
  big_io = local.p.slot_mb > 15

  ssm_prefix = "/pgw/${var.name}"
  az         = var.availability_zone != "" ? var.availability_zone : data.aws_availability_zones.this.names[0]
  vpc_id     = var.create_vpc ? aws_vpc.this[0].id : var.vpc_id
  subnet_id  = var.create_vpc ? aws_subnet.private[0].id : var.subnet_id
}

data "aws_availability_zones" "this" {
  state = "available"
}

data "aws_vpc" "selected" {
  id = local.vpc_id
}

# ---- network (optional) ---------------------------------------------------------------------------------------
resource "aws_vpc" "this" {
  count                = var.create_vpc ? 1 : 0
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${var.name}-vpc" }
}

resource "aws_subnet" "private" {
  count             = var.create_vpc ? 1 : 0
  vpc_id            = aws_vpc.this[0].id
  availability_zone = local.az
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, 1)
  tags              = { Name = "${var.name}-private" }
}

resource "aws_subnet" "public" {
  count             = var.create_vpc ? 1 : 0
  vpc_id            = aws_vpc.this[0].id
  availability_zone = local.az
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, 0)
  tags              = { Name = "${var.name}-public" }
}

resource "aws_internet_gateway" "this" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.this[0].id
}

resource "aws_eip" "nat" {
  count  = var.create_vpc ? 1 : 0
  domain = "vpc"
}

resource "aws_nat_gateway" "this" {
  count         = var.create_vpc ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.this]
}

resource "aws_route_table" "public" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.this[0].id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this[0].id
  }
}

resource "aws_route_table" "private" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.this[0].id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this[0].id
  }
}

resource "aws_route_table_association" "public" {
  count          = var.create_vpc ? 1 : 0
  subnet_id      = aws_subnet.public[0].id
  route_table_id = aws_route_table.public[0].id
}

resource "aws_route_table_association" "private" {
  count          = var.create_vpc ? 1 : 0
  subnet_id      = aws_subnet.private[0].id
  route_table_id = aws_route_table.private[0].id
}

# ---- security groups: only gateway <-> provider, gateway <-> mask service, clients -> gateway TLS ------------------
resource "aws_security_group" "gateway" {
  name        = "${var.name}-gateway"
  description = "PGW gateway: client API (TLS 8443) from client_api_cidrs only; egress to the provider and the mask service"
  vpc_id      = local.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "gateway_clients" {
  for_each          = toset(var.client_api_cidrs)
  security_group_id = aws_security_group.gateway.id
  description       = "client API (TLS)"
  ip_protocol       = "tcp"
  from_port         = 8443
  to_port           = 8443
  cidr_ipv4         = each.value
}

resource "aws_vpc_security_group_ingress_rule" "gateway_from_nlb" {
  count                        = var.expose_client_api_publicly ? 1 : 0
  security_group_id            = aws_security_group.gateway.id
  description                  = "client API via the NLB (TLS passthrough)"
  ip_protocol                  = "tcp"
  from_port                    = 8443
  to_port                      = 8443
  referenced_security_group_id = aws_security_group.nlb[0].id
}

resource "aws_vpc_security_group_egress_rule" "gateway_to_provider_endpoint" {
  for_each                     = var.provider_endpoint_service_name != "" ? toset([for p in var.provider_ports : tostring(p)]) : toset([])
  security_group_id            = aws_security_group.gateway.id
  description                  = "provider compute service (PrivateLink): API and trip lane"
  ip_protocol                  = "tcp"
  from_port                    = tonumber(each.value)
  to_port                      = tonumber(each.value)
  referenced_security_group_id = aws_security_group.endpoint[0].id
}

resource "aws_vpc_security_group_egress_rule" "gateway_to_provider_cidrs" {
  for_each          = { for pair in setproduct(var.provider_cidrs, var.provider_ports) : "${pair[0]}:${pair[1]}" => pair }
  security_group_id = aws_security_group.gateway.id
  description       = "provider compute service: API and trip lane"
  ip_protocol       = "tcp"
  from_port         = each.value[1]
  to_port           = each.value[1]
  cidr_ipv4         = each.value[0]
}

resource "aws_vpc_security_group_egress_rule" "gateway_to_mask_service" {
  count                        = local.separate_vm ? 1 : 0
  security_group_id            = aws_security_group.gateway.id
  description                  = "pull signed, HPKE-sealed mask segments from the mask service outbox"
  ip_protocol                  = "tcp"
  from_port                    = 8470
  to_port                      = 8470
  referenced_security_group_id = aws_security_group.mask_service[0].id
}

resource "aws_vpc_security_group_egress_rule" "gateway_setup_https" {
  count             = var.allow_setup_egress ? 1 : 0
  security_group_id = aws_security_group.gateway.id
  description       = "setup: registry, Hugging Face (public weights), AWS APIs"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_egress_rule" "gateway_dns" {
  security_group_id = aws_security_group.gateway.id
  description       = "VPC resolver"
  ip_protocol       = "udp"
  from_port         = 53
  to_port           = 53
  cidr_ipv4         = "${cidrhost(data.aws_vpc.selected.cidr_block, 2)}/32"
}

resource "aws_security_group" "mask_service" {
  count       = local.separate_vm ? 1 : 0
  name        = "${var.name}-mask-service"
  description = "PGW GPU mask service: outbox (8470) from the gateway only; no other inbound"
  vpc_id      = local.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "mask_service_from_gateway" {
  count                        = local.separate_vm ? 1 : 0
  security_group_id            = aws_security_group.mask_service[0].id
  description                  = "gateway pulls segments (TLS, bearer token)"
  ip_protocol                  = "tcp"
  from_port                    = 8470
  to_port                      = 8470
  referenced_security_group_id = aws_security_group.gateway.id
}

resource "aws_vpc_security_group_egress_rule" "mask_service_setup_https" {
  count             = local.separate_vm && var.allow_setup_egress ? 1 : 0
  security_group_id = aws_security_group.mask_service[0].id
  description       = "setup: registry, Hugging Face (public weights), AWS APIs"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_egress_rule" "mask_service_dns" {
  count             = local.separate_vm ? 1 : 0
  security_group_id = aws_security_group.mask_service[0].id
  description       = "VPC resolver"
  ip_protocol       = "udp"
  from_port         = 53
  to_port           = 53
  cidr_ipv4         = "${cidrhost(data.aws_vpc.selected.cidr_block, 2)}/32"
}

resource "aws_security_group" "endpoint" {
  count       = var.provider_endpoint_service_name != "" ? 1 : 0
  name        = "${var.name}-provider-endpoint"
  description = "Interface endpoint to the provider's compute service: reachable from the gateway only"
  vpc_id      = local.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "endpoint_from_gateway" {
  for_each                     = var.provider_endpoint_service_name != "" ? toset([for p in var.provider_ports : tostring(p)]) : toset([])
  security_group_id            = aws_security_group.endpoint[0].id
  ip_protocol                  = "tcp"
  from_port                    = tonumber(each.value)
  to_port                      = tonumber(each.value)
  referenced_security_group_id = aws_security_group.gateway.id
}

resource "aws_vpc_endpoint" "provider" {
  count               = var.provider_endpoint_service_name != "" ? 1 : 0
  vpc_id              = local.vpc_id
  service_name        = var.provider_endpoint_service_name
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [local.subnet_id]
  security_group_ids  = [aws_security_group.endpoint[0].id]
  private_dns_enabled = true
  tags                = { Name = "${var.name}-provider" }
}

# ---- optional public entry for the client API (TLS passthrough; the gateway terminates TLS) ---------------------
resource "aws_security_group" "nlb" {
  count       = var.expose_client_api_publicly ? 1 : 0
  name        = "${var.name}-client-nlb"
  description = "Client API entry: TCP 8443 from client_api_cidrs to the gateway"
  vpc_id      = local.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "nlb_clients" {
  for_each          = var.expose_client_api_publicly ? toset(var.client_api_cidrs) : toset([])
  security_group_id = aws_security_group.nlb[0].id
  ip_protocol       = "tcp"
  from_port         = 8443
  to_port           = 8443
  cidr_ipv4         = each.value
}

resource "aws_vpc_security_group_egress_rule" "nlb_to_gateway" {
  count                        = var.expose_client_api_publicly ? 1 : 0
  security_group_id            = aws_security_group.nlb[0].id
  ip_protocol                  = "tcp"
  from_port                    = 8443
  to_port                      = 8443
  referenced_security_group_id = aws_security_group.gateway.id
}

resource "aws_lb" "client" {
  count              = var.expose_client_api_publicly ? 1 : 0
  name               = "${var.name}-client"
  load_balancer_type = "network"
  internal           = false
  subnets            = var.create_vpc ? [aws_subnet.public[0].id] : [var.subnet_id]
  security_groups    = [aws_security_group.nlb[0].id]
}

resource "aws_lb_target_group" "client" {
  count       = var.expose_client_api_publicly ? 1 : 0
  name        = "${var.name}-client"
  port        = 8443
  protocol    = "TCP"
  target_type = "instance"
  vpc_id      = local.vpc_id
}

resource "aws_lb_target_group_attachment" "client" {
  count            = var.expose_client_api_publicly ? 1 : 0
  target_group_arn = aws_lb_target_group.client[0].arn
  target_id        = aws_instance.gateway.id
  port             = 8443
}

resource "aws_lb_listener" "client" {
  count             = var.expose_client_api_publicly ? 1 : 0
  load_balancer_arn = aws_lb.client[0].arn
  port              = 8443
  protocol          = "TCP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.client[0].arn
  }
}
