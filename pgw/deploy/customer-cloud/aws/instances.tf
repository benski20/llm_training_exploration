# VMs, their roles and the SSM parameters through which the two VMs exchange *public* keys at first boot.

data "aws_caller_identity" "this" {}
data "aws_partition" "this" {}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}

data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }
}

locals {
  param_arn = "arn:${data.aws_partition.this.partition}:ssm:${var.region}:${data.aws_caller_identity.this.account_id}:parameter${local.ssm_prefix}"
  # Written by the VMs at first boot (public values only); Terraform creates the placeholders so that destroy
  # removes them. "pending" means "not written yet".
  exchange_params = concat(
    ["gateway/client-tls-cert"],
    local.separate_vm ? ["gateway/recipient-public-key-hex", "gateway/store-id", "gateway/model",
    "mask-service/signing-public-key-hex", "mask-service/outbox-cert"] : [],
  )
}

# ---- SSM parameters -------------------------------------------------------------------------------------------
resource "aws_ssm_parameter" "enrollment_code" {
  name        = "${local.ssm_prefix}/enrollment-code"
  description = "One-time PGW enrollment code; the gateway overwrites it with 'consumed' after enrolling"
  type        = "SecureString"
  key_id      = var.kms_key_arn != "" ? var.kms_key_arn : null
  value       = var.enrollment_code
  lifecycle {
    ignore_changes = [value]
  }
}

resource "random_password" "courier" {
  count   = local.separate_vm ? 1 : 0
  length  = 48
  special = false
}

resource "aws_ssm_parameter" "courier_token" {
  count       = local.separate_vm ? 1 : 0
  name        = "${local.ssm_prefix}/courier-token"
  description = "Bearer token between the gateway and the mask service outbox (VPC-internal)"
  type        = "SecureString"
  key_id      = var.kms_key_arn != "" ? var.kms_key_arn : null
  value       = random_password.courier[0].result
}

resource "aws_ssm_parameter" "exchange" {
  for_each    = toset(local.exchange_params)
  name        = "${local.ssm_prefix}/${each.value}"
  description = "PGW first-boot exchange of a public value (written by the VM)"
  type        = "String"
  tier        = "Advanced" # certificates can exceed 4 KB
  value       = "pending"
  lifecycle {
    ignore_changes = [value]
  }
}

# ---- IAM ------------------------------------------------------------------------------------------------------
data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "gateway" {
  statement {
    sid       = "ReadOwnParameters"
    actions   = ["ssm:GetParameter"]
    resources = ["${local.param_arn}/*"]
  }
  statement {
    sid       = "WriteGatewayParameters"
    actions   = ["ssm:PutParameter"]
    resources = ["${local.param_arn}/enrollment-code", "${local.param_arn}/gateway/*"]
  }
  statement {
    sid       = "PullImages"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["*"]
  }
  dynamic "statement" {
    for_each = var.kms_key_arn != "" ? [1] : []
    content {
      actions   = ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"]
      resources = [var.kms_key_arn]
    }
  }
}

data "aws_iam_policy_document" "mask_service" {
  statement {
    sid     = "ReadGatewayPublicValuesAndToken"
    actions = ["ssm:GetParameter"]
    # never the enrollment code
    resources = ["${local.param_arn}/gateway/*", "${local.param_arn}/courier-token", "${local.param_arn}/mask-service/*"]
  }
  statement {
    sid       = "WriteMaskServiceParameters"
    actions   = ["ssm:PutParameter"]
    resources = ["${local.param_arn}/mask-service/*"]
  }
  statement {
    sid       = "PullImages"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["*"]
  }
  dynamic "statement" {
    for_each = var.kms_key_arn != "" ? [1] : []
    content {
      actions   = ["kms:Decrypt"]
      resources = [var.kms_key_arn]
    }
  }
}

resource "aws_iam_role" "gateway" {
  name               = "${var.name}-gateway"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy" "gateway" {
  role   = aws_iam_role.gateway.id
  policy = data.aws_iam_policy_document.gateway.json
}

resource "aws_iam_role_policy_attachment" "gateway_ssm" {
  role       = aws_iam_role.gateway.name
  policy_arn = "arn:${data.aws_partition.this.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gateway" {
  name = "${var.name}-gateway"
  role = aws_iam_role.gateway.name
}

resource "aws_iam_role" "mask_service" {
  count              = local.separate_vm ? 1 : 0
  name               = "${var.name}-mask-service"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy" "mask_service" {
  count  = local.separate_vm ? 1 : 0
  role   = aws_iam_role.mask_service[0].id
  policy = data.aws_iam_policy_document.mask_service.json
}

resource "aws_iam_role_policy_attachment" "mask_service_ssm" {
  count      = local.separate_vm ? 1 : 0
  role       = aws_iam_role.mask_service[0].name
  policy_arn = "arn:${data.aws_partition.this.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "mask_service" {
  count = local.separate_vm ? 1 : 0
  name  = "${var.name}-mask-service"
  role  = aws_iam_role.mask_service[0].name
}

# ---- cloud-init ------------------------------------------------------------------------------------------------
data "cloudinit_config" "gateway" {
  gzip          = true
  base64_encode = true
  part {
    content_type = "text/cloud-config"
    content = templatefile("${path.module}/cloud-init.yaml.tftpl", {
      script = file("${path.module}/../scripts/pgw-gateway-bootstrap.sh")
      env = {
        AWS_REGION          = var.region
        PGW_SSM_PREFIX      = local.ssm_prefix
        PGW_GATEWAY_NAME    = "${var.name}-gateway"
        PGW_MODEL_HINT      = var.model
        PGW_MASK_SUPPLY     = local.supply
        PGW_IMAGE_REGISTRY  = var.image_registry
        PGW_IMAGE_TAG       = var.image_tag
        PGW_CHECK_FIELD     = var.check_field
        PGW_HE_SCHEME       = local.he_scheme
        PGW_SEGMENT_SLOTS   = tostring(local.segment_slots)
        PGW_LOW_WATER_SLOTS = tostring(local.low_water_slots)
        PGW_TARGET_SLOTS    = tostring(local.target_slots)
        PGW_FACTORY_CPUS    = "6"
        PGW_MASK_SERVICE_IP = local.separate_vm ? aws_instance.mask_service[0].private_ip : ""
      }
    })
  }
}

data "cloudinit_config" "mask_service" {
  count         = local.separate_vm ? 1 : 0
  gzip          = true
  base64_encode = true
  part {
    content_type = "text/cloud-config"
    content = templatefile("${path.module}/cloud-init.yaml.tftpl", {
      script = file("${path.module}/../scripts/pgw-mask-service-bootstrap.sh")
      env = {
        AWS_REGION              = var.region
        PGW_SSM_PREFIX          = local.ssm_prefix
        PGW_IMAGE_REGISTRY      = var.image_registry
        PGW_IMAGE_TAG           = var.image_tag
        PGW_CHECK_FIELD         = var.check_field
        PGW_SEGMENT_SLOTS       = tostring(local.segment_slots)
        PGW_LOW_WATER_SLOTS     = tostring(local.segment_slots)
        PGW_OUTBOX_TARGET_SLOTS = tostring(local.outbox_slots)
        PGW_OUTBOX_MAX_BYTES    = tostring(ceil(3 * local.outbox_slots * local.p.slot_mb * 1000000))
      }
    })
  }
}

# ---- VMs ------------------------------------------------------------------------------------------------------
resource "aws_instance" "mask_service" {
  count                  = local.separate_vm ? 1 : 0
  ami                    = data.aws_ami.gpu.id
  instance_type          = local.mask_service_instance_type
  subnet_id              = local.subnet_id
  vpc_security_group_ids = [aws_security_group.mask_service[0].id]
  iam_instance_profile   = aws_iam_instance_profile.mask_service[0].name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null
  user_data_base64       = data.cloudinit_config.mask_service[0].rendered
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    encrypted   = true
    kms_key_id  = var.kms_key_arn != "" ? var.kms_key_arn : null
  }
  ebs_block_device {
    device_name           = "/dev/sdf"
    volume_size           = local.mask_service_data_gb
    volume_type           = "gp3"
    throughput            = local.big_io ? 1000 : 250
    iops                  = local.big_io ? 16000 : 4000
    encrypted             = true
    kms_key_id            = var.kms_key_arn != "" ? var.kms_key_arn : null
    delete_on_termination = true
  }
  tags       = { Name = "${var.name}-mask-service", "pgw:role" = "mask-service" }
  depends_on = [aws_ssm_parameter.exchange, aws_ssm_parameter.courier_token]
}

resource "aws_instance" "gateway" {
  ami                    = local.same_host ? data.aws_ami.gpu.id : data.aws_ami.ubuntu.id
  instance_type          = local.gateway_instance_type
  subnet_id              = local.subnet_id
  vpc_security_group_ids = [aws_security_group.gateway.id]
  iam_instance_profile   = aws_iam_instance_profile.gateway.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null
  user_data_base64       = data.cloudinit_config.gateway.rendered
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  root_block_device {
    volume_size = 60
    volume_type = "gp3"
    encrypted   = true
    kms_key_id  = var.kms_key_arn != "" ? var.kms_key_arn : null
  }
  # /var/lib/pgw: public pack(s), encrypted mask store (single-use; never restore it from a snapshot)
  ebs_block_device {
    device_name           = "/dev/sdf"
    volume_size           = local.gateway_data_gb
    volume_type           = "gp3"
    throughput            = local.big_io ? 1000 : 250
    iops                  = local.big_io ? 16000 : 4000
    encrypted             = true
    kms_key_id            = var.kms_key_arn != "" ? var.kms_key_arn : null
    delete_on_termination = true
  }
  # /var/lib/pgw-counter: the mask store's monotonic generation counter, on its own volume (exclude from backups)
  ebs_block_device {
    device_name           = "/dev/sdg"
    volume_size           = 1
    volume_type           = "gp3"
    encrypted             = true
    kms_key_id            = var.kms_key_arn != "" ? var.kms_key_arn : null
    delete_on_termination = true
    tags                  = { "pgw:backup" = "never" }
  }
  tags       = { Name = "${var.name}-gateway", "pgw:role" = "gateway" }
  depends_on = [aws_ssm_parameter.enrollment_code, aws_ssm_parameter.exchange, aws_vpc_endpoint.provider, aws_nat_gateway.this]
}
