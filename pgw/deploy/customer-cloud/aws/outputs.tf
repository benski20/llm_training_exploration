output "gateway_private_ip" {
  description = "Gateway VM address (client API: https://<ip>:8443/v1, TLS)."
  value       = aws_instance.gateway.private_ip
}

output "gateway_instance_id" {
  description = "Gateway VM (aws ssm start-session --target <id>)."
  value       = aws_instance.gateway.id
}

output "client_api_url" {
  description = "Client API URL for the premises thin client."
  value       = var.expose_client_api_publicly ? "https://${aws_lb.client[0].dns_name}:8443/v1" : "https://${aws_instance.gateway.private_ip}:8443/v1"
}

output "mask_supply" {
  description = "Mask supply in use: cpu_encrypted, separate_vm or same_host."
  value       = local.supply
}

output "mask_service_instance_id" {
  description = "GPU mask service VM (separate_vm), else null."
  value       = local.separate_vm ? aws_instance.mask_service[0].id : null
}

output "instance_types" {
  description = "Chosen instance types."
  value = {
    gateway      = local.gateway_instance_type
    mask_service = local.separate_vm ? local.mask_service_instance_type : null
  }
}

output "mask_levels" {
  description = "Mask store levels in slots and the store volume size (GB)."
  value = {
    low_water_slots = local.low_water_slots
    target_slots    = local.target_slots
    segment_slots   = local.segment_slots
    gateway_data_gb = local.gateway_data_gb
  }
}

output "client_tls_cert_parameter" {
  description = "SSM parameter holding the gateway's self-signed client API certificate (pin it in the thin client), unless you supplied your own."
  value       = "${local.ssm_prefix}/gateway/client-tls-cert"
}
