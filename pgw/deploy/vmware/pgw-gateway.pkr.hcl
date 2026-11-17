# Packer template: PGW gateway appliance (OVA) for VMware vSphere / ESXi.
#
#   packer init deploy/vmware && packer build -var-file=deploy/vmware/site.pkrvars.hcl deploy/vmware
#
# Builds an Ubuntu 24.04 VM with PGW installed by deploy/systemd/install.sh (gateway + CPU factory),
# then exports an OVA with OVF properties (see ovf-properties.xml) that first boot turns into
# /etc/pgw configuration (pgw-ovf-config.sh). No secrets are baked into the image.

packer {
  required_plugins {
    vsphere = {
      source  = "github.com/hashicorp/vsphere"
      version = ">= 1.4.0"
    }
  }
}

variable "vcenter_server" { type = string }
variable "vcenter_username" { type = string }
variable "vcenter_password" {
  type      = string
  sensitive = true
}
variable "datacenter" { type = string }
variable "cluster" { type = string }
variable "datastore" { type = string }
variable "network" { type = string }
variable "iso_path" {
  type    = string
  default = "[datastore1] iso/ubuntu-24.04-live-server-amd64.iso"
}
variable "pgw_version" {
  type    = string
  default = "0.1.0"
}
variable "pgw_wheel" {
  type        = string
  description = "Path to the pgw wheel built by CI (python -m build)"
}
variable "build_password" {
  type        = string
  sensitive   = true
  description = "Temporary password of the build user; the account is locked at the end of the build"
}

source "vsphere-iso" "pgw_gateway" {
  vcenter_server      = var.vcenter_server
  username            = var.vcenter_username
  password            = var.vcenter_password
  insecure_connection = false
  datacenter          = var.datacenter
  cluster             = var.cluster
  datastore           = var.datastore

  vm_name              = "pgw-gateway-${var.pgw_version}"
  guest_os_type        = "ubuntu64Guest"
  CPUs                 = 8
  RAM                  = 32768
  firmware             = "efi"
  disk_controller_type = ["pvscsi"]
  storage {
    disk_size             = 40960   # OS
    disk_thin_provisioned = true
  }
  storage {
    disk_size             = 512000  # /var/lib/pgw: packs + mask store (encrypted by PGW; use an encrypted datastore too)
    disk_thin_provisioned = true
  }
  network_adapters {
    network      = var.network
    network_card = "vmxnet3"
  }
  iso_paths    = [var.iso_path]
  cd_files     = ["${path.root}/autoinstall/user-data", "${path.root}/autoinstall/meta-data"]
  cd_label     = "cidata"
  boot_command = ["<wait>e<down><down><down><end> autoinstall ds=nocloud;<F10>"]
  boot_wait    = "5s"

  ssh_username = "pgw-build"
  ssh_password = var.build_password
  ssh_timeout  = "45m"

  export {
    force            = true
    output_directory = "output/pgw-gateway-${var.pgw_version}"
    output_format    = "ova"
    options          = ["extraconfig"]
  }
  configuration_parameters = {
    "guestinfo.pgw.managed" = "true"
  }
}

build {
  sources = ["source.vsphere-iso.pgw_gateway"]

  provisioner "file" {
    sources     = [var.pgw_wheel, "${path.root}/../systemd", "${path.root}/pgw-ovf-config.sh"]
    destination = "/tmp/"
  }

  provisioner "shell" {
    execute_command = "echo '${var.build_password}' | sudo -S sh -eu '{{ .Path }}'"
    inline = [
      "apt-get update && apt-get -y upgrade && apt-get -y install python3.12-venv open-vm-tools nftables",
      "mkfs.ext4 -L pgwdata /dev/sdb && echo 'LABEL=pgwdata /var/lib/pgw ext4 defaults,nodev,nosuid,noexec 0 2' >> /etc/fstab",
      "mkdir -p /var/lib/pgw && mount /var/lib/pgw",
      "sh /tmp/systemd/install.sh --role gateway --wheel /tmp/$(basename ${var.pgw_wheel})",
      "sh /tmp/systemd/install.sh --role factory --wheel /tmp/$(basename ${var.pgw_wheel})",
      "install -m 0755 /tmp/pgw-ovf-config.sh /usr/local/sbin/pgw-ovf-config",
      "printf '[Unit]\\nDescription=PGW OVF first-boot configuration\\nBefore=pgw-gateway.service pgw-factory.service\\nConditionPathExists=!/etc/pgw/.ovf-configured\\n[Service]\\nType=oneshot\\nExecStart=/usr/local/sbin/pgw-ovf-config\\n[Install]\\nWantedBy=multi-user.target\\n' > /etc/systemd/system/pgw-ovf-config.service",
      "systemctl enable pgw-ovf-config.service pgw-gateway.service pgw-factory.service",
      "passwd -l pgw-build && rm -f /etc/sudoers.d/pgw-build",
      "cloud-init clean --logs && truncate -s 0 /etc/machine-id && rm -f /etc/ssh/ssh_host_*",
    ]
  }
}
