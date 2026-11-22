terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40, < 7.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5"
    }
    cloudinit = {
      source  = "hashicorp/cloudinit"
      version = ">= 2.3"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = merge({ "pgw:deployment" = var.name, "pgw:managed-by" = "terraform" }, var.tags)
  }
}
