terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "validator" {
  ami           = "ami-0c55b159cbfafe1f0" # Ubuntu 20.04 LTS
  instance_type = "c5.4xlarge"
  key_name      = "solana-bench-key"

  tags = {
    Name = "solana-validator"
  }
}

resource "aws_instance" "execution_node" {
  ami           = "ami-0c55b159cbfafe1f0" # Ubuntu 20.04 LTS
  instance_type = "c5.4xlarge"
  key_name      = "solana-bench-key"

  tags = {
    Name = "execution-node"
  }
}

resource "aws_instance" "load_generator" {
  ami           = "ami-0c55b159cbfafe1f0" # Ubuntu 20.04 LTS
  instance_type = "c5.4xlarge"
  key_name      = "solana-bench-key"

  tags = {
    Name = "load-generator"
  }
}

resource "aws_security_group" "bench_sg" {
  name        = "solana-bench-sg"
  description = "Allow all internal traffic and SSH"

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
