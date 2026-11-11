#!/usr/bin/env bash
# Build the Lambda image, push it to ECR and deploy with Terraform.
# Prereqs: AWS CLI configured, Docker, Terraform >= 1.6. Usage: scripts/deploy.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF="$ROOT/infra/terraform"
REGION="${AWS_REGION:-eu-west-1}"
TAG="$(git -C "$ROOT" rev-parse --short HEAD)"

terraform -chdir="$TF" init -input=false
# 1) create the registry first so there is somewhere to push
terraform -chdir="$TF" apply -input=false -target=aws_ecr_repository.app -var "region=$REGION"
REPO="$(terraform -chdir="$TF" output -raw ecr_repository_url)"

# 2) build and push the image
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build --platform linux/amd64 -f "$ROOT/Dockerfile.lambda" -t "$REPO:$TAG" "$ROOT"
docker push "$REPO:$TAG"

# 3) deploy everything else pointing at that image
terraform -chdir="$TF" apply -input=false -var "region=$REGION" -var "image_tag=$TAG"
echo "Function URL: $(terraform -chdir="$TF" output -raw function_url)"
