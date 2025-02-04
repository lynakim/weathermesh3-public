#!/bin/bash

set -e

# Variables
REGION="us-west-2"
FUNCTION_NAME="nlbot-agent"
ECR_URI="205237451631.dkr.ecr.$REGION.amazonaws.com"
IMAGE_NAME="nlbot-agent"
IMAGE_TAG="latest"

# Build the Docker image
echo "Building the Docker image..."
docker build -t $IMAGE_NAME .

# Tag the Docker image with ECR repository
docker tag $IMAGE_NAME:latest $ECR_URI/$IMAGE_NAME:$IMAGE_TAG

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Push the Docker image to ECR
REPO_EXISTS=$(aws ecr describe-repositories --repository-names $IMAGE_NAME --region $REGION 2>&1 || echo "not found")
if [[ $REPO_EXISTS == *"not found"* ]]; then
    echo "ECR repository '$IMAGE_NAME' not found, creating it..."
    aws ecr create-repository --repository-name $IMAGE_NAME --region $REGION
fi

docker push $ECR_URI/$IMAGE_NAME:$IMAGE_TAG

echo "Image URI: $ECR_URI/$IMAGE_NAME:$IMAGE_TAG"

# Update Lambda function to use the new image
aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --region $REGION \
    --image-uri $ECR_URI/$IMAGE_NAME:$IMAGE_TAG
