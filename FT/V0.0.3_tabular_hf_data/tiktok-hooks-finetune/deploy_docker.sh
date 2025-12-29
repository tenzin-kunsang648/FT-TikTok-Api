#!/bin/bash

# Deploy Lambda Using Docker Container Image
# Supports packages up to 10 GB (vs 250 MB limit for zip)

set -e

echo "========================================================================"
echo "Docker Container Deployment to AWS Lambda"
echo "========================================================================"

# Configuration
FUNCTION_NAME="tiktok-virality-predictor"
REGION="us-east-1"
IMAGE_NAME="tiktok-virality"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

echo ""
echo "Configuration:"
echo "  AWS Account: ${ACCOUNT_ID}"
echo "  Region: ${REGION}"
echo "  Image: ${IMAGE_NAME}"
echo "  Repository: ${REPOSITORY_URI}"

# ============================================================================
# STEP 1: Create ECR Repository (if doesn't exist)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: Creating ECR repository"
echo "========================================================================"

if aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${REGION} &> /dev/null; then
    echo "  ✓ Repository already exists"
else
    echo "  Creating repository..."
    aws ecr create-repository \
        --repository-name ${IMAGE_NAME} \
        --region ${REGION} \
        > /dev/null
    echo "  ✓ Repository created"
fi

# ============================================================================
# STEP 2: Build Docker Image
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Building Docker image"
echo "========================================================================"

echo "  Building image (this takes 5-10 minutes)..."
DOCKER_BUILDKIT=0 docker build --platform linux/amd64 -t ${IMAGE_NAME}:latest .

echo "  ✓ Image built"

# ============================================================================
# STEP 3: Login to ECR
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 3: Authenticating with ECR"
echo "========================================================================"

aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "  ✓ Authenticated"

# ============================================================================
# STEP 4: Tag and Push Image
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 4: Pushing image to ECR"
echo "========================================================================"

docker tag ${IMAGE_NAME}:latest ${REPOSITORY_URI}:latest

echo "  Pushing image (this takes 3-5 minutes)..."
docker push ${REPOSITORY_URI}:latest

echo "  ✓ Image pushed to ECR"

# ============================================================================
# STEP 5: Create or Update Lambda Function
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 5: Deploying Lambda function"
echo "========================================================================"

# Check if function exists
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} &> /dev/null; then
    echo "  Updating existing function..."
    
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${REPOSITORY_URI}:latest \
        --region ${REGION} \
        > /dev/null
    
    echo "  ✓ Function updated"
else
    echo "  Creating new function..."
    
    # You need to set your IAM role ARN here
    ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/TikTokViralityLambdaRole"
    
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${REPOSITORY_URI}:latest \
        --role ${ROLE_ARN} \
        --timeout 30 \
        --memory-size 2048 \
        --region ${REGION} \
        > /dev/null
    
    echo "  ✓ Function created"
fi

# ============================================================================
# STEP 6: Test
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 6: Testing deployment"
echo "========================================================================"

TEST_PAYLOAD='{"body": "{\"text_hook\": \"Test\", \"caption\": \"test\", \"main_category\": \"Music\", \"subcategory\": \"Pop\", \"length\": 10}"}'

aws lambda invoke \
    --function-name ${FUNCTION_NAME} \
    --payload "${TEST_PAYLOAD}" \
    --region ${REGION} \
    response.json \
    > /dev/null

echo "  ✓ Function invoked"
echo ""
echo "  Response:"
cat response.json | python3 -m json.tool

rm -f response.json

echo ""
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Your Lambda function is live!"
echo "  Function: ${FUNCTION_NAME}"
echo "  Region: ${REGION}"
echo "  Image: ${REPOSITORY_URI}:latest"
echo ""