#!/bin/bash

# Simple Lambda Deployment Using AWS Managed Layers
#
# This approach uses AWS's pre-built layers for dependencies
# Your package only contains code + models (~6 MB)

set -e

FUNCTION_NAME="tiktok-virality-predictor"
RUNTIME="python3.11"
HANDLER="lambda_handler.lambda_handler"
MEMORY_SIZE=2048
TIMEOUT=30
REGION="us-east-1"  # Change if needed

# AWS Managed Layer for data science libraries (includes numpy, pandas, scikit-learn)
AWS_DATA_LAYER="arn:aws:lambda:${REGION}:336392948345:layer:AWSSDKPandas-Python311:13"

echo "========================================================================"
echo "Simple Lambda Deployment (Using AWS Layers)"
echo "========================================================================"

# Get your account ID and role ARN
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/FilmTrends-TikTok-Lambda-Role"  

echo ""
echo "Configuration:"
echo "  Account: ${ACCOUNT_ID}"
echo "  Region: ${REGION}"
echo "  Using layer: ${AWS_DATA_LAYER}"
echo ""
echo "⚠️  IMPORTANT: Edit this script and replace YOUR_IAM_ROLE_NAME with your actual IAM role"
echo ""

# Check if lightweight package exists
if [ ! -f "lambda_deployment_light.zip" ]; then
    echo "❌ Package not found!"
    echo "   Run: bash package_lambda_optimized.sh"
    exit 1
fi

PACKAGE_SIZE=$(du -h lambda_deployment_light.zip | cut -f1)
echo "Package: lambda_deployment_light.zip (${PACKAGE_SIZE})"

# Check if function exists
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" &> /dev/null; then
    echo ""
    echo "Updating existing function..."
    
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --zip-file fileb://lambda_deployment_light.zip \
        --region "${REGION}"
    
    echo "✓ Code updated"
    
else
    echo ""
    echo "Creating new function..."
    
    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --runtime "${RUNTIME}" \
        --role "${ROLE_ARN}" \
        --handler "${HANDLER}" \
        --zip-file fileb://lambda_deployment_light.zip \
        --memory-size "${MEMORY_SIZE}" \
        --timeout "${TIMEOUT}" \
        --layers "${AWS_DATA_LAYER}" \
        --region "${REGION}"
    
    echo "✓ Function created with AWS managed layer"
fi

echo ""
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "========================================================================"
echo ""
echo "Test in AWS Console:"
echo "  https://console.aws.amazon.com/lambda/home?region=${REGION}#/functions/${FUNCTION_NAME}"
echo ""