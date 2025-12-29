#!/bin/bash

# Optimized Lambda Packaging - Uses Layers to Stay Under Limit
#
# Strategy: Split into two packages
# 1. Lambda Layer: Heavy dependencies (numpy, pandas, xgboost) - deploy once, reuse
# 2. Function code: Just your code + models (~6 MB) - fast updates

set -e

echo "========================================================================"
echo "Optimized Lambda Packaging - Layer + Function"
echo "========================================================================"

# ============================================================================
# OPTION 1: Create Lightweight Package (Models + Code Only)
# ============================================================================

echo ""
echo "Creating lightweight package (code + models only)..."

PACKAGE_DIR="lambda_package_light"
rm -rf ${PACKAGE_DIR} lambda_deployment_light.zip

mkdir -p ${PACKAGE_DIR}/models/regression/xgboost
mkdir -p ${PACKAGE_DIR}/models/classification/xgboost
mkdir -p ${PACKAGE_DIR}/artifacts

# Copy handler and models only (NO dependencies)
cp lambda_handler.py ${PACKAGE_DIR}/
cp models/regression/xgboost/model.pkl ${PACKAGE_DIR}/models/regression/xgboost/
cp models/classification/xgboost/model.pkl ${PACKAGE_DIR}/models/classification/xgboost/
cp artifacts/*.pkl ${PACKAGE_DIR}/artifacts/ 2>/dev/null || true
cp artifacts/*.json ${PACKAGE_DIR}/artifacts/ 2>/dev/null || true

# Create zip
cd ${PACKAGE_DIR}
zip -r ../lambda_deployment_light.zip . -q
cd ..

LIGHT_SIZE=$(du -h lambda_deployment_light.zip | cut -f1)
echo "  ✓ Created lambda_deployment_light.zip (${LIGHT_SIZE})"

# ============================================================================
# OPTION 2: Use Pre-built AWS Lambda Layer
# ============================================================================

echo ""
echo "========================================================================"
echo "RECOMMENDED APPROACH: Use AWS Lambda Layer"
echo "========================================================================"
echo ""
echo "AWS provides pre-built layers with ML libraries!"
echo "No need to package dependencies yourself."
echo ""
echo "For Python 3.11 in us-east-1, use this layer ARN:"
echo "  arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python311:13"
echo ""
echo "This includes: numpy, pandas, scikit-learn, and more"
echo ""
echo "For XGBoost, use:"
echo "  arn:aws:lambda:us-east-1:123456789012:layer:xgboost:1"
echo "  (or we can create our own XGBoost layer)"
echo ""
echo "Deployment steps:"
echo "  1. Upload lambda_deployment_light.zip (${LIGHT_SIZE})"
echo "  2. Add layer ARN in Lambda configuration"
echo "  3. Done!"
echo ""

# ============================================================================
# OPTION 3: Create Custom Dependency Layer (If Needed)
# ============================================================================

echo "========================================================================"
echo "Alternative: Create Custom Dependency Layer"
echo "========================================================================"
echo ""
echo "If AWS layers don't have XGBoost, create your own:"
echo ""
echo "  mkdir -p layer/python"
echo "  pip install numpy pandas scikit-learn xgboost -t layer/python/"
echo "  cd layer && zip -r ../dependencies_layer.zip . && cd .."
echo "  "
echo "  aws lambda publish-layer-version \\"
echo "    --layer-name tiktok-virality-dependencies \\"
echo "    --zip-file fileb://dependencies_layer.zip \\"
echo "    --compatible-runtimes python3.11"
echo ""
echo "Then use the layer ARN when creating your function."
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================================================"
echo "PACKAGE BUILT"
echo "========================================================================"
echo ""
echo "✓ Lightweight package: lambda_deployment_light.zip (${LIGHT_SIZE})"
echo ""
echo "Next steps:"
echo "  1. Use AWS-provided layer (easiest)"
echo "  2. Or create custom layer for XGBoost"
echo "  3. Deploy function with layer attached"
echo ""
echo "Benefits of layers:"
echo "  • Stay under 50 MB limit (can upload via console)"
echo "  • Update code without re-uploading dependencies"
echo "  • Share dependencies across multiple functions"
echo "  • Faster deployment iterations"
echo ""