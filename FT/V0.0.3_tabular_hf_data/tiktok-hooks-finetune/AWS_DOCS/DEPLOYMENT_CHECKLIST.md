# AWS Lambda Deployment Checklist
## Deployment Steps I Follow

---

## ✅ **Pre-Deployment Checklist**

### **1. Verify Required Files Exist:**

```bash
# Check files exist
ls -la lambda_handler.py        # ✓ Should exist
ls -la requirements.txt         # ✓ Should exist  
ls -la package_lambda.sh        # ✓ Should exist
ls -la test_local.py            # ✓ Should exist
ls -la deploy.sh                # ✓ Should exist

# Check models exist
ls -la models/regression/xgboost/model.pkl        # ✓ Should exist
ls -la models/classification/xgboost/model.pkl    # ✓ Should exist

# Check artifacts exist
ls -la artifacts/scaler_*.pkl                     # ✓ Should exist
ls -la artifacts/encoders_*.pkl                   # ✓ Should exist
ls -la artifacts/feature_metadata_*.json          # ✓ Should exist
```

### **2. Install AWS CLI (if needed):**

```bash
# macOS
brew install awscli

# Verify
aws --version  # Should show: aws-cli/2.x.x
```

### **3. Configure AWS Credentials:**

```bash
aws configure
```

**Required from AWS Console (IAM → Users → Security Credentials):**
- AWS Access Key ID: `AKIA...`
- AWS Secret Access Key: `wJalrXUtn...`
- Default region: `us-east-1` (or preferred region)
- Output format: `json`

**Test configuration:**
```bash
aws sts get-caller-identity
```

Should show account ID and user ARN.

---

## 🚀 **Deployment Steps**

### **STEP 1: Test Locally First**

**Why:** Catch bugs locally (free, fast) before deploying to AWS (slower, harder to debug)

```bash
# Make test executable
chmod +x test_local.py

# Run tests
python3 test_local.py
```

**Expected output:**
```
✅ ALL TESTS PASSED!
Lambda handler is ready for deployment!
```

**If tests fail:** Fix errors before proceeding.

---

### **STEP 2: Package for Lambda**

**What this does:** 
- Installs Lambda-compatible dependencies
- Creates proper directory structure
- Zips everything correctly

```bash
# Make script executable
chmod +x package_lambda.sh

# Run packaging
bash package_lambda.sh
```

**Expected output:**
```
BUILD COMPLETE!
✓ Package created: lambda_deployment.zip
✓ Size: 45M (or similar)
✓ Ready for AWS Lambda deployment
```

**Package size guidelines:**
- ✅ Under 50 MB → Upload via console (easiest)
- ⚠️ 50-250 MB → Upload via S3 (extra step)
- ❌ Over 250 MB → Need to optimize or use container images

---

### **STEP 3: Deploy to AWS**

**Method A: Automated (Recommended)**

```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
bash deploy.sh
```

**What happens:**
1. Validates AWS credentials
2. Creates IAM role (if needed)
3. Creates or updates Lambda function
4. Configures memory and timeout
5. Tests deployment
6. Shows access URLs

**Method B: Manual (AWS Console)**

1. Go to: https://console.aws.amazon.com/lambda
2. Click "Create function"
3. Choose "Author from scratch"
4. Settings:
   - Function name: `tiktok-virality-predictor`
   - Runtime: Python 3.11
   - Architecture: x86_64
5. Click "Create function"
6. In "Code" tab:
   - Click "Upload from" → ".zip file"
   - Upload `lambda_deployment.zip`
7. In "Configuration" → "General configuration" → "Edit":
   - Memory: 2048 MB
   - Timeout: 30 seconds
8. Click "Save"

---

### **STEP 4: Test Deployment**

**In AWS Console:**

1. Go to Lambda function
2. Click "Test" tab
3. Create new test event:
   - Event name: `test-business-video`
   - Template: `API Gateway AWS Proxy`
   - Event JSON:
```json
{
  "body": "{\"text_hook\": \"Test video hook\", \"caption\": \"Test caption #test\", \"main_category\": \"Business\", \"subcategory\": \"HR & Payroll\", \"length\": 7.0}"
}
```
4. Click "Test"
5. Check response in "Execution results"

**Expected:**
```json
{
  "statusCode": 200,
  "body": "{\"virality_score\": 0.42, \"viral_tier\": \"viral\", ...}"
}
```

**Via AWS CLI:**

```bash
aws lambda invoke \
  --function-name tiktok-virality-predictor \
  --payload '{"body": "{\"text_hook\": \"Test\", \"caption\": \"test\", \"main_category\": \"Music\", \"subcategory\": \"Pop\", \"length\": 10}"}' \
  --region us-east-1 \
  response.json

# View response
cat response.json | python3 -m json.tool
```

---

### **STEP 5: Monitor Performance**

**Check CloudWatch Logs:**

1. AWS Console → CloudWatch → Log groups
2. Find: `/aws/lambda/tiktok-virality-predictor`
3. Click latest log stream
4. See: Initialization logs, requests, responses, errors

**Key metrics to watch:**
- **Duration:** Should be ~50ms (warm) or ~2s (cold start)
- **Memory used:** Should be ~800-1200 MB (well under 2048 MB limit)
- **Errors:** Should be 0%

**AWS CLI:**
```bash
# Tail logs in real-time
aws logs tail /aws/lambda/tiktok-virality-predictor --follow
```

---

## 🐛 **Troubleshooting**

### **Error: "Task timed out after 3.00 seconds"**

**Cause:** Default timeout is 3 seconds (too short for ML models)

**Fix:**
```bash
aws lambda update-function-configuration \
  --function-name tiktok-virality-predictor \
  --timeout 30
```

---

### **Error: "Runtime exited with error: signal: killed"**

**Cause:** Out of memory

**Fix:** Increase memory allocation
```bash
aws lambda update-function-configuration \
  --function-name tiktok-virality-predictor \
  --memory-size 3008  # Try 3 GB
```

---

### **Error: "Unable to import module 'lambda_handler'"**

**Cause:** Package structure is wrong or dependencies missing

**Fix:**
1. Check `lambda_handler.py` is at root of zip (not in subfolder)
2. Check dependencies are in `python/` folder
3. Rebuild: `bash package_lambda.sh`

---

### **Error: "No module named 'numpy'" (or pandas, xgboost)**

**Cause:** Dependencies not packaged correctly

**Fix:**
```bash
# Verify dependencies are in package
unzip -l lambda_deployment.zip | grep numpy

# Should show: python/numpy/...
# If not, rebuild package
```

---

### **Error: Permission denied (IAM)**

**Cause:** Lambda role doesn't have required permissions

**Fix:**
```bash
# Check role has basic execution policy
aws iam list-attached-role-policies --role-name TikTokViralityLambdaRole

# Should include: AWSLambdaBasicExecutionRole
```

---

## 🎯 **Post-Deployment**

### **Test with Different Video Types:**

```bash
# Music video
aws lambda invoke \
  --function-name tiktok-virality-predictor \
  --payload '{"body": "{\"text_hook\": \"Top 10 songs of 2024\", \"caption\": \"#music #top10\", \"main_category\": \"Music\", \"subcategory\": \"Pop\", \"length\": 30}"}' \
  response.json && cat response.json

# Education video
aws lambda invoke \
  --function-name tiktok-virality-predictor \
  --payload '{"body": "{\"text_hook\": \"How to code in Python\", \"caption\": \"#coding #python #tutorial\", \"main_category\": \"Education\", \"subcategory\": \"Programming\", \"length\": 120}"}' \
  response.json && cat response.json
```

---

## 📊 **Monitor Costs**

**Check Lambda metrics:**
1. AWS Console → Lambda → Function
2. Click "Monitor" tab
3. See: Invocations, Duration, Errors, Throttles

**Cost estimate:**
```
Requests: 1,000 predictions/day × 30 days = 30,000/month
Duration: 0.5s average
Memory: 2 GB

Cost = (30,000 × 0.5s × 2GB) × $0.0000166667 = ~$0.50/month
(Well within free tier!)
```

---

## 🔄 **Updating the Model**

**When retraining models:**

1. **Retrain locally:**
```bash
python feature_engineering.py  # New timestamp
python train_models.py
```

2. **Update Lambda handler if timestamp changed:**
   - Handler auto-detects latest timestamp ✓
   - Or set specific timestamp via environment variable

3. **Repackage:**
```bash
bash package_lambda.sh
```

4. **Redeploy:**
```bash
bash deploy.sh
```

**Zero downtime!** Lambda automatically switches to new version.

---

## ✅ **Deployment Complete When I See:**

```
DEPLOYMENT COMPLETE!

✅ Lambda function deployed successfully!

📍 Access Information:
   Function name: tiktok-virality-predictor
   Region: us-east-1
   
✅ Test passed with status code 200
```

**Deployment is live!** 🚀
