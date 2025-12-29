# AWS Lambda Quick Reference Card
## Common Commands I Use

---

## 🚀 **Initial Deployment**

```bash
# 1. Test locally
python3 test_local.py

# 2. Package
bash package_lambda.sh

# 3. Deploy
bash deploy.sh
```

---

## 🔄 **Update Deployed Function**

```bash
# After retraining models or changing code:

# Repackage
bash package_lambda.sh

# Update code only
aws lambda update-function-code \
  --function-name tiktok-virality-predictor \
  --zip-file fileb://lambda_deployment.zip
```

---

## 🧪 **Testing**

```bash
# Invoke function
aws lambda invoke \
  --function-name tiktok-virality-predictor \
  --payload '{"body": "{\"text_hook\":\"test\",\"caption\":\"test\",\"main_category\":\"Music\",\"subcategory\":\"Pop\",\"length\":10}"}' \
  response.json

# View response
cat response.json | python3 -m json.tool
```

---

## 📊 **Monitoring**

```bash
# View logs (real-time)
aws logs tail /aws/lambda/tiktok-virality-predictor --follow

# Get function info
aws lambda get-function --function-name tiktok-virality-predictor

# List recent invocations
aws lambda list-functions | grep tiktok-virality-predictor
```

---

## ⚙️ **Configuration Changes**

```bash
# Increase memory
aws lambda update-function-configuration \
  --function-name tiktok-virality-predictor \
  --memory-size 3008

# Increase timeout
aws lambda update-function-configuration \
  --function-name tiktok-virality-predictor \
  --timeout 60

# Add environment variable
aws lambda update-function-configuration \
  --function-name tiktok-virality-predictor \
  --environment "Variables={MODEL_VERSION=v2,LOG_LEVEL=DEBUG}"
```

---

## 🗑️ **Cleanup**

```bash
# Delete function
aws lambda delete-function \
  --function-name tiktok-virality-predictor

# Delete IAM role
aws iam detach-role-policy \
  --role-name TikTokViralityLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam delete-role --role-name TikTokViralityLambdaRole
```

---

## 🎯 **My Deployment Configuration**

- **Function:** `tiktok-virality-predictor`
- **Runtime:** Python 3.11
- **Memory:** 2048 MB
- **Timeout:** 30 seconds
- **Package:** ~170 MB
- **Models:** XGBoost (regression + classification)
- **Cold start:** ~2 seconds
- **Warm latency:** ~50ms
- **Cost:** ~$0.035 per 1000 predictions
