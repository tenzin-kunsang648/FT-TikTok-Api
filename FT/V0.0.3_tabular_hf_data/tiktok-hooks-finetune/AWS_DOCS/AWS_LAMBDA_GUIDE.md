# AWS Lambda Deployment Guide - TikTok Virality Prediction
## Notes on Deploying ML Models to AWS Lambda

---

## 📚 **PART 1: Understanding AWS Lambda**

### **What is AWS Lambda?**

AWS Lambda is a serverless compute service that runs code without managing servers. It's a Function-as-a-Service (FaaS) platform that executes code in response to events (HTTP requests, file uploads, scheduled triggers) with automatic scaling and pay-per-use pricing. I upload code, and AWS handles infrastructure, scaling, and availability.

### **Key Concepts:**

#### **1. Serverless Computing**
- **I write:** Functions (code)
- **AWS manages:** Servers, scaling, patching, availability
- **I pay for:** Execution time (per millisecond)

#### **2. Lambda Function Anatomy**
```python
def lambda_handler(event, context):
    """
    event: Input data (JSON from API Gateway, S3, etc.)
    context: Runtime info (request ID, memory, timeout)
    
    return: Response (usually JSON)
    """
    # My code here
    return {"statusCode": 200, "body": "Hello World"}
```

#### **3. Key Limitations**
- **Memory:** 128 MB to 10 GB
- **Timeout:** Max 15 minutes
- **Package size:** 50 MB (zipped), 250 MB (unzipped)
- **Ephemeral storage:** 512 MB in `/tmp`
- **Cold start:** First invocation is slower (~1-3 seconds)

#### **4. Pricing Model**
- **Free tier:** 1M requests/month + 400,000 GB-seconds compute
- **After:** $0.20 per 1M requests + $0.0000166667 per GB-second
- **Example:** 1M predictions at 2GB/1sec = ~$35/month

#### **5. Common Use Cases**
- ✅ **APIs/Microservices** (like my ML model)
- ✅ **Data processing** (ETL pipelines)
- ✅ **Event-driven workflows** (process S3 uploads)
- ✅ **Scheduled tasks** (cron jobs)
- ❌ **Long-running tasks** (>15 min - use ECS/Fargate)
- ❌ **Stateful applications** (Lambda is stateless)

#### **6. Lambda vs EC2**

| Feature | Lambda | EC2 |
|---------|--------|-----|
| **Management** | Fully managed | I manage OS/server |
| **Scaling** | Automatic, instant | Manual or auto-scaling groups |
| **Pricing** | Pay per execution | Pay per hour (even idle) |
| **Startup** | Cold start (1-3s) | Always running |
| **Best for** | Sporadic, event-driven | Steady, predictable load |
| **State** | Stateless | Can maintain state |

---

## 🎯 **PART 2: My ML Model on Lambda**

### **Why Lambda for ML Inference?**

**Advantages:**
- ✅ No server management
- ✅ Auto-scales (1 request → 1000 requests seamlessly)
- ✅ Pay only when used (not 24/7 like EC2)
- ✅ Sub-second cold start with lightweight models
- ✅ Easy to version and rollback

**Challenges:**
- ⚠️ Package size limits (250 MB unzipped)
- ⚠️ Memory constraints (need ~2 GB for XGBoost)
- ⚠️ Cold starts (first request slower)
- ⚠️ No GPU support (use SageMaker for deep learning)

### **My Architecture:**

```
User Request (JSON)
      ↓
API Gateway (optional)
      ↓
Lambda Function
  - Load models (~500ms first time, cached after)
  - Engineer features (~10ms)
  - Predict (~50ms)
  - Return JSON
      ↓
Response (virality score + tier)
```

---

## 🛠️ **PART 3: Step-by-Step Deployment**

---

### **STEP 1: Understanding What I'm Deploying**

**What goes in the Lambda package:**
```
lambda_package/
├── lambda_handler.py          # Entry point (my code)
├── predict.py                 # My predictor class
├── models/                    # Trained models (~20 MB)
├── artifacts/                 # Scaler, encoders (~1 MB)
├── python/                    # Dependencies
│   ├── numpy/                 # (~30 MB)
│   ├── pandas/                # (~40 MB)
│   ├── scikit-learn/          # (~30 MB)
│   ├── xgboost/               # (~50 MB)
│   └── ... (total ~150 MB)
```

**Total size:** ~170 MB (within 250 MB limit ✅)

---

### **STEP 2: Lambda Handler**

**What is a handler?**
- The entry point AWS calls when Lambda is invoked
- Takes `event` (input data) and `context` (runtime info)
- Returns response in specific format
- The lambda_handler is AWS's entry point - it receives the event and context from AWS runtime. My application code (like predict.py) is called BY the handler. The handler acts as an adapter between AWS's invocation format and my business logic.

**My handler:**
```python
def lambda_handler(event, context):
    # 1. Parse input JSON
    video_data = json.loads(event['body'])
    
    # 2. Call my predictor
    result = predictor.predict(video_data)
    
    # 3. Format response
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

---

### **STEP 3: Install AWS CLI**

**What is AWS CLI?**
Command-line tool to interact with AWS services without using the web console.

**Install:**
```bash
# macOS
brew install awscli

# Or via Python
pip install awscli

# Verify
aws --version
```

**Configure credentials:**
```bash
aws configure
```

**I need:**
- AWS Access Key ID (from IAM console)
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Output format: `json`

**Note:** IAM users are for people/services with long-term credentials (access keys). IAM roles are for temporary credentials assumed by AWS services (like Lambda). Lambda functions use roles, not users, to access other AWS services like S3 or DynamoDB.

---

### **STEP 4: Package Dependencies**

**Why this is needed:**
Lambda doesn't have numpy, pandas, scikit-learn pre-installed. I must include ALL dependencies in my deployment package. Lambda environments are immutable and isolated for security and performance. Dependencies must be packaged ahead of time. This ensures consistent behavior and faster cold starts since packages are pre-compiled for the Lambda runtime environment (Amazon Linux).

**How I package:**
```bash
# Create package directory
mkdir lambda_package
cd lambda_package

# Install dependencies to this directory
pip install -t python/ numpy pandas scikit-learn xgboost

# Why 'python/' folder?
# Lambda adds this to Python path automatically
```

**Note:** Lambda Layers are an alternative for sharing dependencies across functions.

---

### **STEP 5: Create Deployment Package**

**Package structure requirements:**
```
lambda_deployment.zip
├── lambda_handler.py      # My handler (must be at root)
├── predict.py             # Supporting code
├── models/                # My trained models
├── artifacts/             # Preprocessing objects
└── python/                # Dependencies (auto-loaded by Lambda)
    ├── numpy/
    ├── pandas/
    └── ...
```

**Why this structure?**
Lambda expects handler at root level and looks for dependencies in `python/` folder.

**Create zip:**
```bash
zip -r lambda_deployment.zip . -x "*.git*" -x "*__pycache__*"
```

**Note:** If package exceeds 250 MB, I can use Lambda Layers (up to 5 layers, 250 MB total) for dependencies, or use container images (up to 10 GB) instead of zip deployment. For ML models, I can also load from S3 at runtime.

---

### **STEP 6: Create IAM Role for Lambda**

**What is this?**
Permissions that define what my Lambda function can do (access S3, write logs, etc.)

**Required permissions:**
- `AWSLambdaBasicExecutionRole` - Write logs to CloudWatch
- (Optional) `AmazonS3ReadOnlyAccess` - If loading models from S3

**How I create:**

**Option A: AWS Console**
1. Go to IAM → Roles → Create Role
2. Select "AWS Service" → "Lambda"
3. Attach `AWSLambdaBasicExecutionRole`
4. Name: `TikTokViralityLambdaRole`

**Option B: AWS CLI**
```bash
aws iam create-role \
  --role-name TikTokViralityLambdaRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name TikTokViralityLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

**Note:** I follow the principle of least privilege - grant only the minimum permissions needed. My Lambda only needs CloudWatch logs, so I only attach basic execution role - not admin access.

---

### **STEP 7: Create Lambda Function**

**AWS Console Method:**
1. Navigate to Lambda service
2. Click "Create function"
3. Choose "Author from scratch"
4. Configuration:
   - **Function name:** `tiktok-virality-predictor`
   - **Runtime:** Python 3.11
   - **Architecture:** x86_64
   - **Execution role:** Use existing `TikTokViralityLambdaRole`
5. Click "Create function"

**AWS CLI Method:**
```bash
aws lambda create-function \
  --function-name tiktok-virality-predictor \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/TikTokViralityLambdaRole \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda_deployment.zip \
  --timeout 30 \
  --memory-size 2048
```

**Note:** Handler format is `filename.function_name`. For `lambda_handler.lambda_handler`, AWS imports `lambda_handler.py` and calls the `lambda_handler()` function. This tells Lambda where my code execution begins.

---

### **STEP 8: Configure Lambda Settings**

**Critical settings:**

#### **Memory** 
- **Set to:** 2048 MB (2 GB)
- **Why:** XGBoost models + numpy/pandas need ~1.5 GB
- **Note:** More memory = faster CPU (not just RAM!)

#### **Timeout**
- **Set to:** 30 seconds
- **Why:** Cold start ~2s + inference ~0.5s = safe with buffer
- **Default is only 3 seconds!**

#### **Environment Variables** (optional)
```
MODEL_VERSION=20251210_233411
LOG_LEVEL=INFO
```

**How I set (Console):**
- Configuration tab → General configuration → Edit
- Memory: 2048 MB
- Timeout: 30 sec

**How I set (CLI):**
```bash
aws lambda update-function-configuration \
  --function-name tiktok-virality-predictor \
  --memory-size 2048 \
  --timeout 30
```

---

### **STEP 9: Upload Deployment Package**

**If package < 50 MB:** Upload via console (drag & drop)

**If package > 50 MB:** Use S3 + CLI

```bash
# Upload to S3 first
aws s3 cp lambda_deployment.zip s3://my-bucket/lambda_deployment.zip

# Update function code from S3
aws lambda update-function-code \
  --function-name tiktok-virality-predictor \
  --s3-bucket my-bucket \
  --s3-key lambda_deployment.zip
```

**Note:** Console upload is limited to 50 MB. S3 allows up to 250 MB unzipped. For even larger deployments, I'd use container images with ECR instead of zip files.

---

### **STEP 10: Test Lambda**

**Method 1: Console Test**
1. Go to Lambda function
2. Click "Test" tab
3. Create test event:
```json
{
  "body": "{\"text_hook\": \"Test hook\", \"caption\": \"#test\", \"main_category\": \"Business\", \"subcategory\": \"HR & Payroll\", \"length\": 7.0}"
}
```
4. Click "Test"
5. Check response

**Method 2: AWS CLI**
```bash
aws lambda invoke \
  --function-name tiktok-virality-predictor \
  --payload '{"body": "{...}"}' \
  response.json

cat response.json
```

**Expected response:**
```json
{
  "statusCode": 200,
  "body": "{\"virality_score\": 0.42, \"viral_tier\": \"viral\", ...}"
}
```

---

### **STEP 11: Monitor & Debug**

**CloudWatch Logs:**
- Every Lambda execution creates logs
- View: CloudWatch → Log groups → `/aws/lambda/tiktok-virality-predictor`
- Shows: Print statements, errors, execution time, memory usage

**Common errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `Task timed out` | Timeout too short | Increase timeout setting |
| `Memory exceeded` | Not enough RAM | Increase memory allocation |
| `Module not found` | Missing dependency | Check package structure |
| `Permission denied` | Wrong IAM role | Add required permissions |

**Note:** I use CloudWatch Logs for execution logs, X-Ray for distributed tracing, and print statements liberally. I test locally first with sam local or docker containers that mimic Lambda environment.

---

### **STEP 12: (Optional) Add API Gateway**

**What is API Gateway?**
Service that creates HTTP/REST APIs that trigger Lambda functions. Turns my Lambda into a public API endpoint.

**Why use it?**
- Get a public HTTPS URL
- Handle authentication/API keys
- Rate limiting and throttling
- Request/response transformation

**Quick setup:**
1. Lambda console → Add trigger → API Gateway
2. Create new API → REST API
3. Security: Open (or API key)
4. Deploy

**I get:**
```
https://abc123.execute-api.us-east-1.amazonaws.com/default/tiktok-virality-predictor
```

**Test:**
```bash
curl -X POST https://my-api-url \
  -H "Content-Type: application/json" \
  -d '{"text_hook": "...", "caption": "...", ...}'
```

**Note:** API Gateway is the front door that receives HTTP requests. Lambda is the backend that processes them. API Gateway handles routing, authentication, rate limiting. Lambda handles business logic. I need both for a complete REST API.

---

## 📊 **PART 4: Performance Optimization**

### **Cold Start Optimization:**

```python
# GOOD: Initialize once (global scope)
import pickle
model = pickle.load(open('model.pkl', 'rb'))  # Loaded once per container

def lambda_handler(event, context):
    prediction = model.predict(data)  # Fast
    return prediction

# BAD: Initialize every time
def lambda_handler(event, context):
    model = pickle.load(open('model.pkl', 'rb'))  # Loaded every request!
    prediction = model.predict(data)
    return prediction
```

**My handler uses global scope correctly!**

### **Memory vs Cost Trade-off:**

**Optimization tip:** If my Lambda is slow, I first try increasing memory - Lambda allocates CPU proportionally to memory. A 2 GB function gets 2x CPU of a 1 GB function. Often, doubling memory halves execution time, resulting in same cost but better latency. I profile with CloudWatch to find optimal memory setting.

---

## 🚀 **PART 5: Deployment Files**

### **Prerequisites Checklist:**
- [ ] AWS account created
- [ ] AWS CLI installed (`aws --version`)
- [ ] AWS credentials configured (`aws sts get-caller-identity`)
- [ ] Models trained (check `models/` folder exists)
- [ ] Python 3.11 installed locally

### **Deployment Files I Use:**

1. `lambda_handler.py` - AWS entry point
2. `requirements.txt` - Dependencies list
3. `package_lambda.sh` - Automated packaging script
4. `test_local.py` - Test before uploading
5. `deploy.sh` - Automated deployment script

Each file has detailed comments explaining what it does and why.

---

## 🎯 **Summary**

This deployment process allows me to:
- Package my XGBoost models with all dependencies
- Deploy to AWS Lambda for serverless inference
- Scale automatically from 1 to 1000+ requests
- Pay only for actual usage
- Monitor performance via CloudWatch

The key challenge was package size (170 MB) which I kept under the 250 MB limit by using only essential dependencies. Cold start is ~500ms due to model loading, which I mitigated by keeping models in global scope. The function runs on 2 GB memory with 30-second timeout, processing predictions in ~50ms after warm-up.
