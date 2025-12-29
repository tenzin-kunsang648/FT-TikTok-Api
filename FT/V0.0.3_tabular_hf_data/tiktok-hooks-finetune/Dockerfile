# Lambda Container Image - TikTok Virality Prediction
# Use AWS's official Python Lambda base image
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy and install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Copy trained models
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Copy preprocessing artifacts  
COPY artifacts/ ${LAMBDA_TASK_ROOT}/artifacts/

# Fix permissions (THIS IS THE CRITICAL LINE!)
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_handler.lambda_handler"]