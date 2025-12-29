"""
Local Lambda Testing Script

Purpose: Test your Lambda handler locally BEFORE deploying to AWS
Why: Catch bugs locally (free, fast) instead of debugging in AWS (slower, harder)

This simulates the Lambda environment on your local machine.
"""

import json 
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock Lambda context
class MockLambdaContext:
    """
    Simulates AWS Lambda context object
    
    What information does Lambda context provide?
    
    Answer: Runtime metadata like function name, memory limit, request ID,
    remaining execution time. Used for logging, monitoring, and adaptive
    timeout handling (e.g., stop processing if time running out).
    """
    def __init__(self):
        self.function_name = "tiktok-virality-predictor-local"
        self.function_version = "$LATEST"
        self.invoked_function_arn = "arn:aws:lambda:local:123456789:function:test"
        self.memory_limit_in_mb = 2048
        self.request_id = "local-test-request-id"
        self.log_group_name = "/aws/lambda/local-test"
        self.log_stream_name = "local-test-stream"
    
    def get_remaining_time_in_millis(self):
        """Simulates 30 second timeout"""
        return 30000

def test_api_gateway_format():
    """
    Test with API Gateway event format (most common)
    
    API Gateway wraps your JSON in event['body'] as a string.
    This is the format you'll use in production.
    """
    print("\n" + "="*80)
    print("TEST 1: API Gateway Format")
    print("="*80)
    
    from lambda_handler import lambda_handler
    
    # Simulate API Gateway event
    event = {
        'body': json.dumps({
            'text_hook': 'When the interviewer asks if you have a background in finance',
            'caption': 'I fear this would work on me as an interviewer #corporatehumor',
            'main_category': 'Business',
            'subcategory': 'HR & Payroll',
            'length': 7.0
        }),
        'headers': {
            'Content-Type': 'application/json'
        },
        'httpMethod': 'POST'
    }
    
    context = MockLambdaContext()
    
    print("\nInput Event:")
    print(json.dumps(event, indent=2))
    
    print("\n🔮 Invoking Lambda handler...")
    response = lambda_handler(event, context)
    
    print("\n" + "="*80)
    print("RESPONSE")
    print("="*80)
    print(f"Status Code: {response['statusCode']}")
    print(f"\nBody:")
    
    body = json.loads(response['body'])
    print(json.dumps(body, indent=2))
    
    # Validate response
    assert response['statusCode'] == 200, "Expected 200 status code"
    assert 'virality_score' in body, "Missing virality_score in response"
    assert 'viral_tier' in body, "Missing viral_tier in response"
    
    print("\n✅ Test passed!")
    return response

def test_direct_invoke_format():
    """
    Test with direct invocation format
    
    When you invoke Lambda directly (not via API Gateway), event IS the payload.
    """
    print("\n" + "="*80)
    print("TEST 2: Direct Invocation Format")
    print("="*80)
    
    from lambda_handler import lambda_handler
    
    # Direct invoke - no 'body' wrapper
    event = {
        'text_hook': 'How neural networks actually learn',
        'caption': 'Deep dive into backpropagation #AI #machinelearning #deeplearning',
        'main_category': 'Education',
        'subcategory': 'Technology',
        'length': 45.0,
        'upload_dayofweek': 2,  # Wednesday
        'upload_month': 12,
        'is_weekend': 0
    }
    
    context = MockLambdaContext()
    
    print("\nInput Event:")
    print(json.dumps(event, indent=2))
    
    print("\n🔮 Invoking Lambda handler...")
    response = lambda_handler(event, context)
    
    print("\n" + "="*80)
    print("RESPONSE")
    print("="*80)
    
    body = json.loads(response['body'])
    print(json.dumps(body, indent=2))
    
    print("\n✅ Test passed!")
    return response

def test_error_handling():
    """
    Test error handling with invalid input
    
    Important: Lambda should gracefully handle errors and return proper status codes.
    """
    print("\n" + "="*80)
    print("TEST 3: Error Handling")
    print("="*80)
    
    from lambda_handler import lambda_handler
    
    # Missing required fields
    event = {
        'body': json.dumps({
            'text_hook': 'Test hook',
            # Missing: caption, main_category, length
        })
    }
    
    context = MockLambdaContext()
    
    print("\n🔮 Testing with incomplete data...")
    response = lambda_handler(event, context)
    
    print(f"\nStatus Code: {response['statusCode']}")
    body = json.loads(response['body'])
    print(f"Error: {body.get('error', 'No error')}")
    
    assert response['statusCode'] == 400, "Should return 400 for bad request"
    assert 'error' in body, "Should include error message"
    
    print("\n✅ Error handling works correctly!")
    return response

def test_multiple_predictions():
    """
    Test multiple predictions to verify consistency
    
    Does Lambda maintain state between invocations?
    
    Answer: Within same container (warm start), global variables persist.
    But Lambda can spin up multiple containers for concurrent requests.
    Never rely on state - design for stateless execution.
    """
    print("\n" + "="*80)
    print("TEST 4: Multiple Predictions (Consistency Check)")
    print("="*80)
    
    from lambda_handler import lambda_handler
    
    context = MockLambdaContext()
    
    videos = [
        {'text_hook': 'Test 1', 'caption': 'Caption 1 #test', 'main_category': 'Music', 'subcategory': 'Pop', 'length': 10},
        {'text_hook': 'Test 2', 'caption': 'Caption 2 #test', 'main_category': 'Education', 'subcategory': 'Science', 'length': 20},
        {'text_hook': 'Test 3', 'caption': 'Caption 3 #test', 'main_category': 'Entertainment', 'subcategory': 'Comedy', 'length': 15},
    ]
    
    results = []
    for i, video in enumerate(videos, 1):
        event = {'body': json.dumps(video)}
        response = lambda_handler(event, context)
        body = json.loads(response['body'])
        results.append(body)
        print(f"  Video {i}: score={body['virality_score']:.4f}, tier={body['viral_tier']}")
    
    print("\n✅ All predictions completed successfully!")
    return results

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LOCAL LAMBDA TESTING SUITE")
    print("="*80)
    print("\nThis simulates AWS Lambda environment on your local machine.")
    print("Use this to validate your handler before deploying.\n")
    
    try:
        # Run all tests
        test_api_gateway_format()
        test_direct_invoke_format()
        test_error_handling()
        test_multiple_predictions()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYour Lambda handler is ready for deployment!")
        print("\nNext step: Run package_lambda.sh to create deployment zip")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        print("\nFix the error before deploying to AWS!")
        sys.exit(1)
