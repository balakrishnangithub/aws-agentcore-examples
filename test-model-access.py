import boto3
from botocore.exceptions import ClientError

# Common Claude model IDs to test
models_to_test = [
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

# Create Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-west-2")

print("Testing model access...\n")

for model_id in models_to_test:
    try:
        # Make a minimal API call to test access
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"maxTokens": 10}
        )
        print(f"✓ {model_id} - ACCESS GRANTED")
    except ClientError as e:
        if "AccessDeniedException" in str(e):
            print(f"✗ {model_id} - ACCESS DENIED")
        else:
            print(f"? {model_id} - ERROR: {e.response['Error']['Code']}")
