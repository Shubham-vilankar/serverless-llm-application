# lambda_function.py
import json
import boto3
import uuid
from datetime import datetime

sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")
dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
table = dynamodb.Table("phi3-llmproject-logs")

ENDPOINT_NAME = "phi3-llmproject-endpoint-final"  # will need to updated enpoint name after eachb deployment.

def lambda_handler(event, context):
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    try:
        body = json.loads(event.get("body", "{}"))
        user_prompt = body.get("prompt", "")

        if not user_prompt:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'prompt' in request body"})
            }

        formatted_prompt = f"### Instruction:\n{user_prompt}\n\n### Response:\n"
        payload = {"inputs": formatted_prompt}

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())
        generated_text = result.get("generated_text", "")

        # Log successful request to DynamoDB
        table.put_item(Item={
            "request_id": request_id,
            "timestamp": timestamp,
            "prompt": user_prompt,
            "response": generated_text,
            "status": "success",
        })

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": generated_text})
        }

    except Exception as e:
        # Log failed request....(e.g., endpoint not running)
        table.put_item(Item={
            "request_id": request_id,
            "timestamp": timestamp,
            "prompt": user_prompt if 'user_prompt' in locals() else "unknown",
            "error": str(e),
            "status": "error",
        })

        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }