import os
import json
import boto3
from datetime import datetime
import uuid

# Grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']  # Add the DynamoDB table name to environment variables


# AWS clients
runtime = boto3.client('runtime.sagemaker')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)  # Connect to DynamoDB table

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    # Parse input data from the API Gateway event
    body = json.loads(event['body'])
    
    # The input data is in the 'data' field, which contains an array of arrays
    if "data" in body:
        data = body['data']
    else:
        data = body
    
    # Format the data as a CSV string for SageMaker
    formatted_data = ",".join(map(str, data[0]))  # Flatten the list and convert to CSV format
    
    print(f"Formatted input data for SageMaker: {formatted_data}")
    
    try:
        # Invoke the SageMaker endpoint for prediction
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=str(data)  # Send the data as CSV
        )
        
        # Parse the response
        result = response['Body'].read().decode()[1]
        print("Model prediction result:", result)
        
        # Assuming the model returns a single prediction value (e.g., 0, 1, 2, etc.)
        try:
            pred = int(result.strip())  # Clean any leading/trailing whitespace and convert to int
        except ValueError:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Invalid prediction result format',
                    'details': result
                })
            }

        # Map prediction to a human-readable label
        equivalent = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
        }
        
        predicted_label = equivalent.get(pred, "Unknown")
        
        # Generate a unique Prediction ID and timestamp
        prediction_id = str(uuid.uuid4())  # Unique identifier for each prediction
        timestamp = datetime.now().isoformat()  # ISO 8601 format timestamp
        
        # Prepare input features and log data into DynamoDB
        input_features = data[0]  # Assuming the first array is the feature set
        response = table.put_item(
            Item={
                'PredictionID': prediction_id,
                'InputFeatures': json.dumps(input_features),
                'Prediction': predicted_label,
                'Timestamp': timestamp
            }
        )
        
        print("DynamoDB PutItem Response:", response)
        
        # Return the prediction to the client
        return {
            'statusCode': 200,
            'body': json.dumps({
                'PredictionID': prediction_id,
                'Prediction': predicted_label,
                'Timestamp': timestamp
            })
        }
    
    except Exception as e:
        print(f"Error invoking SageMaker endpoint: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Error invoking SageMaker endpoint',
                'details': str(e)
            })
        }
