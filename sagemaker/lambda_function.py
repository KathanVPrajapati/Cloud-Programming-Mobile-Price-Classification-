import os
import io
import boto3
import json
import csv
import base64
from urllib.parse import unquote

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

# AWS clients
runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    print(data)
    payload = data['body']

    print("Payload ======== ", payload, type(payload))
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
                                       
    result = json.loads(response['Body'].read().decode())
    print("Result ===", result)
    pred = int(result[0])
    
    equivalent = {0:"Low Cost",
                  1:"Medium Cost",
                  2:"High Cost",
                  3:"Very High Cost"}
                  
    predicted_label = equivalent[pred]
    
        

    return predicted_label
    