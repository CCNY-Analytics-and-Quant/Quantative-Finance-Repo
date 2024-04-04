import boto3
import json

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    bucket_name = 'jcb2001dataport'
    file_key = 'new_output/updated_portfolio_returns.json'
    
    try:
        file = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = file['Body'].read().decode('utf-8')
        data = json.loads(file_content)
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps(data)
        }
    except Exception as e:
        print(e)
        return {
            'statusCode': 500,
            'body': json.dumps('Error fetching data from S3')
        }