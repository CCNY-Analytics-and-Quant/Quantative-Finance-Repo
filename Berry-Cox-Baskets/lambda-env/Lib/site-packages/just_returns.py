import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import json
import boto3

def lambda_handler(event, context):
    tickers = ['TSLA', 'MRNA', 'NVDA', 'SMCI', 'NVR', 'ENPH', 'BLDR', 'CMG', 'AZO', 'AMD']
    current_date1 = datetime.now()
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    project_completion_date = datetime(2023, 11, 15)
    yf_data = yf.download(tickers, start='2023-11-15', end=current_date)['Adj Close']

    

    # difference between the two dates
    difference_days = current_date1 - project_completion_date 

    # number of weeks
    weeks = difference_days.days // 7

    weights_minimal_risk = {
    'TSLA': 0.051118, 'MRNA': 0.101076, 'NVDA': 0.032128, 'SMCI': 0.010970,
    'NVR': 0.293823, 'ENPH': 0.004250, 'BLDR': 0.002243, 'CMG': 0.080609,
    'AZO': 0.377251, 'AMD': 0.046532}


    weights_maximum_risk = {
    'TSLA': 0.013270, 'MRNA': 0.155588, 'NVDA': 0.250389, 'SMCI': 0.150165,
    'NVR': 0.022550, 'ENPH': 0.073423, 'BLDR': 0.019092, 'CMG': 0.022661,
    'AZO': 0.244763, 'AMD': 0.048099}

    equal_weights = {
    'TSLA': 0.1, 'MRNA': 0.1, 'NVDA': 0.1, 'SMCI': 0.1,
    'NVR': 0.1, 'ENPH': 0.1, 'BLDR': 0.1, 'CMG': 0.1,
    'AZO': 0.1, 'AMD': 0.1}

    
    # Calculate daily portfolio returns for both scenarios
    portfolio_returns_equal_weights = (yf_data.pct_change() * pd.Series(equal_weights)).sum(axis=1)
    portfolio_returns_minimal_risk = (yf_data.pct_change() * pd.Series(weights_minimal_risk)).sum(axis=1)
    portfolio_returns_maximum_risk = (yf_data.pct_change() * pd.Series(weights_maximum_risk)).sum(axis=1)


    # Calculate cumulative returns to see the overall performance
    cumulative_returns_equal_weights = ((1 + portfolio_returns_equal_weights).cumprod() - 1)*100
    cumulative_returns_minimal_risk = ((1 + portfolio_returns_minimal_risk).cumprod() - 1)*100
    cumulative_returns_maximum_risk = ((1 + portfolio_returns_maximum_risk).cumprod() - 1)*100

    output_dir = "/tmp/new_output"
    os.makedirs(output_dir, exist_ok=True)

    json_file_path = os.path.join(output_dir, 'updated_portfolio_returns.json')

    
    
    output_data = {
        "Equal Weights Portfolio Returns": cumulative_returns_equal_weights[-1],
        "Equal Weights Portfolio Average Weekly Returns": cumulative_returns_equal_weights[-1] / weeks,
        "Max Risk Portfolio Returns": cumulative_returns_maximum_risk[-1],
        "Max Risk Portfolio Average Weekly Returns": cumulative_returns_maximum_risk[-1] / weeks,
        "Min Risk Portfolio Returns": cumulative_returns_minimal_risk[-1],
        "Min Risk Portfolio Average Weekly Returns": cumulative_returns_minimal_risk[-1] / weeks,
        "Difference in Returns between Max-Risk and Equal-Weight Portfolios": cumulative_returns_maximum_risk[-1] - cumulative_returns_equal_weights[-1],
        "Difference in Returns between Max-Risk and Equal-Weight Portfolios (Weekly)": (cumulative_returns_maximum_risk[-1] - cumulative_returns_equal_weights[-1]) / weeks
    }

    # Convert your data to JSON format
    json_data = json.dumps(output_data)
    
    # Initialize a boto3 client
    s3 = boto3.client('s3')
    
    # Define the bucket name and the file key (name)
    bucket_name = 'jcb2001dataport'
    file_key = f'updated_portfolio_returns.json'
    
    # Upload the file
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=json_data)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Data successfully uploaded to S3!')
    }

