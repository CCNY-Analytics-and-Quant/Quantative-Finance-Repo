import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os


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
    'AZO': 0.377251, 'AMD': 0.046532
}

weights_maximum_risk = {
    'TSLA': 0.013270, 'MRNA': 0.155588, 'NVDA': 0.250389, 'SMCI': 0.150165,
    'NVR': 0.022550, 'ENPH': 0.073423, 'BLDR': 0.019092, 'CMG': 0.022661,
    'AZO': 0.244763, 'AMD': 0.048099}

    
# Calculate daily portfolio returns for both scenarios
portfolio_returns_minimal_risk = (yf_data.pct_change() * pd.Series(weights_minimal_risk)).sum(axis=1)
portfolio_returns_maximum_risk = (yf_data.pct_change() * pd.Series(weights_maximum_risk)).sum(axis=1)


# Calculate cumulative returns to see the overall performance
cumulative_returns_minimal_risk = ((1 + portfolio_returns_minimal_risk).cumprod() - 1)*100
cumulative_returns_maximum_risk = ((1 + portfolio_returns_maximum_risk).cumprod() - 1)*100
print ("Max Risk portfolio returns:")
print (cumulative_returns_maximum_risk)
print ("Min Risk portfolio returns:")
print (cumulative_returns_minimal_risk)

print (f"Avegare Weekly Max Risk Portfolio Returns from Project Completion to Current Date ({weeks} weeks): ")
print (cumulative_returns_maximum_risk[-1]/weeks)

print (f"Avegare Weekly Min Risk Portfolio Returns from Project Completion to Current Date ({weeks} weeks): ")
print (cumulative_returns_minimal_risk[-1]/weeks)


import json

output_dir = "/tmp/new_output"
os.makedirs(output_dir, exist_ok=True)

# Define the full path for the JSON file
json_file_path = os.path.join(output_dir, 'updated_portfolio_returns.json')

# Assuming 'portfolio_returns', 'cumulative_returns_maximum_risk', 'cumulative_returns_minimal_risk', and 'weeks'
# are the variables containing the data you want to output
output_data = {
    #"Portfolio 1 Returns": portfolio_returns[-1],
    #"Portfolio 1 Average Weekly Returns": portfolio_returns[-1] / weeks,
    "Max Risk Portfolio Returns": cumulative_returns_maximum_risk[-1],
    "Max Risk Portfolio Average Weekly Returns": cumulative_returns_maximum_risk[-1] / weeks,
    "Min Risk Portfolio Returns": cumulative_returns_minimal_risk[-1],
    "Min Risk Portfolio Average Weekly Returns": cumulative_returns_minimal_risk[-1] / weeks,
    #"Difference in E.W Portfolio and MaxRisk Portfolio": cumulative_returns_maximum_risk[-1] - portfolio_returns[-1],
    
}

# Convert the data to a JSON string
json_data = json.dumps(output_data, indent=4)

# Write the JSON data to the specified file
with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)

# Confirm the path where the file was saved
print(f"File saved to {json_file_path}")