#!/usr/bin/env python
# coding: utf-8

# # Efficient Frontier Project
# by: Jean B and Alex G

# ### Import libraries

# In[1]:


import yfinance as yf
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import os
print(os.getcwd())


# ### Step 2: Retrieve daily open or close data on your assets for the previous 2 years

# In[2]:


tickers = ["NVDA","META", "COKE", "MSFT", "AAPL", "AMD", "GOOG"] 
#"NVDA", "META", "COKE", "MSFT", "AAPL", "AMD", "GOOG", "SBUX", "F", "TSLA", "BP", "EBAY", "AMZN", "ABNB"
etfs = ["SPY", "IWM", "DIA"]
assets = tickers + etfs

# This should be two years
data = yf.download(tickers, period = '10y')['Adj Close']
dataT = yf.download(tickers, period = '5y')['Adj Close']
""" Using 1 year for testing purpose
Using tickers instead of assets because portfolio only includes tickers"""
returns = data.pct_change().dropna()
returns


# ### Step 2: Retrieve daily open or close data on your assets for the previous 2 years

# In[17]:


data.tail()


# ### Step 3: Calculate the mean, variance, and correlation matrix for all assets

# In[18]:


# returns.mean()
# returns.var()

# Mean and Variance
table = pd.DataFrame(index=assets)
table['Mean'] = returns.mean()  # Calculating mean on returns
table['Variance'] = returns.var()  # Calculating variance on returns

table.round(4)  # Displaying the mean and variance table


# In[19]:


# Correlation matrix
plt.figure(figsize=(6, 6))
sn.heatmap(returns.corr(), annot=True, cmap="coolwarm")

#plt.savefig('resources/correlation-matrix.png', dpi=400)


# ## Source
# Monte Carlo Simulation 
# 
# https://www.youtube.com/watch?v=wlaLmM_LvWg&ab_channel=SpencerPao

# In[13]:


# where dataframe = yf.Download(basket of stocks, period = '10y')
class EfficientFrontier:
    def __init__(self, dataframe):
        self.df = dataframe
        self.basket = self.df.columns
        self.num_assets = len(self.basket)
        self.num_portfolios = 200000
        self.portfolio_returns = []
        self.portfolio_volatility = []
        self.sharpe_ratio = []
        self.stock_weights = []
        self.sharpe_portfolio = None
        self.min_variance_port = None

    def calculate_portfolio(self):
        expected_returns_a = self.df.pct_change()
        expected_returns_a.columns = self.df.columns
        expected_returns_aA = pd.DataFrame(expected_returns_a.mean()*250)
        expected_returns_aA = expected_returns_aA.T
        dar = self.df.pct_change().iloc[1:,:]+1
        gar = pd.DataFrame(np.prod(dar)**(1/float(6)) - 1)
        full_return_annual = (pd.concat([expected_returns_aA.T, gar], axis = 1))
        self.annual_returns = (expected_returns_a.mean() * 250) + 1  
        self.cov_daily = expected_returns_a.cov()  
        self.cov_annual = self.cov_daily * 250
        self.annual_returns.columns = ["Annual Returns:"]
        print(self.annual_returns)

    def plot_frontier(self):
        np.random.seed(3)
        for i in range(self.num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            returns = (np.dot(weights, (self.annual_returns)))
            volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_annual, weights)))
            sharpe = ((returns-1) / volatility)
            self.sharpe_ratio.append(sharpe)
            self.portfolio_returns.append(returns-1)
            self.portfolio_volatility.append(volatility)
            self.stock_weights.append(weights)

        portfolio = {'Returns': self.portfolio_returns,
                    'Volatility': self.portfolio_volatility,
                 'Sharpe Ratio': self.sharpe_ratio}

        for counter, symbol in enumerate(self.basket):
         portfolio[symbol+' Weight'] = [Weight[counter] for Weight in self.stock_weights]

        simulations = pd.DataFrame(portfolio)
        min_volatility = simulations['Volatility'].min()
        max_sharpe = simulations['Sharpe Ratio'].max()

        self.sharpe_portfolio = simulations.loc[simulations['Sharpe Ratio'] == max_sharpe]
        self.min_variance_port = simulations.loc[simulations['Volatility'] == min_volatility]

        plt.style.use('fivethirtyeight')
        simulations.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
              cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
        plt.scatter(x=self.sharpe_portfolio['Volatility'], y=self.sharpe_portfolio['Returns'], c='red', marker='D', s=200)
        plt.scatter(x=self.min_variance_port['Volatility'], y=self.min_variance_port['Returns'], c='blue', marker='D', s=200 )
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        plt.show()

    def get_optimal_portfolio(self):
        r_ef = pd.concat([self.min_variance_port.T,self.sharpe_portfolio.T], axis = 1)
        r_ef.columns = ["Minimum Risk Adjusted Values", "Max Risk Adjusted Values"]
        print(r_ef)


# In[15]:


ef = EfficientFrontier(data)
ef.calculate_portfolio() #Printing Annual Returns


# In[11]:


ef.plot_frontier()


# In[12]:


ef.get_optimal_portfolio()

