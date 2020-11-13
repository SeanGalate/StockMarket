# Extracted stock data from Yahoo Finance with Python and utilized visualizations to analyze various tech stocks 

# Data analysis and visualization imports
from __future__ import division
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read stock data from yahoo
from pandas_datareader import DataReader, wb

# Time stamps
from datetime import datetime

# Visualization style
sns.set_style('whitegrid')

# Tech stocks in this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# End and start dates for stock data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# For loop gathering finance data and setting as a DataFrame

for stock in tech_list:
    globals()[stock] = DataReader(stock, 'yahoo', start, end)

# AAPL summary
print(AAPL.describe())
print(AAPL.info())

# Historical view of closing price
AAPL['Adj Close'].plot(legend=True, figsize=(10, 4))
plt.show()

# Historical view of stock being traded
AAPL['Volume'].plot(legend=True, figsize=(10, 4))
plt.show()

# Moving averages
ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = "MA for %s days" % (str(ma))
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma).mean()

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize=(10, 4))
plt.show()

# Percentage change for each day
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

# Daily return percentage
AAPL['Daily Return'].plot(figsize=(12, 4), legend=True, linestyle='--', marker='o')
plt.show()

# Daily return histogram
sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')
plt.show()

# All the closing prices for tech stock list
closing_df = DataReader(['AAPL', 'GOOG', 'MSFT', 'AMZN'], 'yahoo', start, end)['Adj Close']
print(closing_df.head(5))

# New Tech Returns DataFrame
tech_rets = closing_df.pct_change()

# Joinplot to compare daily returns of Google and Microsoft
sns.jointplot('GOOG', 'MSFT', tech_rets, kind='scatter')
plt.show()

# Pairplot for visual analysis of all comparisons
sns.pairplot(tech_rets.dropna())
plt.show()

# Show the relationship on daily returns between all stocks
# Call pairplot on the DataFrame
returns_fig = sns.PairGrid(tech_rets.dropna())

# Specify upper triangle format
returns_fig.map_upper(plt.scatter, color='purple')

# Specify lower triangle format
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Diagonal is a series of histogram plots of the daily return
returns_fig.map_diag(plt.hist, bins=30)
plt.show()

# Correlation chart for the daily returns
tech_rets_corr = tech_rets.dropna().corr()
sns.heatmap(tech_rets_corr, annot=True, fmt='.2f')
plt.show()

# Risk analysis: Daily percentage returns vs expected returns
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(), s=area)

# X and Y limits of the plot
plt.xlabel('Expected returns')
plt.ylabel('Risk')


for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy=(x, y), xytext=(10, 10),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='-'))
plt.show()
