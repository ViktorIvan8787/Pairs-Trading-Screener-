import yfinance as yf
import numpy as np
import pandas as pd 
import statsmodels
import matplotlib.pyplot as plt


# S&P500 Financial Sector Companies
sp = [
    # Big Banks
    'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'HBAN', 'RF', 'CFG', 'MTB', 'KEY', 'CMA', 'ZION', 'FHN',
    # Investment Banks 
    'GS', 'MS', 'SCHW', 'AXP', 'RJF', 'AMTD',
    # Insurance 
    'BRK-B', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'CB', 'AIG', 'HIG','LNC', 'UNM', 'GL', 'AIZ', 'RE', 'EG', 'WRB', 'RNR', 'CINF',
    # Asset Management
    'BLK', 'BEN', 'IVZ', 'AMG', 'TROW', 'VCTR', 'WDR',
    # Market Infrastructure
    'CME', 'ICE', 'NDAQ', 'CBOE', 'MKTX', 'VIRT',
    # Fintech 
    'V', 'MA', 'PYPL', 'FIS', 'FISV', 'GPN', 'WEX', 'JKHY',
    # Real Estate 
    'SFI', 'OPEN',
    # Consumer 
    'DFS', 'SYF', 'ALLY', 'OMF', 'CACC'
]


# Download data for all S&P Financial Sector companies
data = yf.download(sp, start='2020-01-01', end='2024-01-01')['Close']
data = data.dropna(axis=1, how='any')

# Create a dictionary for the companies that have a correlation
correlated = []

# Iterate through and check correlation
from itertools import combinations

for i, j in combinations(data.columns, 2):
    correlation = data[i].corr(data[j])
    if correlation > 0.8:
        correlated.append({'stock1': i, 'stock2': j, 'correlation': correlation})


        
# Calculate the hedge ratio using Ordinary Least Squares for each pair
# And also perform Augmented Dickey-Fuller tests (ADF) as 5 percent level to confirm stationary relationships
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

for pair in correlated:
    stock1 = pair['stock1']
    stock2 = pair['stock2']

    # Add a constant to the first stock table and fit with OLS using stock1 as y-axis and stock2 as x-axis
    X = add_constant(data[stock2])
    model = OLS(data[stock1], X).fit()

    # Finding hedge ratio from slope of OLS line and calculating spread
    hedge_ratio = model.params[stock2]
    spread = data[stock1] - hedge_ratio * data[stock2]

    pair['hedge_ratio'] = hedge_ratio
    pair['spread'] = spread

    # Performing ADF tests on the spreads to see whether companies are mean-reverting
    adf_test_result = adfuller(spread.dropna())
    # p-value at adf_test_result[1]
    pair['adf_p_value'] = adf_test_result[1]

# Creating a new dictionary of pairs that includes only the ones that pass ADF test as 5 percent level

stationary_pairs = []

for pair in correlated:
    if pair['adf_p_value'] < 0.05:
        stationary_pairs.append(pair)

print(f"\nCorrelated pairs: {len(correlated)}")
print(f"Stationary pairs after ADF: {len(stationary_pairs)}\n")



# BackTesting
# Calculating z-score, total_returns (with transaction cost of 1%), sharpe_ratio, and max drawdowns
# Will use a z-score of 1.5 to long/short

results = []

for pair in stationary_pairs:
    stock1 = pair['stock1']
    stock2 = pair['stock2']
    hedge_ratio = pair['hedge_ratio']
    spread = pair['spread']



    # Calculating z-score
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    # Creating dataframe of signals. Original signal is always 0 (signifies no purchase and therefore no overall return)
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0
    # Using z-score of 1.5
    signals.loc[z_score > 1.5, 'signal'] = -1
    signals.loc[z_score < -1.5, 'signal'] = 1



    # Finding total returns
    returns = data.pct_change()
    # The shift prevents lookahead bias
    signals['pair_returns'] = (signals['signal'].shift(1) * (returns[stock1] - hedge_ratio * returns[stock2]))
    # Adding transaction cost of 0.001 (0.1%)
    transaction_cost = 0.001
    trades = signals['signal'].diff().abs() / 2 # Checks wether trade occured
    signals['pair_returns'] -= (trades * transaction_cost)
    # Computing total return - setting cumulative returns then taking out the final value (total return)
    signals['cumulative_returns'] = (1 + signals['pair_returns']).cumprod()
    total_return = signals['cumulative_returns'].iloc[-1] - 1



    # Finding sharpe_ratio 
    # 252 trading days in a year. Simplified Sharpe ratio with risk-free rate set to zero. (Usually subtract risk-free rate using 3 month Treasury rate)
    sharpe_ratio = ( (signals['pair_returns'].mean() - 0) / signals['pair_returns'].std() ) * np.sqrt(252)

    # Finding max drawdown
    max_drawdown =  (signals['cumulative_returns'] / signals['cumulative_returns'].cummax() - 1).min()

    
    # Adding a test for after 2022 and test sharpe ratio to see whether the strategy holds
    test = signals['2022-01-01':]
    test_sharpe = ((test['pair_returns'].mean() - 0) / test['pair_returns'].std() ) * np.sqrt(252)

    
    results.append(
        {
        'stock1': stock1,
        'stock2': stock2,
        'correlation': pair['correlation'],
        'adf_p_value': pair['adf_p_value'],
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'test_sharpe': test_sharpe
    }
    )

# Turning into DataFrame and sorting by sharpe values
results_df = pd.DataFrame(results)
results_df_sharpe_sort = results_df.sort_values('sharpe_ratio', ascending=False)
results_df_test_sharpe_sort = results_df.sort_values('test_sharpe', ascending=False)

results_df_sharpe_sort.to_csv('screener_results_sharpe_sort.csv', index=False)
results_df_test_sharpe_sort.to_csv('screener_results_test_sharpe_sort.csv', index=False)

print("Sharpe sorted results saved to ' screener_results_sharpe_sort.csv ' .")
print("Test sharpe sorted results saved to ' screener_results_test_sharpe_sort.csv ' .")




