import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels 
import matplotlib.pyplot as plt

# =============== Graphs and stats for ================
# BlackRock (BLK) - Franklin Templeton (BEN),
# and 
# Citizens Financial Group (CFG) - PNC Financial Services (PNC)


# Pair trade function
def pair_trade_metrics(stock1, stock2, start_date, end_date, z_score_boundary):
    # ------------------------------------
    # Input: 2 stocks to pair trade
    # 
    # Donwloads the data for the stocks between the start and end period. Calculates correlation between two stocks. 
    # Calculates hedge ratio and spread using Ordinary Least Squares. 
    # Checks whether stocks are mean-reverting / stationary with Augmented Dickey-Fuller test at 5% level.
    # Calculates z-score and backtests trades bteween z_score_boundary - give sharpe ratio, total returns with transaction costs involved,
    # and give max drawdown.
    # Also tests the strategy between halfway of the start and end date, using the first half to train and second half to test.
    #
    # Output: 
    # {
    # 'stock1': stock1,
    # 'stock2': stock2,
    # 'correlation': correlation,
    # 'hedge_ratio': hedge_ratio,
    # 'spread': spread,
    # 'adf_p_value': adf_p_value,
    # 'total_return': total_return,
    # 'sharpe_ratio': sharpe_ratio,
    # 'max_drawdown': max_drawdown,
    # 'test_sharpe': test_sharpe
    # }
    # ------------------------------------

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    data = yf.download([stock1,stock2], start=start_date, end=end_date)['Close']
    print(f"\nMetrics and Graphs for pairs trading between {stock1} and {stock2}. \n")

    # Confirming Correlation for Pairs Testing
    correlation = data[stock1].corr(data[stock2])
    print(f"Correlation: {correlation:.5f} \n")



    # Calculting hedge ratio
    # Import Ordinary Least Squares method
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    # Add a constant to stock2 table and fit data with ordinary least squares regression using stock1 as y-axis and stock2 as x-axis
    X = add_constant(data[stock2])
    model = OLS(data[stock1], X).fit()
    # Finding the hedge ratio from the slope of OLS line and calculating the spread
    hedge_ratio = model.params[stock2]
    spread = data[stock1] - hedge_ratio * data[stock2]
    print(f"Hedge Ratio between stock1 and stock2: {hedge_ratio:.4f}")

    # Augmented Dickey-Fuller (ADF) to decide whether the spread is mean-reverting enough to be used in Pairs-Trading
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(spread.dropna())
    adf_p_value = adf_result[1]
    # Using adf_p_value as that gives us the p value
    print(f"ADF probability result: {adf_p_value:.4f}")
    # Check ADF test as 5% level
    if adf_p_value < 0.05:
        print(f"Reject Null Hypothesis, spread IS stationary. p: {adf_p_value} \n")
    else:
        print(f"Accept Null Hypothesis, spread is NOT stationary. p:{adf_p_value} \n")





    # Calculating z-score

    # Creating signals DataFrame
    signals = pd.DataFrame(index=data.index)
    # z-score is number of standard deviations away from the mean.
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    # Setting the signal to -1 when z > 1.5 std (Short stock1, Long stock2), and signal to 1 when z < -1.5 (Long stock1, Short stock2)
    # Signal is 0 if no z_score presnet (null data)
    signals['z_score'] = z_score
    signals['signal'] = 0
    signals.loc[z_score > z_score_boundary, 'signal'] = -1
    signals.loc[z_score < -z_score_boundary, 'signal'] = 1




    # Backtesting

    returns = data.pct_change()
    # Computing strategy returns
    # Shift 1 to prevent lookahead bias 
    signals['strategy_returns'] = (signals['signal'].shift(1) * (returns[stock1] - hedge_ratio * returns[stock2]))
    # Adding transaction costs of 0.1 % and taking away from returns
    transaction_loss = 0.001
    trades = signals['signal'].diff().abs() / 2   # Detects whether a trade occured
    signals['strategy_returns'] = signals['strategy_returns'] - (trades * transaction_loss)
    # Cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
    # Computing total return
    # Taking the last return ratio from the cumulative returns and taking away 1 to remove 100 percent and give growth percentage
    total_return = signals['cumulative_returns'].iloc[-1] - 1
    # 252 trading days in a year. Simplified Sharpe ratio with risk-free rate set to zero. (Usually subtract risk-free rate using 3 month Treasury rate)
    sharpe_ratio = ( (signals['strategy_returns'].mean() - 0 ) / signals['strategy_returns'].std()) * np.sqrt(252)
    max_drawdown = (signals['cumulative_returns'] / signals['cumulative_returns'].cummax() - 1).min()

    print(f"Total return: {total_return* 100 :.3f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}")
    print(f"Max drawdown: {max_drawdown*100 :.4f}% \n")




    # Test 
    # Setting the split time frame at halfway.
    date_half = pd.Timestamp(start_date) + ( pd.Timestamp(end_date) -  pd.Timestamp(start_date)   ) / 2
    test = signals[date_half:]
    # Calculating sharpe, and returns for the test data 
    test_returns = test['strategy_returns']
    test_cum_returns = (1 + test_returns).cumprod()
    test_total_returns = test_cum_returns.iloc[-1] - 1
    test_sharpe = ( (test['strategy_returns'].mean() - 0) / test['strategy_returns'].std()) * np.sqrt(252)
    print(f"Test total return: {test_total_returns * 100:.2f}%")
    print(f"Test Sharpe ratio: {test_sharpe:.4f}\n \n")





    # Figure for stock1 AND stock2 normalised closing prices (for correlation comparison)
    # Normalise the data
    plot_data = data / data.iloc[0] * 100

    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')

    plt.figure(figsize=(10,5))
    plt.plot(plot_data[stock1], label=(f'{stock1} Closing Price'), color='blue', )
    plt.plot(plot_data[stock2], label=(f'{stock2} Closing Price'), color='red')
    plt.title(f'Closing price between {stock1} and {stock2} ({start_str}-{end_str})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    #plt.savefig(f'Normalised_relation_{stock1}_and_{stock2}_({start_str}-{end_str}).png', dpi=150, bbox_inches='tight')
    plt.show()


    # Figure for z-score with signal lines
    plt.figure(figsize=(10,5))
    plt.plot(z_score, label="z-score", color='black')
    plt.axhline(y=z_score_boundary, label=(f"+{z_score_boundary}"), color='green', linestyle='--')
    plt.axhline(y=-z_score_boundary, label=(f"-{z_score_boundary}"), color='red', linestyle='--')
    plt.axhline(y=0, label='mean',color='blue', linestyle='--')
    plt.title(f'Z-score {stock1} and {stock2} ({start_str}-{end_str})')
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    #plt.savefig(f'z_score_graph_{stock1}_and_{stock2}_({start_str}-{end_str}).png', dpi=150, bbox_inches='tight')
    plt.show()


    # Figure for cumulative returns
    plt.figure(figsize=(10,5))
    plt.plot(signals['cumulative_returns'] - 1, label="cumulative returns", color="blue")
    plt.axhline(y=0, color="black")
    plt.title(f"Cumulative returns {stock1} and {stock2} ({start_str} and {end_str})")
    plt.xlabel("Date")
    plt.ylabel("Total return")
    plt.grid(True)
    plt.legend()
    #plt.savefig(f'Cumulative_returns_{stock1}_and_{stock2}_({start_str}-{end_str}).png', dpi=150, bbox_inches='tight')
    plt.show()





    # Returning the new info in dictionary
    return {
        'stock1': stock1,
        'stock2': stock2,
        'correlation': correlation,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'adf_p_value': adf_p_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'test_sharpe': test_sharpe
        }



# For BLK and BEN
BLK_BEN_pairs = pair_trade_metrics('BEN','BLK','2020-01-01','2024-01-01', 1.5)

# For CFG and PNC
CFG_PNC_pairs = pair_trade_metrics('CFG','PNC','2020-01-01','2024-01-01', 1.5)