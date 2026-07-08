# Pairs Trading Screener — Statistical Arbitrage on S&P 500 Financials

**Note** Pairs Trading Screener can be modified for any time-range and any group of stocks. Function for metrics and graphs of two stocks can be run for any stocks, any yfinance supported time frames, and any z-scores boundaries. Train/Test timeframes are split halfway for backtesting. CSV files initially sorted by Sharpe ratio and Test Sharpe ratio.

A pairs trading code in Python that screens all stocks in the S&P financial sector, identifies statistically correlated trading pairs using Augmented Dickey-Fuller (ADF) test, backtests the mean-reversion strategies, and ranks by performance per unit risk, returning CSV files of all valid companies and their backtests. The time frames can be manipulated as well as the companies tested (can screen whole S&P or just certain sectors over any yfinance period).

Also includes a function to not only calculate all the metrics mentioned above for two paired companies, but display graphs of correlation, z-score, and cumulative returns with matplotlib. This is tested on the two best pairs found by the screener between 2020-2024  -  **BEN/BLK** (Franklin Templeton / BlackRock) and **CFG/PNC** (Citizens Financial / PNC Financial). Both achieved sharpe ratios above 1.5, maximum drawdowns below 9%, and held strongly on unseen data.

---

## Files

| File | Description |
|------|-------------|
| `pair_screener.py` | Screens all S&P 500 financial sector pairs — correlation, ADF, backtest, ranking |
| `pairs_trading_function.py` | Reusable function: full analysis + 3 charts for any pair |
| `screener_results_sharpe_sort.csv` | All 150 stationary pairs ranked by in sample Sharpe |
| `screener_results_test_sharpe_sort.csv` | All 150 stationary pairs ranked by out of sample Sharpe |

---

## What is Pairs Trading?

Pairs trading is a market-neutral strategy that goes long on a undervalued asset and short on an overvalued asset at the same time. The two assets are highly correlated and this framework profits from the temporary price divergence that will go back to the historical verage.

For example: Pepsi and Cola are mean-reverting (correlated to the point where stocks are predicted to come back together) so when a slight divergence in their relation appears, you can short the outperformer and long the underperformer, profiting when they converge back.

Being a market-neutral strategy means you only need to predict the *relative* movement between two stocks and can ignore whether the stocks go up or down.

---

## The Three-Step Pipeline

### Step 1 — Correlation 
Every pair from the S&P 500 financial sector is tested for correlation (around 2,400 pairs) and only pairs with a correlation above 0.8 proceed. In our case, this reduced the pairs from 2,400 to 437.

High correlation doesn't confirm however that two stocks are stationary (mean-reverting). Their spread could increase in one direction indefinitely. 

### Step 2 — Stationarity Test (ADF)
The Augmented Dickey-Fuller (ADF) test confirms whether the spread between two stocks is stationary (whether in the long run they revert reliably to the mean). The test uses a null hypothesis that states the spread has a unit root (not stationary) and checks at the at the 95% percent confidence level whether the spread doesn't have a unit root. A p-value < 0.05 confirms that two stocks are stationary.

Implementing this into the code, the pairs in the financial sector dropped from 437 to 150 stationary pairs. These are the ones which are valid for pair trading. Once again this can be done for many different stocks.

So how do we see which pairs are the most reliable and lucrative?

### Step 3 — Backtest and Rank
Each stationary pair is backtested using:
- **Hedge ratio** computed via Ordinary Least Squares (OLS) regression — scales one stock's price to match the other
- **Z-score signals**: trades triggered at +-1.5 standard deviations from the mean spread (this can be altered to +-2 or any other desired z-score) (gives a spread value of when to long/short)
- **Transaction costs** of 0.1% per trade to simulate realistic execution
- **Lookahead bias prevention** via `.shift(1)` on signals — trades only execute using yesterday's signal, never today's closing price
- **Train/test split** : strategy validated on unseen data (2022–2024) after being developed on 2020–2022 (can be altered in the code. More recent time frames used here)

All 150 pairs are ranked by Sharpe ratio, and by test Sharpe ratio (sharpe ratio during the testing split). All results in both sectors are saved to 2 separate CSV files. The CSV files from the published code are also published.

---

## Top Results

| Pair | Correlation | ADF p-value | Total Return | Sharpe | Max Drawdown | Test Sharpe |
|------|-------------|-------------|--------------|--------|--------------|-------------|
| BEN / BLK | 0.905 | 0.002 | 163.9% | 1.864 | -8.1% | 1.450 |
| CFG / PNC | 0.974 | 0.015 | 97.3% | 1.544 | -7.9% | 2.419 |
| GS / MS | 0.977 | 0.019 | 487% | 1.621 | -27.4% | 1.892 |

**Why BEN/BLK works:** 
BlackRock and Franklin Templeton are both pure-play asset managers, meaning their revenues depend on assets under management and therefore equity markets. They experience the same macroeconomic forces making their relative performance very predictable and mean-reverting. BEN / BLK had the lowest ADF p-value (most mean-reverting financial stock) and also the largest Sharpe ratio, signifying they are the best option for return per unit risk. In fact the slightly lower correlation but very low ADF p-value suggest higher returns since they have more fluctuation but are the most reliable when it comes to being mean-reverting - this was proven with the high total return of 163 percent.

**Why CFG/PNC works:** 
Citizens Financial Group and PNC Financial Services are both large US regional banks where their revenue is driven once again by similar factors, such as US consumer economy and interest rate. These banks are more homogenous that investment banks and therefore have higher cointegration. What was particularily interesting was the very high **Test** Sharpe ratio of above 2. This suggest in more recent years, the pair trading method would have worked well for CFG/PNC, maybe because of the regional bank covergence after the regional banking crisis where both CFG and PNC recovered well together afterwards, making them more predictable.

**Why the drawdown matters:** GS/MS produced a similar Sharpe to BEN/BLK but had a -27% max drawdown compared to -8% respectively. A -27% drawdown could cause margin calls before recovery, making the investment return much worse. However, when weighing out factors, its also best to consider that GS/MS had a very high total return of 487%.

---

## Technical Decisions

**OLS for the hedge ratio:**
The hedge ratio allows one stock to be scaled onto the other so different in stock value can be ignored when using spread. OLS regression does this by minimising sum of squared errors between predicted and actual prices. We can see that `add_constant` is used to give the regression line a free intercept to make sure the line doesn't pass through the origin (financial stocks never trade at 0). 

**Why z-score at ±1.5 rather than ±2.0:**
Signal frequency DOES affect signal quality. A threshold of +-2.0 z-score produces less signals but cleaner ones, whereas +-1.5 performs more signals but risks including noise. +-1.5 was chosen after testing both since it gave better total returns without affecting drawdown and Sharpe ratio as much. In some cases. +-2.0 will outperform +-1.5 but this is to be tested for different stocks.

**`.shift(1)`:**
The shift(1) simply shifts the signals one day before so that the trades are made the day before the results not on the day (this wouldn't be possible in present moment as you can't predict the future. Here we know the future). It is preventing **lookahead bias**.

**Train and Testing (Out of sample):**
If we optimise on the same data we used to train, then performance is faked since we know the outcomes. The train/test split, splits the dates halfway and uses the test  time period to test the strategy based off the training time period, giving a more reliable performance of the strategy.

---

## Limitations

- Fixed hedge ratio: The OLS hedge ratio is calculated once over the full period. In reality the relation between the stocks shifts over time (e.g COVID pandemic). A rolling hedge ratio is an upgrade.

- Single regime: the backtest covers 2020–2024, which includes COVID volatility and post-COVID normalisation. Performance may differ in other market regimes.
- Concentration risk: Many of the top stocks include repeated companies (e.g BLK, BEN). IF investing in all, there is a risk to exposure in these repeated companies.
- No short-selling constraints: Borrow costs and availability of shares to short add further friction which aren't included here.

## Setup

```bash
pip install yfinance pandas numpy matplotlib statsmodels
```

Run the screener:
```bash
python pair_screener.py
```

Analyse a specific pair:
```python
from pairs_trading_function import pair_trade_metrics
result = pair_trade_metrics('BEN', 'BLK', '2020-01-01', '2024-01-01', 1.5)
```

---

## Potential Extensions

- Rolling hedge ratio using a 60-day OLS window for regime adaptability
- Walk-forward validation across multiple time windows rather than a single train/test split
- Expand universe beyond financials to cross-sector pairs
- Chow test for structural break detection — identifying when a cointegrating relationship permanently breaks down
- Portfolio-level optimisation: selecting the optimal subset of pairs to maximise Sharpe while controlling for concentration risk
