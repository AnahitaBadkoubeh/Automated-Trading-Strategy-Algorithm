import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from termcolor import colored as cl


plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

# Function for Simple Moving Average, Bollinger Bands, and Stochastic Oscillator
def sma(data, lookback):
    return data.rolling(lookback).mean()

def get_bb(data, lookback):
    std = data.rolling(lookback).std()
    upper_bb = sma(data, lookback) + std * 2
    lower_bb = sma(data, lookback) - std * 2
    middle_bb = sma(data, lookback)
    return upper_bb, lower_bb, middle_bb

def get_stoch_osc(high, low, close, k_lookback, d_lookback):
    lowest_low = low.rolling(k_lookback).min()
    highest_high = high.rolling(k_lookback).max()
    k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d_line = k_line.rolling(d_lookback).mean()
    return k_line, d_line

# Fetching historical data for multiple stocks
def get_historical_data(symbols, start_date):
    stock_data = {}
    for symbol in symbols:
        df = yf.download(symbol, start=start_date)
        stock_data[symbol] = df
    return stock_data

# Specify stocks and start date
symbols = ['AAPL', 'JPM', 'JNJ', 'XOM', 'PG']
start_date = '2023-01-01'
stocks = get_historical_data(symbols, start_date)

# Calculate Bollinger Bands and Stochastic Oscillator for each stock
for symbol in stocks:
    df = stocks[symbol]
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = get_bb(df['Close'], 20)
    df['%k'], df['%d'] = get_stoch_osc(df['High'], df['Low'], df['Close'], 14, 3)
    df = df.dropna()  
    stocks[symbol] = df 
# Plot 
def plot_stock_data(stock_data, symbol):
    plot_data = stock_data[stock_data.index >= '2023-01-01']

    # Plot Bollinger Bands
    plt.figure(figsize=(14, 7))
    plt.plot(plot_data['Close'], label='Close', linewidth=2.5)
    plt.plot(plot_data['upper_bb'], label='Upper BB 20', linestyle='--', linewidth=1, color='black')
    plt.plot(plot_data['middle_bb'], label='Middle BB 20', linestyle='--', linewidth=1.2, color='grey')
    plt.plot(plot_data['lower_bb'], label='Lower BB 20', linestyle='--', linewidth=1, color='black')
    plt.title(f'{symbol} Bollinger Bands')
    plt.legend(loc='upper left')
    plt.show()

    # Plot Stochastic Oscillator
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})
    ax1.plot(plot_data['Close'], label='Close', color='blue', linewidth=2.5)
    ax1.set_title(f'{symbol} Stock Prices')
    ax1.legend()

    ax2.plot(plot_data['%k'], label='%K', color='deepskyblue', linewidth=1.5)
    ax2.plot(plot_data['%d'], label='%D', color='orange', linewidth=1.5)
    ax2.axhline(70, color='black', linewidth=1, linestyle='--')
    ax2.axhline(30, color='black', linewidth=1, linestyle='--')
    ax2.set_title(f'{symbol} Stochastic Oscillator')
    ax2.legend(loc='upper right')
    plt.show()

for symbol in stocks:
    plot_stock_data(stocks[symbol], symbol)

# Define trade strategy
def bb_stoch_strategy(prices, k, d, upper_bb, lower_bb):
    buy_price = [np.nan]  
    sell_price = [np.nan]  
    bb_stoch_signal = [0] 
    signal = 0
    
    for i in range(1, len(prices)):
        if k[i-1] > 30 and d[i-1] > 30 and k[i] < 30 and d[i] < 30 and prices[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_stoch_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_stoch_signal.append(0)
        elif k[i-1] < 70 and d[i-1] < 70 and k[i] > 70 and d[i] > 70 and prices[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                bb_stoch_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_stoch_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_stoch_signal.append(0)

    # Ensure lists are the correct length
    if len(buy_price) < len(prices):
        buy_price.append(np.nan)  
        sell_price.append(np.nan)  
        bb_stoch_signal.append(0)  
    
    return buy_price, sell_price, bb_stoch_signal

# Apply the strategy to all stocks in the dictionary
for symbol in stocks:
    df = stocks[symbol]
    df['buy_price'], df['sell_price'], df['bb_stoch_signal'] = bb_stoch_strategy(df['Close'], df['%k'], df['%d'], 
                                                                                   df['upper_bb'], df['lower_bb'])
# Plot trading strategy
def plot_trading_signals(df, symbol):
    # Create subplot grid
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot closing price and Bollinger Bands
    ax1.plot(df['Close'], label='Close', color='blue', linewidth=2)
    ax1.plot(df['upper_bb'], label='Upper BB', linestyle='--', color='black')
    ax1.plot(df['middle_bb'], label='Middle BB', linestyle='--', color='grey')
    ax1.plot(df['lower_bb'], label='Lower BB', linestyle='--', color='black')
    ax1.scatter(df.index, df['buy_price'], label='Buy Signal', marker='^', color='green', alpha=1, s=100)
    ax1.scatter(df.index, df['sell_price'], label='Sell Signal', marker='v', color='red', alpha=1, s=100)
    ax1.set_title(f'{symbol} Stock Price and BB')
    ax1.legend()

    # Plot %K and %D from Stochastic Oscillator
    ax2.plot(df['%k'], label='%K', color='deepskyblue', linewidth=1.5)
    ax2.plot(df['%d'], label='%D', color='orange', linewidth=1.5)
    ax2.axhline(70, color='black', linewidth=1, linestyle='--')
    ax2.axhline(30, color='black', linewidth=1, linestyle='--')
    ax2.set_title(f'{symbol} Stochastic Oscillator')
    ax2.legend()

    plt.show()

# Apply the strategy and plot for all stocks
for symbol in stocks:
    df = stocks[symbol]
    df['buy_price'], df['sell_price'], df['bb_stoch_signal'] = bb_stoch_strategy(df['Close'], df['%k'], df['%d'], df['upper_bb'], df['lower_bb'])
    plot_trading_signals(df, symbol)
    
    
# Define position
""" 
 - Buy signal: 
   When PREV_ST_COM > 30 & CUR_ST_COMP < 30 & CL < LOWER_BB 
 - Sell signal: 
   When PREV_ST_COM > 70 & CUR_ST_COMP < 70 & CL < UPPER_BB 

  where,
  PRE_ST_COM = Previous Day Stochastic Oscillator components' readings
  CUR_ST_COM = Current Day Stochastic Oscillator components' readings
  CL = Last Closing Price
  LOWER_BB = Current Day Lower Band reading
  UPPER_BB = Current Day Upper Band reading
"""
def calculate_position(df):
    position = []
    for signal in df['bb_stoch_signal']:
        if signal == 1:
            position.append(1)  # Buy
        elif signal == -1:
            position.append(0)  # Sell
        else:
            position.append(None)

    # Fill forward the position status
    for i in range(1, len(position)):
        if position[i] is None:
            position[i] = position[i - 1]

    position[0] = 0  
    df['position'] = position
    return df

# Apply the strategy and plot for each stock
for symbol in stocks:
    df = stocks[symbol]
    df = calculate_position(df)
    plot_trading_signals(df, symbol)
    
# Define backtest
def backtest_strategy(stock_data, initial_investment):
    # Assuming 'Close' is a column in stock_data representing closing prices
    if 'Close' not in stock_data.columns:
        return None, None  

    # Calculate daily percentage returns
    daily_returns = np.diff(stock_data['Close']) / stock_data['Close'][:-1]
    daily_returns = np.append(daily_returns, 0)  

    # Calculate strategy returns
    strategy_returns = daily_returns * stock_data['position'].shift(1).fillna(0) 
    stock_data['strategy_returns'] = strategy_returns

    # Calculate total returns from the strategy
    number_of_stocks = floor(initial_investment / stock_data['Close'][0])  # stocks bought at start
    investment_returns = strategy_returns * number_of_stocks * stock_data['Close'][1:]  # multiply by stock price for dollar amount
    investment_returns = np.append(investment_returns, [0])  # Ensure the length matches

    total_investment_return = np.sum(investment_returns)
    profit_percentage = (total_investment_return / initial_investment) * 100

    return total_investment_return, profit_percentage


# Usage within the loop
initial_investment = 100000
results = {}

for symbol in stocks:
    if 'position' not in stocks[symbol].columns:
        print(f"No position data for {symbol}, skipping...")
        continue

    total_return, profit_pct = backtest_strategy(stocks[symbol], initial_investment)
    if total_return is not None:  # Check if the function returned data
        results[symbol] = {'Total Return': total_return, 'Profit Percentage': profit_pct}
        print(cl(f'Profit gained from the strategy by investing ${initial_investment} in {symbol}: ${total_return:.2f}', attrs=['bold']))
        print(cl(f'Profit percentage of the strategy in {symbol}: {profit_pct:.2f}%', attrs=['bold']))
    else:
        print(f"Skipped {symbol} due to data issues.")

# Print or analyze the results DataFrame
if results:
    results_df = pd.DataFrame(results).T
    print(results_df)
    
  