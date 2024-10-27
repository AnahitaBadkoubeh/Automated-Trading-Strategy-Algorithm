# Automated Trading Strategy: Bollinger Bands & Stochastic Oscillators
## Overview
This repository contains a Python-based automated trading strategy for a portfolio of five large-cap stocks. 

- The strategy combines Bollinger Bands and Stochastic Oscillators to identify optimal buy and sell signals by assessing price volatility and momentum shifts.
The trading approach is inspired by the research paper Using Bollinger Bands and Stochastic Oscillators as a Trading Strategy for Large Cap Stocks by Maxum, R. (2016), Texas Christian University (https://repository.tcu.edu/bitstream/handle/116099117/11341/Maxum__Ryan-Honors_Project.pdf?sequence=1&isAllowed=y).

 - The algorithm utilizes a dual-indicator system with the following rules:
Buy Signal: Triggered when the Stochastic Oscillator shifts from above 30 to below 30 while the stock price is below the lower Bollinger Band, signaling a potential rebound.
Sell Signal: Triggered when the Stochastic Oscillator moves from below 70 to above 70 while the stock price is above the upper Bollinger Band, indicating a likely price decline.

## Key Features
This strategy monitors daily stock data to identify market entry and exit points. By combining Bollinger Bands for price context and Stochastic Oscillators for momentum, it filters out false signals, aiming for higher accuracy in trades. It has been backtested over a 1-year period, from January 1, 2023, to the present date, with an initial investment of $100,000.
