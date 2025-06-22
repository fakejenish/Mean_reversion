#Just the main code for data to chart converter - will be the crux for the main code (first being the algo)

import yfinance as yf
import matplotlib.pyplot as plt

# List of stock symbols to compare
symbols = ['AAPL', 'MSFT', 'GOOGL']
window = 20  # rolling window size

plt.figure(figsize=(14, 8))

for symbol in symbols:
    # Download historical daily data for last 3 months
    data = yf.download(symbol, period='3mo', interval='1d', auto_adjust=True)

    # Calculate rolling mean
    data['rolling_mean'] = data['Close'].rolling(window=window).mean()

    # Plot closing price
    plt.plot(data.index, data['Close'], label=f'{symbol} Close')

    # Plot rolling mean
    plt.plot(data.index, data['rolling_mean'], label=f'{symbol} {window}-day MA')

plt.title('Closing Prices and Rolling Means')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()