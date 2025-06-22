# Mean Reversion Strategy - Jenish Malla (High Schooler)

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from matplotlib import dates as mdates

# Configuration
STOCK = "GMS"
DATA_PERIOD = "12mo"
SHORT_MA, MEDIUM_MA, LONG_MA = 15, 40, 120
ENTRY_THRESHOLD, EXIT_THRESHOLD = 0.04, 0.03
MIN_TREND = 0.01
RISK_FREE = 0.02

plt.style.use('dark_background')
colors = {
    'price': '#1E88E5', 'fast_ma': '#FFC107', 
    'medium_ma': '#FF5722', 'slow_ma': '#9C27B0',
    'buy': '#00E676', 'sell': '#FF5252',
    'profit': '#4CAF50', 'loss': '#F44336'
}

def calculate_trend(data):
    data['Trend_Strength'] = data['Medium_MA'].pct_change(periods=5)
    return data

def evaluate_trades(trades, stock_data):
    if not trades: return None
    
    trade_log = pd.DataFrame(trades, columns=['Action', 'Date', 'Price'])
    trade_log['Next_Price'] = trade_log['Date'].apply(
        lambda x: stock_data.loc[stock_data.index > x, 'Close'].iloc[0] 
        if any(stock_data.index > x) else np.nan
    )
    trade_log.dropna(inplace=True)
    
    trade_log['Pct_Change'] = np.where(
        trade_log['Action'] == 'BUY',
        (trade_log['Next_Price'] - trade_log['Price']) / trade_log['Price'],
        (trade_log['Price'] - trade_log['Next_Price']) / trade_log['Price']
    )
    
    wins = len(trade_log[trade_log['Pct_Change'] > 0])
    win_rate = wins / len(trade_log)
    avg_win = trade_log[trade_log['Pct_Change'] > 0]['Pct_Change'].mean()
    avg_loss = trade_log[trade_log['Pct_Change'] <= 0]['Pct_Change'].mean()
    sharpe = (trade_log['Pct_Change'].mean() - RISK_FREE/252) / trade_log['Pct_Change'].std() * np.sqrt(252)
    
    return {
        'trades': trade_log,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'total_return': trade_log['Pct_Change'].sum()
    }

def generate_signals(data):
    signals = []
    in_trade = False
    
    for i in range(2, len(data)):
        today = data.iloc[i]
        yesterday = data.iloc[i-1]
        day_before = data.iloc[i-2]
        
        uptrend = today['Fast_MA'] > today['Slow_MA']
        downtrend = today['Fast_MA'] < today['Slow_MA']
        strong_up = uptrend and (today['Trend_Strength'] > MIN_TREND)
        strong_down = downtrend and (today['Trend_Strength'] < -MIN_TREND)
        
        if not in_trade:
            # Buy conditions
            if (strong_up or not strong_down):
                if (yesterday['Close'] < yesterday['Fast_MA'] * (1 - ENTRY_THRESHOLD) 
                    and today['Close'] > yesterday['Close']):
                    signals.append(('BUY', today.name, today['Close']))
                    in_trade = True
                
                elif (day_before['Close'] < day_before['Fast_MA'] 
                      and yesterday['Close'] < yesterday['Fast_MA'] 
                      and today['Close'] > today['Fast_MA']):
                    signals.append(('BUY', today.name, today['Close']))
                    in_trade = True
            
            # Sell conditions
            if (strong_down or not strong_up):
                if (yesterday['Close'] > yesterday['Fast_MA'] * (1 + ENTRY_THRESHOLD) 
                    and today['Close'] < yesterday['Close']):
                    signals.append(('SELL', today.name, today['Close']))
                    in_trade = True
                
                elif (day_before['Close'] > day_before['Fast_MA'] 
                      and yesterday['Close'] > yesterday['Fast_MA'] 
                      and today['Close'] < today['Fast_MA']):
                    signals.append(('SELL', today.name, today['Close']))
                    in_trade = True
        
        elif in_trade:
            if today['Close'] > today['Fast_MA'] * (1 + EXIT_THRESHOLD):
                signals.append(('SELL', today.name, today['Close']))
                in_trade = False
            elif today['Close'] < today['Fast_MA'] * (1 - EXIT_THRESHOLD):
                signals.append(('BUY', today.name, today['Close']))
                in_trade = False
    
    return signals

def run_strategy():
    print(f"\nAnalyzing {STOCK} Mean Reversion Strategy")
    
    try:
        data = yf.download(STOCK, period=DATA_PERIOD, auto_adjust=True)
        if len(data) < LONG_MA:
            print(f"Need at least {LONG_MA} trading days")
            return
        
        df = pd.DataFrame(index=data.index)
        df['Close'] = data['Close']
        df['Fast_MA'] = data['Close'].rolling(SHORT_MA).mean()
        df['Medium_MA'] = data['Close'].rolling(MEDIUM_MA).mean()
        df['Slow_MA'] = data['Close'].rolling(LONG_MA).mean()
        df = calculate_trend(df).dropna()
        
        signals = generate_signals(df)
        results = evaluate_trades(signals, df)
        
        plt.figure(figsize=(16, 9))
        ax = plt.gca()
        ax.set_facecolor('#121212')
        
        plt.plot(df.index, df['Close'], label='Price', color=colors['price'], linewidth=2)
        plt.plot(df.index, df['Fast_MA'], '--', label=f'{SHORT_MA}-Day MA', color=colors['fast_ma'])
        plt.plot(df.index, df['Medium_MA'], '--', label=f'{MEDIUM_MA}-Day MA', color=colors['medium_ma'])
        plt.plot(df.index, df['Slow_MA'], ':', label=f'{LONG_MA}-Day MA', color=colors['slow_ma'])
        
        buys = [s for s in signals if s[0] == 'BUY']
        sells = [s for s in signals if s[0] == 'SELL']
        
        if buys:
            buy_dates, buy_prices = zip(*[(s[1], s[2]) for s in buys])
            plt.scatter(buy_dates, buy_prices, s=150, color=colors['buy'], edgecolors='white', label='Buy')
        
        if sells:
            sell_dates, sell_prices = zip(*[(s[1], s[2]) for s in sells])
            plt.scatter(sell_dates, sell_prices, s=150, color=colors['sell'], edgecolors='white', label='Sell')
        
        if results and not results['trades'].empty:
            for _, trade in results['trades'].iterrows():
                color = colors['profit'] if trade['Pct_Change'] > 0 else colors['loss']
                plt.plot([trade['Date'], trade['Date'] + timedelta(days=1)],
                         [trade['Price'], trade['Next_Price']], color=color, alpha=0.6, linewidth=2)
            
            stats = [
                f"Win Rate: {results['win_rate']:.1%}",
                f"Avg Win: {results['avg_win']:.2%}",
                f"Avg Loss: {results['avg_loss']:.2%}",
                f"Sharpe: {results['sharpe']:.2f}",
                f"Total Return: {results['total_return']:.1%}"
            ]
            
            ax.text(0.02, 0.98, "\n".join(stats), transform=ax.transAxes,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(facecolor='#121212', alpha=0.8, edgecolor='white'))
        
        plt.title(f"{STOCK} Mean Reversion Strategy", pad=20)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(color='#424242', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
        plt.xticks(rotation=45)
        plt.legend(facecolor='#121212', edgecolor='white')
        plt.tight_layout()
        plt.show()
        
        if results:
            print(f"\nTrades: {len(results['trades'])}")
            print(f"Win Rate: {results['win_rate']:.1%}")
            print(f"Total Return: {results['total_return']:.1%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_strategy()