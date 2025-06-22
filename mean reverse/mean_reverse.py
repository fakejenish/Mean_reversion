# Mean Reversion Trading Strategy for Stocks
# By: Jenish Malla, High School Quant Researcher
# Last Updated: 21-06-025

# This strategy looks for stocks that have pulled back from their moving averages
# and are likely to bounce back (mean revert) for short-term gains.

# ---------------------------
# SECTION 1: SETUP AND IMPORTS
# ---------------------------
# First, we'll import all the tools we need
import yfinance as yf           # For getting stock data
import matplotlib.pyplot as plt # For making charts
import numpy as np              # For math calculations
import pandas as pd             # For data organization
import warnings                # To hide annoying messages
from datetime import timedelta  # For working with dates
from matplotlib import dates as mdates  # For formatting dates on charts

# Turn off those red warning messages
warnings.filterwarnings("ignore")

# ---------------------------
# SECTION 2: STRATEGY SETTINGS
# ---------------------------
# Let's configure our strategy parameters
STOCK = "NVDA"            # We'll analyze NVIDIA (change to any stock)
DATA_PERIOD = "12mo"      # Get 1 year of historical data

# Moving Average Settings (Optimized):
SHORT_MA = 15    # Fast moving average (15 days)
MEDIUM_MA = 40   # Medium moving average (40 days)
LONG_MA = 120    # Slow trend MA (120 days)

# Trading Rules (Optimized):
ENTRY_THRESHOLD = 0.04   # Enter when price is 4% below Fast MA
EXIT_THRESHOLD = 0.03     # Exit when price is 3% above Fast MA
MIN_TREND = 0.01          # Require 1% slope on Medium MA vs Long MA

# Visual Settings:
MARKER_SIZE = 150         # Size of buy/sell markers on chart
RISK_FREE = 0.02          # Used for Sharpe ratio calculation

# Make our charts look cool with a dark theme
plt.style.use('dark_background')

# Color scheme for the charts
CHART_COLORS = {
    'price': '#1E88E5',       # Blue for the price line
    'fast_ma': '#FFC107',     # Yellow for fast MA
    'medium_ma': '#FF5722',   # Orange for medium MA
    'slow_ma': '#9C27B0',     # Purple for slow MA
    'buy': '#00E676',         # Green for buy signals
    'sell': '#FF5252',        # Red for sell signals
    'profit': '#4CAF50',      # Bright green for winning trades
    'loss': '#F44336'         # Bright red for losing trades
}

# ---------------------------
# SECTION 3: HELPER FUNCTIONS
# ---------------------------
# These functions help us analyze and visualize the data

def calculate_trend(data):
    """Measures how strong the current trend is"""
    # We look at how much the medium MA has changed over 5 days
    data['Trend_Strength'] = data['Medium_MA'].pct_change(periods=5)
    return data

def trade_performance(trades, stock_data):
    """Calculates how good our trades were"""
    if not trades:
        return None
    
    # Organize our trade records
    trade_log = pd.DataFrame(trades, columns=['Action', 'Date', 'Price'])
    
    # Find what the price was one day after each trade
    trade_log['Next_Day_Price'] = trade_log['Date'].apply(
        lambda x: stock_data.loc[stock_data.index > x, 'Close'].iloc[0] 
        if any(stock_data.index > x) else np.nan
    )
    trade_log.dropna(inplace=True)
    
    # Calculate profit/loss for each trade
    trade_log['Result'] = np.where(
        trade_log['Action'] == 'BUY',
        (trade_log['Next_Day_Price'] - trade_log['Price']) / trade_log['Price'],
        (trade_log['Price'] - trade_log['Next_Day_Price']) / trade_log['Price']
    )
    
    # Calculate performance stats
    wins = len(trade_log[trade_log['Result'] > 0])
    win_rate = wins / len(trade_log)
    avg_win = trade_log[trade_log['Result'] > 0]['Result'].mean()
    avg_loss = trade_log[trade_log['Result'] <= 0]['Result'].mean()
    
    # More advanced metrics
    profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    sharpe = (trade_log['Result'].mean() - RISK_FREE/252) / trade_log['Result'].std() * np.sqrt(252)
    
    return {
        'trades': trade_log,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_ratio,
        'sharpe_ratio': sharpe,
        'total_return': trade_log['Result'].sum()
    }

# ---------------------------
# SECTION 4: TRADING STRATEGY
# ---------------------------
# This is where the actual trading logic lives

def find_trade_signals(data):
    """Identifies when to buy and sell based on our rules"""
    signals = []
    holding_position = False
    current_trade_type = None
    
    for i in range(2, len(data)):
        today = data.iloc[i]
        yesterday = data.iloc[i-1]
        day_before = data.iloc[i-2]
        
        # Determine market trend
        uptrend = today['Fast_MA'] > today['Slow_MA']
        downtrend = today['Fast_MA'] < today['Slow_MA']
        strong_up = uptrend and (today['Trend_Strength'] > MIN_TREND)
        strong_down = downtrend and (today['Trend_Strength'] < -MIN_TREND)
        
        # ENTRY RULES
        if not holding_position:
            # BUY when price pulls back in uptrend
            if (strong_up or not strong_down):
                # Simple pullback entry
                if (yesterday['Close'] < yesterday['Fast_MA'] * (1 - ENTRY_THRESHOLD) 
                    and today['Close'] > yesterday['Close']):
                    signals.append(('BUY', today.name, today['Close']))
                    holding_position = True
                    current_trade_type = 'LONG'
                
                # MA crossover entry
                elif (day_before['Close'] < day_before['Fast_MA'] 
                      and yesterday['Close'] < yesterday['Fast_MA'] 
                      and today['Close'] > today['Fast_MA']):
                    signals.append(('BUY', today.name, today['Close']))
                    holding_position = True
                    current_trade_type = 'LONG'
            
            # SELL when price rallies in downtrend
            if (strong_down or not strong_up):
                # Overbought in downtrend
                if (yesterday['Close'] > yesterday['Fast_MA'] * (1 + ENTRY_THRESHOLD) 
                    and today['Close'] < yesterday['Close']):
                    signals.append(('SELL', today.name, today['Close']))
                    holding_position = True
                    current_trade_type = 'SHORT'
                
                # MA crossunder entry
                elif (day_before['Close'] > day_before['Fast_MA'] 
                      and yesterday['Close'] > yesterday['Fast_MA'] 
                      and today['Close'] < today['Fast_MA']):
                    signals.append(('SELL', today.name, today['Close']))
                    holding_position = True
                    current_trade_type = 'SHORT'
        
        # EXIT RULES
        elif holding_position:
            should_exit = None
            
            if current_trade_type == 'LONG':
                # Take profit at target
                if today['Close'] > today['Fast_MA'] * (1 + EXIT_THRESHOLD):
                    should_exit = 'SELL'
                # Stop if trend reverses
                elif today['Fast_MA'] < today['Medium_MA']:
                    should_exit = 'SELL'
            
            elif current_trade_type == 'SHORT':
                # Take profit at target
                if today['Close'] < today['Fast_MA'] * (1 - EXIT_THRESHOLD):
                    should_exit = 'BUY'
                # Stop if trend reverses
                elif today['Fast_MA'] > today['Medium_MA']:
                    should_exit = 'BUY'
            
            # Exit if price crosses medium MA
            if ((current_trade_type == 'LONG' and today['Close'] < today['Medium_MA']) or
                (current_trade_type == 'SHORT' and today['Close'] > today['Medium_MA'])):
                should_exit = 'BUY' if current_trade_type == 'SHORT' else 'SELL'
            
            if should_exit:
                signals.append((should_exit, today.name, today['Close']))
                holding_position = False
    
    return signals

# ---------------------------
# SECTION 5: MAIN PROGRAM
# ---------------------------
# This puts everything together and runs our analysis

def run_analysis():
    print(f"\n🚀 Analyzing {STOCK} Mean Reversion Strategy")
    print("-----------------------------------")
    
    try:
        # Step 1: Get the stock data
        print("📡 Downloading market data...")
        raw_data = yf.download(STOCK, period=DATA_PERIOD, auto_adjust=True)
        
        if len(raw_data) < LONG_MA:
            print(f"⚠️ Need at least {LONG_MA} trading days (only got {len(raw_data)})")
            return
        
        # Step 2: Calculate indicators
        print("🧮 Calculating indicators...")
        analysis_data = pd.DataFrame(index=raw_data.index)
        analysis_data['Close'] = raw_data['Close']
        analysis_data['Fast_MA'] = raw_data['Close'].rolling(SHORT_MA).mean()
        analysis_data['Medium_MA'] = raw_data['Close'].rolling(MEDIUM_MA).mean()
        analysis_data['Slow_MA'] = raw_data['Close'].rolling(LONG_MA).mean()
        analysis_data = calculate_trend(analysis_data)
        analysis_data.dropna(inplace=True)
        
        # Step 3: Find trading signals
        print("🔍 Identifying trade opportunities...")
        trade_signals = find_trade_signals(analysis_data)
        
        # Step 4: Evaluate performance
        results = trade_performance(trade_signals, analysis_data)
        
        # Step 5: Visualize everything
        print("📊 Creating performance chart...")
        plt.figure(figsize=(16, 9))
        chart = plt.gca()
        chart.set_facecolor('#121212')  # Dark background
        
        # Plot price and moving averages
        plt.plot(analysis_data.index, analysis_data['Close'], 
                label='Price', color=CHART_COLORS['price'], linewidth=2)
        plt.plot(analysis_data.index, analysis_data['Fast_MA'], '--', 
                label=f'{SHORT_MA}-Day MA', color=CHART_COLORS['fast_ma'])
        plt.plot(analysis_data.index, analysis_data['Medium_MA'], '--', 
                label=f'{MEDIUM_MA}-Day MA', color=CHART_COLORS['medium_ma'])
        plt.plot(analysis_data.index, analysis_data['Slow_MA'], ':', 
                label=f'{LONG_MA}-Day MA', color=CHART_COLORS['slow_ma'])
        
        # Mark buy/sell signals
        buys = [s for s in trade_signals if s[0] == 'BUY']
        sells = [s for s in trade_signals if s[0] == 'SELL']
        
        if buys:
            buy_dates, buy_prices = zip(*[(s[1], s[2]) for s in buys])
            plt.scatter(buy_dates, buy_prices, s=MARKER_SIZE, 
                       color=CHART_COLORS['buy'], edgecolors='white',
                       label='Buy Signal', zorder=5)
        
        if sells:
            sell_dates, sell_prices = zip(*[(s[1], s[2]) for s in sells])
            plt.scatter(sell_dates, sell_prices, s=MARKER_SIZE,
                       color=CHART_COLORS['sell'], edgecolors='white',
                       label='Sell Signal', zorder=5)
        
        # Show trade results if we have any
        if results and not results['trades'].empty:
            # Draw lines showing trade outcomes
            for _, trade in results['trades'].iterrows():
                trade_color = CHART_COLORS['profit'] if trade['Result'] > 0 else CHART_COLORS['loss']
                plt.plot(
                    [trade['Date'], trade['Date'] + timedelta(days=1)],
                    [trade['Price'], trade['Next_Day_Price']],
                    color=trade_color, alpha=0.6, linewidth=2
                )
                plt.scatter(
                    trade['Date'] + timedelta(days=1), trade['Next_Day_Price'],
                    color=trade_color, s=50, edgecolor='white', zorder=6
                )
            
            # Add performance summary
            stats = [
                f"Win Rate: {results['win_rate']:.1%}",
                f"Avg Win: {results['avg_win']:.2%}",
                f"Avg Loss: {results['avg_loss']:.2%}",
                f"Profit Factor: {results['profit_factor']:.2f}",
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}",
                f"Total Return: {results['total_return']:.1%}"
            ]
            
            chart.text(0.02, 0.98, "\n".join(stats), transform=chart.transAxes,
                      verticalalignment='top', fontfamily='monospace',
                      bbox=dict(facecolor='#121212', alpha=0.8, edgecolor='white'))
        
        # Final chart formatting
        plt.title(f"{STOCK} Mean Reversion Strategy Performance\n{SHORT_MA}/{MEDIUM_MA}/{LONG_MA} Moving Averages", 
                 pad=20, fontsize=14, color='white')
        plt.xlabel("Date", fontsize=12, color='white')
        plt.ylabel("Price ($)", fontsize=12, color='white')
        plt.grid(color='#424242', alpha=0.3)
        
        # Date formatting
        chart.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
        plt.xticks(rotation=45)
        
        # Smart legend that doesn't show duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys(),
                  facecolor='#121212', edgecolor='white',
                  loc='upper left', bbox_to_anchor=(1.01, 1))
        
        plt.tight_layout()
        plt.show()
        
        # Print final report
        print("\n" + "="*50)
        print(f"📊 {STOCK} Strategy Performance Report")
        print("="*50)
        if results:
            print(f"Total Trades: {len(results['trades'])}")
            print(f"Win Rate: {results['win_rate']:.1%}")
            print(f"Average Gain: {results['avg_win']:.2%}")
            print(f"Average Loss: {results['avg_loss']:.2%}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Total Strategy Return: {results['total_return']:.1%}")
        else:
            print("No trades were made during this period")
        print("="*50)
        
    except Exception as error:
        print(f"❌ Error: {str(error)}")

# Run the analysis when this file is executed
if __name__ == "__main__":
    run_analysis()