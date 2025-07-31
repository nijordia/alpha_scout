import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple

def calculate_returns(prices):
    """Calculate percentage return from price series"""
    return (prices[-1] - prices[0]) / prices[0] * 100

def calculate_drawdown(prices):
    """Calculate maximum drawdown percentage"""
    peak = prices[0]
    max_drawdown = 0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio from a list of returns"""
    excess_returns = [r - risk_free_rate for r in returns]
    if not excess_returns or np.std(excess_returns) == 0:
        return 0
    return sum(excess_returns) / (np.std(excess_returns) * (len(excess_returns) ** 0.5))

def calculate_win_rate(signals_df, holding_period=14, target_return=None):
    """
    Calculate win rate for signals, correctly handling buy/sell signals.
    
    Args:
        signals_df: DataFrame with signal dates and prices
        holding_period: Days to hold after a signal
        target_return: Optional target return to consider a win
    
    Returns:
        Win rate as a percentage (0-100)
    """
    wins = 0
    total = 0
    
    for idx, row in signals_df.iterrows():
        if idx + holding_period >= len(signals_df):
            continue
        
        # Skip 'hold' signals for win rate calculation
        if 'signal' in row and row['signal'] == 'hold':
            continue
            
        entry_price = row['close']
        exit_price = signals_df.iloc[idx + holding_period]['close']
        
        # Calculate return
        signal_return = (exit_price - entry_price) / entry_price * 100
        
        # Determine if win based on signal type (buy/sell)
        signal_type = row.get('signal', 'buy')  # Default to buy if not specified
        
        if signal_type == 'buy':
            # For buy signals, positive return is a win
            is_win = (target_return is not None and signal_return >= target_return) or \
                    (target_return is None and signal_return > 0)
        elif signal_type == 'sell':
            # For sell signals, negative return is a win (price dropped as expected)
            is_win = (target_return is not None and signal_return <= -target_return) or \
                    (target_return is None and signal_return < 0)
        else:
            # Hold or unknown signals don't count as wins or losses
            continue
        
        if is_win:
            wins += 1
            
        total += 1
    
    if total == 0:
        return 0
        
    return (wins / total) * 100

def calculate_average_return_per_signal(signals_df, holding_period=14):
    """
    Calculate average return per signal, with correct handling of buy/sell signals.
    
    Args:
        signals_df: DataFrame with signal dates and prices
        holding_period: Days to hold after a signal
    
    Returns:
        Average percentage return
    """
    returns = []
    
    for idx, row in signals_df.iterrows():
        if idx + holding_period >= len(signals_df):
            continue
        
        # Skip 'hold' signals for return calculation
        if 'signal' in row and row['signal'] == 'hold':
            continue
            
        entry_price = row['close']
        exit_price = signals_df.iloc[idx + holding_period]['close']
        
        # Calculate raw return
        price_change_pct = (exit_price - entry_price) / entry_price * 100
        
        # Adjust return interpretation based on signal type
        signal_type = row.get('signal', 'buy')  # Default to buy if not specified
        
        if signal_type == 'buy':
            # For buy signals, positive price change is positive return
            signal_return = price_change_pct
        elif signal_type == 'sell':
            # For sell signals, negative price change is positive return
            # We flip the sign to reflect this
            signal_return = -price_change_pct
        else:
            # Skip hold signals
            continue
            
        returns.append(signal_return)
    
    if not returns:
        return 0
        
    return sum(returns) / len(returns)


def calculate_outperformance(signals_df, benchmark_df, holding_period=14):
    """
    Calculate average outperformance vs benchmark.
    
    Args:
        signals_df: DataFrame with signal dates and prices
        benchmark_df: DataFrame with benchmark dates and prices
        holding_period: Days to hold after a signal
    
    Returns:
        Average outperformance percentage
    """
    outperformances = []
    
    for idx, row in signals_df.iterrows():
        if idx + holding_period >= len(signals_df):
            continue
            
        signal_entry_price = row['close']
        signal_exit_price = signals_df.iloc[idx + holding_period]['close']
        
        # Find corresponding benchmark dates
        signal_date = row['date']
        try:
            benchmark_entry = benchmark_df[benchmark_df['date'] >= signal_date].iloc[0]
            benchmark_exit = benchmark_df[benchmark_df['date'] >= benchmark_df.iloc[benchmark_df.index.get_loc(benchmark_entry.name) + holding_period]['date']].iloc[0]
            
            benchmark_entry_price = benchmark_entry['close']
            benchmark_exit_price = benchmark_exit['close']
            
            # Calculate returns
            signal_return = (signal_exit_price - signal_entry_price) / signal_entry_price * 100
            benchmark_return = (benchmark_exit_price - benchmark_entry_price) / benchmark_entry_price * 100
            
            outperformances.append(signal_return - benchmark_return)
        except (IndexError, KeyError):
            continue
    
    if not outperformances:
        return 0
        
    return sum(outperformances) / len(outperformances)

def calculate_win_vs_market(signals_df, benchmark_df, holding_period=14):
    """
    Calculate percentage of signals that outperform market.
    
    Args:
        signals_df: DataFrame with signal dates and prices
        benchmark_df: DataFrame with benchmark dates and prices
        holding_period: Days to hold after a signal
    
    Returns:
        Percentage of signals that outperform market (0-100)
    """
    wins = 0
    total = 0
    
    for idx, row in signals_df.iterrows():
        if idx + holding_period >= len(signals_df):
            continue
            
        signal_entry_price = row['close']
        signal_exit_price = signals_df.iloc[idx + holding_period]['close']
        
        # Find corresponding benchmark dates
        signal_date = row['date']
        try:
            benchmark_entry = benchmark_df[benchmark_df['date'] >= signal_date].iloc[0]
            benchmark_exit = benchmark_df[benchmark_df['date'] >= benchmark_df.iloc[benchmark_df.index.get_loc(benchmark_entry.name) + holding_period]['date']].iloc[0]
            
            benchmark_entry_price = benchmark_entry['close']
            benchmark_exit_price = benchmark_exit['close']
            
            # Calculate returns
            signal_return = (signal_exit_price - signal_entry_price) / signal_entry_price * 100
            benchmark_return = (benchmark_exit_price - benchmark_entry_price) / benchmark_entry_price * 100
            
            if signal_return > benchmark_return:
                wins += 1
                
            total += 1
        except (IndexError, KeyError):
            continue
    
    if total == 0:
        return 0
        
    return (wins / total) * 100

def performance_summary(prices, returns, signals_df=None, benchmark_df=None, holding_period=14):

    """
    Generate a comprehensive performance summary.
    
    Args:
        prices: List or array of prices
        returns: List or array of returns
        signals_df: Optional DataFrame with signal data
        benchmark_df: Optional DataFrame with benchmark data
        holding_period: Days to hold after a signal
    
    Returns:
        Dictionary with performance metrics
    """
    summary = {
        "total_return": calculate_returns(prices),
        "max_drawdown": calculate_drawdown(prices),
        "sharpe_ratio": calculate_sharpe_ratio(returns)
    }
    
    # Add signal-specific metrics if available
    if signals_df is not None:
        summary["win_rate"] = calculate_win_rate(signals_df, holding_period)
        summary["avg_return_per_signal"] = calculate_average_return_per_signal(signals_df, holding_period)
        
        if benchmark_df is not None:
            summary["outperformance"] = calculate_outperformance(signals_df, benchmark_df, holding_period)
            summary["win_vs_market"] = calculate_win_vs_market(signals_df, benchmark_df, holding_period)
    
    return summary



def extract_trades_from_signals(signals_df: pd.DataFrame) -> list:
    """
    Extract trade pairs (entry/exit) from a DataFrame of signals.
    Returns a list of dicts: {'entry_date', 'entry_price', 'exit_date', 'exit_price', 'signal_type'}
    """
    trades = []
    position = None
    entry_pos = None  # Use integer position, not index label

    for pos, row in enumerate(signals_df.itertuples(index=False)):
        signal = getattr(row, 'signal', None)
        if signal == 'buy' and position is None:
            position = 'long'
            entry_pos = pos
        elif signal == 'sell' and position == 'long':
            entry_row = signals_df.iloc[entry_pos]
            trades.append({
                'entry_date': entry_row['date'],
                'entry_price': entry_row['close'],
                'exit_date': getattr(row, 'date'),
                'exit_price': getattr(row, 'close'),
                'signal_type': 'long'
            })
            position = None
            entry_pos = None
    # Handle open position at end of data
    if position == 'long' and entry_pos is not None:
        entry_row = signals_df.iloc[entry_pos]
        exit_row = signals_df.iloc[-1]
        trades.append({
            'entry_date': entry_row['date'],
            'entry_price': entry_row['close'],
            'exit_date': exit_row['date'],
            'exit_price': exit_row['close'],
            'signal_type': 'long'
        })
    return trades


def calculate_win_rate_from_trades(trades: List[Dict[str, any]], target_return=None) -> float:
    """
    Calculate win rate from a list of trade dicts.
    """
    wins = 0
    total = 0
    for trade in trades:
        entry = trade['entry_price']
        exit = trade['exit_price']
        ret = (exit - entry) / entry * 100
        if trade['signal_type'] == 'long':
            is_win = (target_return is not None and ret >= target_return) or \
                     (target_return is None and ret > 0)
        else:
            continue  # Only long trades for now
        if is_win:
            wins += 1
        total += 1
    if total == 0:
        return 0
    return (wins / total) * 100

def calculate_average_return_from_trades(trades: List[Dict[str, any]]) -> float:
    """
    Calculate average return from a list of trade dicts.
    """
    returns = []
    for trade in trades:
        entry = trade['entry_price']
        exit = trade['exit_price']
        ret = (exit - entry) / entry * 100
        if trade['signal_type'] == 'long':
            returns.append(ret)
        else:
            continue  # Only long trades for now
    if not returns:
        return 0
    return sum(returns) / len(returns)

# Update your summary and reliability logic to use the new functions:
# Example usage:
# trades = extract_trades_from_signals(signals_df)
# win_rate = calculate_win_rate_from_trades(trades)
# avg_return = calculate_average_return_from_trades(trades)
