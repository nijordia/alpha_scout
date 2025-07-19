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
    Calculate win rate for signals.
    
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
            
        entry_price = row['close']
        exit_price = signals_df.iloc[idx + holding_period]['close']
        
        # Calculate return
        signal_return = (exit_price - entry_price) / entry_price * 100
        
        # Determine if win based on target or positive return
        if target_return is not None:
            if signal_return >= target_return:
                wins += 1
        elif signal_return > 0:
            wins += 1
            
        total += 1
    
    if total == 0:
        return 0
        
    return (wins / total) * 100

def calculate_average_return_per_signal(signals_df, holding_period=14):
    """
    Calculate average return per signal.
    
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
            
        entry_price = row['close']
        exit_price = signals_df.iloc[idx + holding_period]['close']
        
        # Calculate return
        signal_return = (exit_price - entry_price) / entry_price * 100
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