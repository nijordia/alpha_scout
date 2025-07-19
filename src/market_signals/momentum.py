
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

class MomentumSignal:
    """
    Base class for momentum-based trading signals
    """
    def __init__(self, data, config=None, price_col='close'):
        """
        Initialize the momentum signal detector.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with date column
        config : dict, optional
            Configuration dictionary
        price_col : str
            Column name in the DataFrame to use for price data (default: 'close')
        """
        self.data = data
        self.price_col = price_col
        self.signals = None
        
        # Load configuration if not provided
        if config is None:
            config = self._load_config()
        
        self.config = config
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config', 
            'config.yml'
        )
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_path}: {e}")
            return {}
    
    def detect_signals(self):
        """
        Base method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def plot_signals(self, figsize=(12, 6), save_path=None):
        """
        Plot the signals.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save the plot (default: None, don't save)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        raise NotImplementedError("Subclasses must implement this method")


class MACrossoverSignal(MomentumSignal):
    """
    Moving Average Crossover signal strategy
    Generates signals based on the crossover of short and long-term moving averages
    """
    def __init__(self, data, config=None, short_window=None, long_window=None, price_col='close'):
        """
        Initialize the MA crossover signal detector.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with date column
        config : dict, optional
            Configuration dictionary
        short_window : int, optional
            Window size for short-term moving average 
            (overrides config if provided)
        long_window : int, optional
            Window size for long-term moving average
            (overrides config if provided)
        price_col : str
            Column name in the DataFrame to use for price data (default: 'close')
        """
        super().__init__(data, config, price_col)
        
        # Set window parameters with priority: explicit parameters > config > defaults
        self.short_window = short_window if short_window is not None else config.get('momentum_short_window', 20)
        self.long_window = long_window if long_window is not None else config.get('momentum_long_window', 50)

    def calculate_indicators(self):
        """Calculate moving averages for the crossover strategy."""
        # Make a copy to avoid modifying the original DataFrame
        self.processed_data = self.data.copy()
        
        # Calculate the short-term moving average
        self.processed_data['short_ma'] = self.processed_data[self.price_col].rolling(
            window=self.short_window
        ).mean()
        
        # Calculate the long-term moving average
        self.processed_data['long_ma'] = self.processed_data[self.price_col].rolling(
            window=self.long_window
        ).mean()
        
        # Drop NaN values that occur at the beginning due to the rolling window
        self.processed_data.dropna(inplace=True)
        
        return self.processed_data

    def detect_signals(self):
        """
        Detect momentum signals based on moving average crossover.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the original data plus signal column
        """
        if not hasattr(self, 'processed_data'):
            self.calculate_indicators()
        
        # Initialize signal column
        self.processed_data['signal'] = 'hold'
        
        # Previous state of short MA compared to long MA
        prev_short_above_long = self.processed_data['short_ma'].iloc[0] > self.processed_data['long_ma'].iloc[0]
        
        # Loop through data points to detect crossovers
        for i in range(1, len(self.processed_data)):
            curr_short = self.processed_data['short_ma'].iloc[i]
            curr_long = self.processed_data['long_ma'].iloc[i]
            
            # Current state
            curr_short_above_long = curr_short > curr_long
            
            # Detect crossover
            if curr_short_above_long and not prev_short_above_long:
                # Bullish crossover (short MA crosses above long MA)
                self.processed_data.loc[self.processed_data.index[i], 'signal'] = 'buy'
            elif not curr_short_above_long and prev_short_above_long:
                # Bearish crossover (short MA crosses below long MA)
                self.processed_data.loc[self.processed_data.index[i], 'signal'] = 'sell'
            
            # Update previous state
            prev_short_above_long = curr_short_above_long
        
        return self.processed_data
    

    def plot_signals(self, figsize=(12, 6), save_path=None):
        """
        Plot the price data with moving averages and signals.
        """
        if not hasattr(self, 'processed_data'):
            self.detect_signals()
        
        # Make a copy for plotting to avoid modifying the original data
        plot_data = self.processed_data.copy()
        
        # Ensure dates are timezone-naive for plotting
        if 'date' in plot_data.columns and hasattr(plot_data['date'].iloc[0], 'tz') and plot_data['date'].iloc[0].tz is not None:
            plot_data['date'] = plot_data['date'].dt.tz_localize(None)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price and moving averages
        ax.plot(plot_data['date'], plot_data[self.price_col], 
                label='Price', color='blue')
        ax.plot(plot_data['date'], plot_data['short_ma'], 
                label=f'{self.short_window}-day MA', color='orange')
        ax.plot(plot_data['date'], plot_data['long_ma'], 
                label=f'{self.long_window}-day MA', color='purple')
        
        # Add buy signals
        buy_signals = plot_data[plot_data['signal'] == 'buy']
        ax.scatter(buy_signals['date'], buy_signals[self.price_col], color='green', 
                marker='^', s=100, label='Buy Signal')
        
        # Add sell signals
        sell_signals = plot_data[plot_data['signal'] == 'sell']
        ax.scatter(sell_signals['date'], sell_signals[self.price_col], color='red', 
                marker='v', s=100, label='Sell Signal')
        
        # Add labels and legend
        ax.set_title(f'Moving Average Crossover Signals ({self.short_window}/{self.long_window})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    def get_latest_signal_formatted(self):
        """
        Get the latest signal with formatting information.
        
        Returns:
        --------
        dict
            Dictionary containing signal information
        """
        # Make sure signals are calculated
        if not hasattr(self, 'processed_data') or self.processed_data is None:
            self.detect_signals()
        
        if self.processed_data.empty:
            return {
                'signal': 'hold',
                'emoji': '⚪',
                'formatted_text': 'HOLD (No data)',
                'details': {}
            }
        
        # Get the latest data point and signal
        latest_data = self.processed_data.iloc[-1]
        latest_signal = latest_data['signal']
        
        # Determine emoji based on signal
        emoji = "🟢" if latest_signal == "buy" else "🔴" if latest_signal == "sell" else "⚪"
        
        # Format signal text
        formatted_text = f"{emoji} {latest_signal.upper()}"
        
        # Include additional details
        details = {
            'price': latest_data[self.price_col],
            'short_ma': latest_data['short_ma'],
            'long_ma': latest_data['long_ma'],
            'short_window': self.short_window,
            'long_window': self.long_window,
            'date': latest_data['date'] if 'date' in latest_data else None
        }
        
        return {
            'signal': latest_signal,
            'emoji': emoji,
            'formatted_text': formatted_text,
            'details': details
        }


class VolatilityBreakoutSignal(MomentumSignal):
    """
    Volatility Breakout strategy with ATR filter to avoid fakeouts
    """
    def __init__(self, data, config=None, atr_window=None, atr_multiplier=None, 
                 breakout_window=None, price_col='close'):
        """
        Initialize the volatility breakout signal detector.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with date column
        config : dict, optional
            Configuration dictionary
        atr_window : int, optional
            Window size for ATR calculation (overrides config if provided)
        atr_multiplier : float, optional
            Multiplier for ATR to determine significant breakouts (overrides config)
        breakout_window : int, optional
            Window size for breakout detection (overrides config if provided)
        price_col : str
            Column name in the DataFrame to use for price data (default: 'close')
        """
        super().__init__(data, config, price_col)
        
        # Set parameters with priority: explicit parameters > config > defaults
        self.atr_window = atr_window if atr_window is not None else config.get('volatility_atr_window', 14)
        self.atr_multiplier = atr_multiplier if atr_multiplier is not None else config.get('volatility_atr_multiplier', 1.5)
        self.breakout_window = breakout_window if breakout_window is not None else config.get('volatility_breakout_window', 20)

    def calculate_true_range(self, high, low, prev_close):
        """Calculate true range for ATR."""
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
    def calculate_atr(self, data):
        """Calculate Average True Range."""
        tr_list = []
        
        # Calculate first true range
        tr_list.append(data['high'].iloc[0] - data['low'].iloc[0])
        
        # Calculate subsequent true ranges
        for i in range(1, len(data)):
            tr = self.calculate_true_range(
                data['high'].iloc[i],
                data['low'].iloc[i],
                data[self.price_col].iloc[i-1]
            )
            tr_list.append(tr)
        
        # Add true range to dataframe
        data['tr'] = tr_list
        
        # Calculate ATR using simple moving average of true range
        data['atr'] = data['tr'].rolling(window=self.atr_window).mean()
        
        return data

    def calculate_indicators(self):
        """Calculate indicators for volatility breakout strategy."""
        # Make a copy to avoid modifying the original DataFrame
        self.processed_data = self.data.copy()
        
        # Ensure we have high and low data
        if 'high' not in self.processed_data.columns or 'low' not in self.processed_data.columns:
            raise ValueError("Data must contain 'high' and 'low' columns for volatility strategy")
        
        # Calculate ATR
        self.processed_data = self.calculate_atr(self.processed_data)
        
        # Calculate rolling high and low for breakout detection
        self.processed_data['rolling_high'] = self.processed_data['high'].rolling(
            window=self.breakout_window
        ).max().shift(1)
        
        self.processed_data['rolling_low'] = self.processed_data['low'].rolling(
            window=self.breakout_window
        ).min().shift(1)
        
        # Calculate breakout thresholds with ATR filter
        self.processed_data['upper_threshold'] = self.processed_data['rolling_high'] + (
            self.processed_data['atr'] * self.atr_multiplier
        )
        
        self.processed_data['lower_threshold'] = self.processed_data['rolling_low'] - (
            self.processed_data['atr'] * self.atr_multiplier
        )
        
        # Drop NaN values that occur at the beginning due to the rolling windows
        self.processed_data.dropna(inplace=True)
        
        return self.processed_data

    def detect_signals(self):
        """
        Detect signals based on volatility breakout with ATR filter.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the original data plus signal column
        """
        if not hasattr(self, 'processed_data'):
            self.calculate_indicators()
        
        # Initialize signal column
        self.processed_data['signal'] = 'hold'
        
        # Detect breakouts with volatility filter
        for i in range(1, len(self.processed_data)):
            curr_high = self.processed_data['high'].iloc[i]
            curr_low = self.processed_data['low'].iloc[i]
            upper_threshold = self.processed_data['upper_threshold'].iloc[i]
            lower_threshold = self.processed_data['lower_threshold'].iloc[i]
            
            # Bullish breakout: price breaks above the upper threshold
            if curr_high > upper_threshold:
                self.processed_data.loc[self.processed_data.index[i], 'signal'] = 'buy'
            
            # Bearish breakout: price breaks below the lower threshold
            elif curr_low < lower_threshold:
                self.processed_data.loc[self.processed_data.index[i], 'signal'] = 'sell'
        
        return self.processed_data
    


    def plot_signals(self, figsize=(12, 8), save_path=None):
        """
        Plot the volatility breakout signals.
        """
        if not hasattr(self, 'processed_data'):
            self.detect_signals()
        
        # Make a copy for plotting to avoid modifying the original data
        plot_data = self.processed_data.copy()
        
        # Ensure dates are timezone-naive for plotting
        if 'date' in plot_data.columns and hasattr(plot_data['date'].iloc[0], 'tz') and plot_data['date'].iloc[0].tz is not None:
            plot_data['date'] = plot_data['date'].dt.tz_localize(None)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and thresholds in top panel
        ax1.plot(plot_data['date'], plot_data[self.price_col], 
                label='Close Price', color='blue')
        ax1.plot(plot_data['date'], plot_data['upper_threshold'], 
                label='Upper Threshold', color='red', linestyle='--', alpha=0.7)
        ax1.plot(plot_data['date'], plot_data['lower_threshold'], 
                label='Lower Threshold', color='green', linestyle='--', alpha=0.7)
        
        # Add buy signals
        buy_signals = plot_data[plot_data['signal'] == 'buy']
        ax1.scatter(buy_signals['date'], buy_signals[self.price_col], color='green', 
                marker='^', s=100, label='Buy Signal')
        
        # Add sell signals
        sell_signals = plot_data[plot_data['signal'] == 'sell']
        ax1.scatter(sell_signals['date'], sell_signals[self.price_col], color='red', 
                marker='v', s=100, label='Sell Signal')
        
        # Plot ATR in bottom panel
        ax2.plot(plot_data['date'], plot_data['atr'], 
                label=f'ATR ({self.atr_window})', color='purple')
        ax2.fill_between(plot_data['date'], 0, plot_data['atr'], 
                        alpha=0.3, color='purple')
        
        # Add labels and legends
        ax1.set_title(f'Volatility Breakout Signals (Window: {self.breakout_window}, ATR Multiplier: {self.atr_multiplier})')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('ATR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig


    def get_latest_signal_formatted(self):
        """
        Get the latest signal with formatting information.
        
        Returns:
        --------
        dict
            Dictionary containing signal information
        """
        # Make sure signals are calculated
        if not hasattr(self, 'processed_data') or self.processed_data is None:
            self.detect_signals()
        
        if self.processed_data.empty:
            return {
                'signal': 'hold',
                'emoji': '⚪',
                'formatted_text': 'HOLD (No data)',
                'details': {}
            }
        
        # Get the latest data point and signal
        latest_data = self.processed_data.iloc[-1]
        latest_signal = latest_data['signal']
        
        # Determine emoji based on signal
        emoji = "🟢" if latest_signal == "buy" else "🔴" if latest_signal == "sell" else "⚪"
        
        # Format signal text
        formatted_text = f"{emoji} {latest_signal.upper()}"
        
        # Include additional details
        details = {
            'price': latest_data[self.price_col],
            'atr': latest_data['atr'],
            'upper_threshold': latest_data['upper_threshold'],
            'lower_threshold': latest_data['lower_threshold'],
            'atr_window': self.atr_window,
            'atr_multiplier': self.atr_multiplier,
            'breakout_window': self.breakout_window,
            'date': latest_data['date'] if 'date' in latest_data else None
        }
        
        return {
            'signal': latest_signal,
            'emoji': emoji,
            'formatted_text': formatted_text,
            'details': details
        }