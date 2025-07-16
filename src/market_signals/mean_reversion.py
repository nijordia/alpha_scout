import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

class MeanReversionSignal:
    def __init__(self, data, config=None, window=None, threshold=None, price_col='close'):
        """
        Initialize the mean reversion signal detector.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price data with date column
        config : dict, optional
            Configuration dictionary with 'mean_reversion_window' and 
            'mean_reversion_threshold' values (default: None)
        window : int, optional
            Window size for calculating the moving average
            (overrides config if provided)
        threshold : float, optional
            Number of standard deviations from the mean to generate signals
            (overrides config if provided)
        price_col : str
            Column name in the DataFrame to use for price data (default: 'close')
        """
        self.data = data
        self.price_col = price_col
        self.signals = None
        
        # Load configuration from YAML if not provided
        if config is None:
            config = self._load_config()
        
        # Set window and threshold with priority: explicit parameters > config > defaults
        self.window = window if window is not None else config.get('mean_reversion_window', 50)
        self.threshold = threshold if threshold is not None else config.get('mean_reversion_threshold', 1.5)
    
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
    
    def calculate_mean_and_bands(self):
        """Calculate moving average and Bollinger Bands."""
        # Make a copy to avoid modifying the original DataFrame
        self.processed_data = self.data.copy()
        
        # Calculate the moving average
        self.processed_data['sma'] = self.processed_data[self.price_col].rolling(window=self.window).mean()
        
        # Calculate the standard deviation
        self.processed_data['std'] = self.processed_data[self.price_col].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        self.processed_data['upper_band'] = self.processed_data['sma'] + (self.threshold * self.processed_data['std'])
        self.processed_data['lower_band'] = self.processed_data['sma'] - (self.threshold * self.processed_data['std'])
        
        # Drop NaN values that occur at the beginning due to the rolling window
        self.processed_data.dropna(inplace=True)
        
        return self.processed_data

    def detect_signals(self):
        """
        Detect mean reversion signals based on price position relative to bands.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the original data plus signal column
        """
        if not hasattr(self, 'processed_data'):
            self.calculate_mean_and_bands()
        
        # Initialize signals column
        self.processed_data['signal'] = 'hold'
        
        # Generate buy signals when price is below lower band
        self.processed_data.loc[self.processed_data[self.price_col] < self.processed_data['lower_band'], 'signal'] = 'buy'
        
        # Generate sell signals when price is above upper band
        self.processed_data.loc[self.processed_data[self.price_col] > self.processed_data['upper_band'], 'signal'] = 'sell'
        
        return self.processed_data
    
    def plot_signals(self, figsize=(12, 6), save_path=None):
        """
        Plot the price data with moving average, bands, and signals.
        
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
        if not hasattr(self, 'processed_data'):
            self.detect_signals()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price and bands
        ax.plot(self.processed_data['date'], self.processed_data[self.price_col], label='Price', color='blue')
        ax.plot(self.processed_data['date'], self.processed_data['sma'], label='SMA', color='black', alpha=0.7)
        ax.plot(self.processed_data['date'], self.processed_data['upper_band'], 
                label=f'Upper Band ({self.threshold} σ)', color='red', linestyle='--', alpha=0.7)
        ax.plot(self.processed_data['date'], self.processed_data['lower_band'], 
                label=f'Lower Band ({self.threshold} σ)', color='green', linestyle='--', alpha=0.7)
        
        # Add buy signals
        buy_signals = self.processed_data[self.processed_data['signal'] == 'buy']
        ax.scatter(buy_signals['date'], buy_signals[self.price_col], color='green', marker='^', 
                   s=100, label='Buy Signal')
        
        # Add sell signals
        sell_signals = self.processed_data[self.processed_data['signal'] == 'sell']
        ax.scatter(sell_signals['date'], sell_signals[self.price_col], color='red', marker='v', 
                   s=100, label='Sell Signal')
        
        # Add labels and legend
        ax.set_title(f'Mean Reversion Signals (Window: {self.window}, Threshold: {self.threshold})')
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