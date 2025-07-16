from typing import Dict, List, Any
import pandas as pd
from src.data.fetcher import fetch_stock_data
from src.market_signals.mean_reversion import MeanReversionSignal
from src.bot.user_preferences import UserPreferencesManager
import logging

class SignalService:
    """
    Service for generating signals for tracked stocks
    """
    def __init__(self, user_prefs=None):
        """
        Initialize the signal service
        
        Parameters:
        -----------
        user_prefs : UserPreferencesManager, optional
            User preferences manager instance
        """
        self.user_prefs = user_prefs or UserPreferencesManager()
        self.logger = logging.getLogger(__name__)

    def generate_signals_for_stock(self, stock: str, user_id: str = None) -> Dict[str, str]:
        """
        Generate signals for a specific stock, optionally using user parameters
        
        Parameters:
        -----------
        stock : str
            Stock symbol
        user_id : str, optional
            Telegram user ID to use for custom parameters
                
        Returns:
        --------
        dict
            Dictionary of signal types and their values
        """
        try:
            # Fetch stock data
            data = fetch_stock_data(stock)
            if data is None:
                self.logger.error(f"Could not fetch data for {stock}")
                return {}
            
            signals = {}
            
            # Generate mean reversion signal
            try:
                mean_rev = MeanReversionSignal(data)
                
                # Use user parameters if user_id is provided
                if user_id:
                    signals_data = mean_rev.detect_signals_for_user(user_id, self.user_prefs)
                else:
                    signals_data = mean_rev.detect_signals()
                    
                latest_signal = signals_data['signal'].iloc[-1] if not signals_data.empty else "hold"
                signals["mean_reversion"] = latest_signal
                
                # Add additional details for logging/debugging
                if not signals_data.empty:
                    latest_data = signals_data.iloc[-1]
                    latest_price = latest_data[mean_rev.price_col]
                    upper_band = latest_data['upper_band']
                    lower_band = latest_data['lower_band']
                    sma = latest_data['sma']
                    
                    self.logger.debug(f"{stock} mean reversion signal: {latest_signal} "
                                    f"(Price: {latest_price:.2f}, SMA: {sma:.2f}, "
                                    f"Lower: {lower_band:.2f}, Upper: {upper_band:.2f})")
            except Exception as e:
                self.logger.error(f"Error generating mean reversion signal for {stock}: {e}")
            
            # Add more signal types here as they become available
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {stock}: {e}")
            return {}

    def generate_signals_for_user(self, user_id: str) -> Dict[str, Dict[str, str]]:
        """
        Generate signals for all stocks tracked by a user
        
        Parameters:
        -----------
        user_id : str
            Telegram user ID
            
        Returns:
        --------
        dict
            Dictionary of stocks and their signals
        """
        preferences = self.user_prefs.get_user_preferences(user_id)
        tracked_stocks = preferences.get("tracked_stocks", [])
        signal_types = preferences.get("signal_types", ["mean_reversion"])
        
        if not tracked_stocks:
            return {}
        
        signals = {}
        
        for stock in tracked_stocks:
            # Pass user_id to use custom parameters
            stock_signals = self.generate_signals_for_stock(stock, user_id)
            
            # Filter signals by user preferences
            filtered_signals = {
                signal_type: value for signal_type, value in stock_signals.items()
                if signal_type in signal_types
            }
            
            if filtered_signals:
                signals[stock] = filtered_signals
        
        return signals


    
    def generate_signals_for_all_users(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Generate signals for all users
        
        Returns:
        --------
        dict
            Dictionary of user IDs, stocks, and their signals
        """
        all_users = self.user_prefs.get_all_users()
        results = {}
        
        for user_id in all_users:
            user_signals = self.generate_signals_for_user(user_id)
            if user_signals:
                results[user_id] = user_signals
        
        return results
    
    def filter_active_signals(self, signals: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Filter to only include stocks with active (buy/sell) signals
        
        Parameters:
        -----------
        signals : dict
            Dictionary of stocks and their signals
            
        Returns:
        --------
        dict
            Dictionary of stocks with active signals
        """
        active_signals = {}
        
        for stock, stock_signals in signals.items():
            active_stock_signals = {
                signal_type: value for signal_type, value in stock_signals.items()
                if value in ["buy", "sell"]
            }
            
            if active_stock_signals:
                active_signals[stock] = active_stock_signals
        
        return active_signals