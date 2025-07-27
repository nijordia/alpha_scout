from typing import Dict, List, Any
import pandas as pd
from src.data.fetcher import fetch_stock_data
from src.market_signals.mean_reversion import MeanReversionSignal
from src.market_signals.momentum import MACrossoverSignal, VolatilityBreakoutSignal
from src.backtesting.signal_reliability import SignalReliabilityService
from src.bot.user_preferences import UserPreferencesManager
import logging
from datetime import datetime, timedelta

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
        self.user_prefs = user_prefs
        self.reliability_service = SignalReliabilityService()
        self.logger = logging.getLogger(__name__)

    def generate_signals_for_stock(self, stock: str, user_id: str = None) -> Dict[str, str]:
        """
        Generate signals for a specific stock, optionally using user parameters
        """
        try:
            data = fetch_stock_data(stock)
            if data is None:
                self.logger.error(f"Could not fetch data for {stock}")
                return {}
            signals = {}
            if user_id:
                preferences = self.user_prefs.get_user_preferences(user_id)
                signal_types = preferences.get("signal_types", ["mean_reversion"])
            else:
                signal_types = ["mean_reversion", "ma_crossover", "volatility_breakout"]

            if "mean_reversion" in signal_types:
                try:
                    mean_rev = MeanReversionSignal(data)
                    if user_id:
                        signals_data = mean_rev.detect_signals_for_user(user_id, self.user_prefs)
                    else:
                        signals_data = mean_rev.detect_signals()
                    latest_signal = signals_data['signal'].iloc[-1] if not signals_data.empty else "hold"
                    signals["mean_reversion"] = latest_signal
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

            if "ma_crossover" in signal_types:
                try:
                    ma_crossover = MACrossoverSignal(data)
                    if user_id:
                        ma_params = self.user_prefs.get_signal_params(user_id, "ma_crossover")
                        if ma_params:
                            short_window = ma_params.get("short_window")
                            long_window = ma_params.get("long_window")
                            if short_window and long_window:
                                ma_crossover = MACrossoverSignal(data, short_window=short_window, long_window=long_window)
                    signals_data = ma_crossover.detect_signals()
                    latest_signal = signals_data['signal'].iloc[-1] if not signals_data.empty else "hold"
                    signals["ma_crossover"] = latest_signal
                    if not signals_data.empty:
                        latest_data = signals_data.iloc[-1]
                        latest_price = latest_data[ma_crossover.price_col]
                        short_ma = latest_data['short_ma']
                        long_ma = latest_data['long_ma']
                        self.logger.debug(f"{stock} MA crossover signal: {latest_signal} "
                                         f"(Price: {latest_price:.2f}, Short MA: {short_ma:.2f}, "
                                         f"Long MA: {long_ma:.2f})")
                except Exception as e:
                    self.logger.error(f"Error generating MA crossover signal for {stock}: {e}")

            if "volatility_breakout" in signal_types:
                try:
                    vol_breakout = VolatilityBreakoutSignal(data)
                    if user_id:
                        vol_params = self.user_prefs.get_signal_params(user_id, "volatility_breakout")
                        if vol_params:
                            atr_window = vol_params.get("atr_window")
                            atr_multiplier = vol_params.get("atr_multiplier")
                            breakout_window = vol_params.get("breakout_window")
                            kwargs = {}
                            if atr_window: kwargs["atr_window"] = atr_window
                            if atr_multiplier: kwargs["atr_multiplier"] = atr_multiplier
                            if breakout_window: kwargs["breakout_window"] = breakout_window
                            if kwargs:
                                vol_breakout = VolatilityBreakoutSignal(data, **kwargs)
                    signals_data = vol_breakout.detect_signals()
                    latest_signal = signals_data['signal'].iloc[-1] if not signals_data.empty else "hold"
                    signals["volatility_breakout"] = latest_signal
                    if not signals_data.empty:
                        latest_data = signals_data.iloc[-1]
                        latest_price = latest_data[vol_breakout.price_col]
                        atr = latest_data['atr']
                        upper = latest_data['upper_threshold']
                        lower = latest_data['lower_threshold']
                        self.logger.debug(f"{stock} Volatility breakout signal: {latest_signal} "
                                         f"(Price: {latest_price:.2f}, ATR: {atr:.2f}, "
                                         f"Upper: {upper:.2f}, Lower: {lower:.2f})")
                except Exception as e:
                    self.logger.error(f"Error generating volatility breakout signal for {stock}: {e}")

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {stock}: {e}")
            return {}

    def get_signal_metrics_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        preferences = self.user_prefs.get_user_preferences(user_id)
        tracked_stocks = preferences.get("tracked_stocks", [])
        signal_types = preferences.get("signal_types", ["mean_reversion"])
        experiment_period = preferences.get("experiment_period", 365)
        signal_params = preferences.get("signal_params", {})

        results = []

        for stock in tracked_stocks:
            stock_result = {"stock": stock, "signals": []}
            max_window = experiment_period
            if "ma_crossover" in signal_types:
                ma_params = signal_params.get("ma_crossover", {})
                long_window = ma_params.get("long_window", 50)
                max_window = max(max_window, long_window)
            if "volatility_breakout" in signal_types:
                vb_params = signal_params.get("volatility_breakout", {})
                breakout_window = vb_params.get("breakout_window", 20)
                atr_window = vb_params.get("atr_window", 14)
                max_window = max(max_window, breakout_window, atr_window)
            if "mean_reversion" in signal_types:
                mr_params = signal_params.get("mean_reversion", {})
                mr_window = mr_params.get("window", 50)
                max_window = max(max_window, mr_window)

            data = fetch_stock_data(
                stock,
                start_date=(datetime.now() - timedelta(days=experiment_period + max_window)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            if data is None or len(data) == 0:
                stock_result["signals"].append({"type": "error", "text": "Could not fetch data"})
                results.append(stock_result)
                continue
            if len(data) > experiment_period:
                data = data.iloc[-experiment_period:]

            for signal_type in signal_types:
                try:
                    if signal_type == "mean_reversion":
                        mr_params = signal_params.get("mean_reversion", {})
                        window = mr_params.get("window", 50)
                        threshold = mr_params.get("threshold", 1.5)
                        mean_rev = MeanReversionSignal(data, window=window, threshold=threshold)
                        signal_info = mean_rev.get_latest_signal_formatted()
                        reliability = self.reliability_service.get_signal_reliability(
                            ticker=stock,
                            strategy_type='mean_reversion',
                            strategy_params={'window': window, 'threshold': threshold},
                            custom_start_date=(datetime.now() - timedelta(days=experiment_period + window)).strftime('%Y-%m-%d'),
                            custom_end_date=datetime.now().strftime('%Y-%m-%d')
                        )
                    elif signal_type == "ma_crossover":
                        ma_params = signal_params.get("ma_crossover", {})
                        short_window = ma_params.get("short_window", 20)
                        long_window = ma_params.get("long_window", 50)
                        ma_signal = MACrossoverSignal(data, short_window=short_window, long_window=long_window)
                        signal_info = ma_signal.get_latest_signal_formatted()
                        reliability = self.reliability_service.get_signal_reliability(
                            ticker=stock,
                            strategy_type='ma_crossover',
                            strategy_params={'short_window': short_window, 'long_window': long_window},
                            custom_start_date=(datetime.now() - timedelta(days=experiment_period + long_window)).strftime('%Y-%m-%d'),
                            custom_end_date=datetime.now().strftime('%Y-%m-%d')
                        )
                    elif signal_type == "volatility_breakout":
                        vb_params = signal_params.get("volatility_breakout", {})
                        atr_window = vb_params.get("atr_window", 14)
                        atr_multiplier = vb_params.get("atr_multiplier", 1.5)
                        breakout_window = vb_params.get("breakout_window", 20)
                        vb_signal = VolatilityBreakoutSignal(
                            data,
                            atr_window=atr_window,
                            atr_multiplier=atr_multiplier,
                            breakout_window=breakout_window
                        )
                        signal_info = vb_signal.get_latest_signal_formatted()
                        reliability = self.reliability_service.get_signal_reliability(
                            ticker=stock,
                            strategy_type='volatility_breakout',
                            strategy_params={
                                'atr_window': atr_window,
                                'atr_multiplier': atr_multiplier,
                                'breakout_window': breakout_window
                            },
                            custom_start_date=(datetime.now() - timedelta(days=experiment_period + max(breakout_window, atr_window))).strftime('%Y-%m-%d'),
                            custom_end_date=datetime.now().strftime('%Y-%m-%d')
                        )
                    else:
                        continue

                    avg_return = reliability.get('avg_return', 0)
                    bh_return = reliability.get('buy_hold_return', 0)
                    vs_bh = avg_return - bh_return if bh_return is not None else 0
                    reliability_text = (
                        f" | Win: {reliability.get('win_rate', 0)}% | Avg: {'+' if avg_return > 0 else ''}{avg_return}% | vs BH: {'+' if vs_bh > 0 else ''}{vs_bh:.2f}%"
                    )
                    stock_result["signals"].append({
                        "type": signal_type,
                        "text": signal_info['formatted_text'] + reliability_text,
                        "raw_signal": signal_info['signal'],
                        "metrics": reliability
                    })
                except Exception as e:
                    stock_result["signals"].append({"type": signal_type, "text": f"Error: {str(e)}"})
            results.append(stock_result)
        return results

    def generate_signals_for_all_users(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        all_users = self.user_prefs.get_all_users()
        results = {}
        for user_id in all_users:
            user_signals = self.generate_signals_for_user(user_id)
            if user_signals:
                results[user_id] = user_signals
        return results

    def filter_active_signals(self, signals: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        active_signals = {}
        for stock, stock_signals in signals.items():
            active_stock_signals = {
                signal_type: value for signal_type, value in stock_signals.items()
                if value in ["buy", "sell"]
            }
            if active_stock_signals:
                active_signals[stock] = active_stock_signals
        return active_signals