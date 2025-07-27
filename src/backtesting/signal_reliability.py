import logging
import pandas as pd
from typing import Dict, Any, Optional
import os
import json
import time
from datetime import timedelta

from src.backtesting.strategy_evaluator import StrategyEvaluator

logger = logging.getLogger(__name__)

# ...existing code...

class SignalReliabilityService:
    """
    Service to provide reliability metrics for trading signals.
    """
    def __init__(self, config=None):
        self.config = config or {}
        reliability_cfg = self.config.get('reliability', {})
        self.experiment_period = reliability_cfg.get('experiment_period', 365)
        self.preferred_period = reliability_cfg.get('preferred_period', 30)
        self.cache_expiry = reliability_cfg.get('cache_expiry', 24 * 60 * 60)
        self.evaluator = StrategyEvaluator(config)
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'cache', 
            'reliability'
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_signal_reliability(self, ticker: str, strategy_type: str, 
                            strategy_params: Dict[str, Any] = None,
                            custom_start_date: str = None,
                            custom_end_date: str = None) -> Dict[str, Any]:
        cached_results = self._get_cached_results(ticker, strategy_type, strategy_params)
        if cached_results:
            logger.info(f"Using cached reliability metrics for {ticker} {strategy_type}")
            return cached_results

        from datetime import datetime, timedelta
        from src.data.fetcher import fetch_stock_data
        from src.backtesting.performance_metrics import calculate_win_rate, calculate_average_return_per_signal

        # Use self.experiment_period for default slicing
        if custom_start_date and custom_end_date:
            start_date = custom_start_date
            end_date = custom_end_date
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.experiment_period)).strftime('%Y-%m-%d')

        historical_data = fetch_stock_data(ticker, start_date, end_date)

        if historical_data is None or len(historical_data) < 100:
            logger.warning(f"Not enough historical data for {ticker} to calculate reliability")
            raise ValueError(f"Insufficient historical data for {ticker}")

        strategy_class = self._get_strategy_class(strategy_type)
        if not strategy_class:
            logger.error(f"Unknown strategy type: {strategy_type}")
            raise ValueError(f"Unknown strategy type: {strategy_type}")
       
        strategy = strategy_class(historical_data, **(strategy_params or {}))
        signals_df = strategy.detect_signals()
        trade_signals = signals_df[signals_df['signal'].isin(['buy', 'sell'])]

        if len(trade_signals) < 1:
            logger.warning(f"Insufficient buy/sell signals for {ticker} with {strategy_type} strategy")
            raise ValueError(f"Insufficient buy/sell signals for {ticker} with {strategy_type} strategy")

        win_count = 0
        returns = []
        skipped = 0

        # Use self.preferred_period for holding period
        for _, row in trade_signals.iterrows():
            signal_date = pd.to_datetime(row['date'])
            entry_price = row['close']
            exit_date = signal_date + timedelta(days=self.preferred_period)
            # Find the closest available date in signals_df
            future_rows = signals_df[pd.to_datetime(signals_df['date']) >= exit_date]
            if future_rows.empty:
                skipped += 1
                continue
            exit_price = future_rows.iloc[0]['close']
            signal_return = (exit_price - entry_price) / entry_price * 100
            if row['signal'] == 'buy':
                if signal_return > 0:
                    win_count += 1
                returns.append(signal_return)
            elif row['signal'] == 'sell':
                if signal_return < 0:
                    win_count += 1
                returns.append(-signal_return)

        logger.info(f"Skipped {skipped} signals due to insufficient future data.")

        win_rate = (win_count / len(returns)) * 100 if returns else 0
        avg_return = sum(returns) / len(returns) if returns else 0

        # Buy & Hold return over the same period as signals
        first_price = historical_data['close'].iloc[0]
        last_price = historical_data['close'].iloc[-1]
        buy_hold_return = (last_price - first_price) / first_price * 100
        vs_bh = avg_return - buy_hold_return

        results = {
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'vs_bh': round(vs_bh, 2),
            'period': self.preferred_period,
            'signal_count': len(returns)
        }
        self._cache_results(ticker, strategy_type, strategy_params, results)
        logger.info(f"Calculated REAL metrics for {ticker} using {strategy_type}: {results}")
        return results




    def _get_strategy_class(self, strategy_type: str):
        if strategy_type == 'mean_reversion':
            from src.market_signals.mean_reversion import MeanReversionSignal
            return MeanReversionSignal
        elif strategy_type == 'ma_crossover':
            from src.market_signals.momentum import MACrossoverSignal
            return MACrossoverSignal
        elif strategy_type == 'volatility_breakout':
            from src.market_signals.momentum import VolatilityBreakoutSignal
            return VolatilityBreakoutSignal
        return None

    def _get_cache_file_path(self, ticker: str, strategy_type: str, strategy_params: Dict[str, Any] = None) -> str:
        param_str = '_'.join([f"{k}_{v}" for k, v in (strategy_params or {}).items()])
        filename = f"{ticker}_{strategy_type}_{param_str}.json".replace(' ', '_')
        return os.path.join(self.cache_dir, filename)

    def _get_cached_results(self, ticker: str, strategy_type: str, 
                           strategy_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        cache_file = self._get_cache_file_path(ticker, strategy_type, strategy_params)
        try:
            if os.path.exists(cache_file):
                if time.time() - os.path.getmtime(cache_file) > self.cache_expiry:
                    return None
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache for {ticker} {strategy_type}: {e}")
        return None

    def _cache_results(self, ticker: str, strategy_type: str, 
                      strategy_params: Dict[str, Any], results: Dict[str, Any]) -> None:
        cache_file = self._get_cache_file_path(ticker, strategy_type, strategy_params)
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f)
        except Exception as e:
            logger.error(f"Error caching results for {ticker} {strategy_type}: {e}")