import logging
from typing import Dict, Any, Optional
import os
import json
import time

from src.backtesting.strategy_evaluator import StrategyEvaluator

logger = logging.getLogger(__name__)

class SignalReliabilityService:
    """
    Service to provide reliability metrics for trading signals.
    """
    def __init__(self, config=None):
        """
        Initialize the signal reliability service.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or {}
        self.evaluator = StrategyEvaluator(config)
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'cache', 
            'reliability'
        )
        self.cache_expiry = self.config.get('reliability_cache_expiry', 24 * 60 * 60)  # 24 hours in seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)


    def get_signal_reliability(self, ticker: str, strategy_type: str, 
                            strategy_params: Dict[str, Any] = None,
                            preferred_period: int = 30,
                            custom_start_date: str = None,
                            custom_end_date: str = None) -> Dict[str, Any]:
        """Calculate reliability metrics for a specific signal using ONLY real data.
        Optionally allows custom date range for historical data to match user experiment period.
        """
        # Check cache first for performance
        cached_results = self._get_cached_results(ticker, strategy_type, strategy_params)
        if cached_results:
            logger.info(f"Using cached reliability metrics for {ticker} {strategy_type}")
            return cached_results

        from datetime import datetime, timedelta
        from src.data.fetcher import fetch_stock_data
        from src.backtesting.performance_metrics import calculate_win_rate, calculate_average_return_per_signal

        # Use custom date range if provided, otherwise default to 2 years
        if custom_start_date and custom_end_date:
            start_date = custom_start_date
            end_date = custom_end_date
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years

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
        buy_signals = signals_df[signals_df['signal'] == 'buy']

        if len(buy_signals) < 0:
            logger.warning(f"Insufficient buy signals for {ticker} with {strategy_type} strategy")
            raise ValueError(f"Insufficient buy signals for {ticker} with {strategy_type} strategy")

        win_rate = calculate_win_rate(signals_df, holding_period=preferred_period)
        avg_return = calculate_average_return_per_signal(signals_df, holding_period=preferred_period)
        first_price = historical_data['close'].iloc[0]
        last_price = historical_data['close'].iloc[-1]
        buy_hold_return = (last_price - first_price) / first_price * 100

        outperform_count = 0
        valid_signals = 0
        for idx in buy_signals.index:
            if idx + preferred_period >= len(signals_df):
                continue
            entry_price = signals_df.loc[idx, 'close']
            exit_price = signals_df.loc[idx + preferred_period, 'close']
            signal_return = (exit_price - entry_price) / entry_price * 100
            if signal_return > buy_hold_return:
                outperform_count += 1
            valid_signals += 1

        market_outperformance = (outperform_count / valid_signals * 100) if valid_signals > 0 else 0

        results = {
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 1),
            'market_outperformance': round(market_outperformance, 1),
            'period': preferred_period,
            'signal_count': len(buy_signals)
        }

        self._cache_results(ticker, strategy_type, strategy_params, results)
        logger.info(f"Calculated REAL metrics for {ticker} using {strategy_type}: {results}")
        return results


    def _get_strategy_class(self, strategy_type: str):
        """Get the strategy class based on strategy type."""
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
        """Get the cache file path for a specific strategy and parameters."""
        # Create a unique key based on ticker, strategy_type and parameters
        param_str = '_'.join([f"{k}_{v}" for k, v in (strategy_params or {}).items()])
        filename = f"{ticker}_{strategy_type}_{param_str}.json".replace(' ', '_')
        return os.path.join(self.cache_dir, filename)
    
    def _get_cached_results(self, ticker: str, strategy_type: str, 
                           strategy_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached results if they exist and are not expired."""
        cache_file = self._get_cache_file_path(ticker, strategy_type, strategy_params)
        
        try:
            if os.path.exists(cache_file):
                # Check if cache is expired
                if time.time() - os.path.getmtime(cache_file) > self.cache_expiry:
                    return None
                
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache for {ticker} {strategy_type}: {e}")
        
        return None
    
    def _cache_results(self, ticker: str, strategy_type: str, 
                      strategy_params: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Cache the evaluation results."""
        #cache_file = self._get_cache_file_path(ticker, strategy_type, strategy_params)
        
        #try:
            #with open(cache_file, 'w') as f:
                #json.dump(results, f)
       # except Exception as e:
            #logger.error(f"Error caching results for {ticker} {strategy_type}: {e}")
        pass
    