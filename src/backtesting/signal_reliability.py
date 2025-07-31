import logging
import pandas as pd
from typing import Dict, Any
from datetime import timedelta

from src.backtesting.strategy_evaluator import StrategyEvaluator
from src.utils.config_loader import get_full_config
from src.backtesting.performance_metrics import (
    extract_trades_from_signals,
    calculate_win_rate_from_trades,
    calculate_average_return_from_trades,
)

logger = logging.getLogger(__name__)

class SignalReliabilityService:
    def __init__(self):
        self.evaluator = StrategyEvaluator({})  # Pass empty or default config if needed

    def get_signal_reliability(self, ticker: str, strategy_type: str, 
                               strategy_params: Dict[str, Any] = None,
                               custom_start_date: str = None,
                               custom_end_date: str = None) -> Dict[str, Any]:
        from datetime import datetime
        from src.data.fetcher import fetch_stock_data

        config = get_full_config()
        reliability_cfg = config.get('reliability', {})
        experiment_period = reliability_cfg.get('experiment_period', 365)

        # Use experiment_period for default slicing
        if custom_start_date and custom_end_date:
            start_date = custom_start_date
            end_date = custom_end_date
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=experiment_period)).strftime('%Y-%m-%d')

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

        # Use signal-to-signal trade logic
        trades = extract_trades_from_signals(signals_df)
        if not trades:
            logger.warning(f"No valid trades for {ticker} with {strategy_type} strategy")
            raise ValueError(f"No valid trades for {ticker} with {strategy_type} strategy")

        win_rate = calculate_win_rate_from_trades(trades)
        avg_return = calculate_average_return_from_trades(trades)

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
            'period': 'signal-to-signal',
            'signal_count': len(trades)
        }
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