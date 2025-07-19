import pandas as pd
import logging
from typing import Dict, Any, List, Optional

from src.data.fetcher import fetch_stock_data
from src.backtesting.performance_metrics import (
    calculate_win_rate,
    calculate_average_return_per_signal, 
    calculate_win_vs_market
)

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """
    Evaluates the historical performance of trading strategies.
    """
    def __init__(self, config=None):
        """
        Initialize the strategy evaluator.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with backtesting parameters
        """
        self.config = config or {}
        self.backtest_days = self.config.get('backtest_days', 365)
        self.reliability_periods = self.config.get('reliability_periods', [14, 30, 90])
        
    def evaluate_strategy(self, ticker: str, strategy_class, strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate a strategy's historical performance for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        strategy_class : class
            Strategy class to evaluate (e.g., MeanReversionSignal, MACrossoverSignal)
        strategy_params : dict, optional
            Parameters to pass to the strategy constructor
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        try:
            # Fetch historical data
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.backtest_days)).strftime('%Y-%m-%d')
            data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)




            if data is None or len(data) < 30:  # Need enough data for meaningful backtest
                logger.warning(f"Not enough data for {ticker} to perform backtest")
                return self._get_empty_results()
            
            # Initialize the strategy with the data
            strategy_params = strategy_params or {}
            strategy = strategy_class(data, **strategy_params)
            
            # Generate signals
            signals_df = strategy.detect_signals()
            if signals_df.empty:
                logger.warning(f"No signals generated for {ticker}")
                return self._get_empty_results()
            
            # Calculate metrics for each holding period
            results = {}
            for period in self.reliability_periods:
                period_results = {}
                
                # Calculate win rate
                period_results['win_rate'] = calculate_win_rate(signals_df, holding_period=period)
                
                # Calculate average return
                period_results['avg_return'] = calculate_average_return_per_signal(signals_df, holding_period=period)
                
                # Calculate outperformance vs buy & hold
                period_results['market_outperformance'] = calculate_win_vs_market(
                    signals_df, signals_df, holding_period=period
                )
                
                results[f'{period}d'] = period_results
            
            return results
        
        except Exception as e:
            logger.error(f"Error evaluating strategy for {ticker}: {e}", exc_info=True)
            return self._get_empty_results()
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results structure when evaluation fails."""
        results = {}
        for period in self.reliability_periods:
            results[f'{period}d'] = {
                'win_rate': 0,
                'avg_return': 0,
                'market_outperformance': 0
            }
        return results