import unittest
import os
from src.data.fetcher import fetch_stock_data
from src.market_signals.mean_reversion import MeanReversionSignal
from src.market_signals.momentum import MACrossoverSignal, VolatilityBreakoutSignal
from src.backtesting.signal_reliability import SignalReliabilityService
import pandas as pd
import yaml

class TestSignalReliability(unittest.TestCase):
    def test_signal_reliability_for_stock(self):
        stock = "SBLK"
        start_date = "2023-01-01"
        end_date = "2025-07-27"

        data = fetch_stock_data(stock, start_date, end_date)
        self.assertIsNotNone(data)
        print(f"Fetched {len(data)} rows for {stock}")

        # Mean Reversion
        mr_params = {'window': 150, 'threshold': 1.5}
        mean_rev = MeanReversionSignal(data, **mr_params)
        mr_signals = mean_rev.detect_signals()
        mr_signals.to_csv(f"{stock}_mean_reversion_signals.csv", index=False)

        # MA Crossover
        ma_params = {'short_window': 30, 'long_window': 60}
        ma_crossover = MACrossoverSignal(data, **ma_params)
        ma_signals = ma_crossover.detect_signals()
        ma_signals.to_csv(f"{stock}_ma_crossover_signals.csv", index=False)

        # Instantiate reliability service with no config
        reliability_service = SignalReliabilityService()

        # Mean Reversion metrics
        mr_metrics = reliability_service.get_signal_reliability(
            ticker=stock,
            strategy_type='mean_reversion',
            strategy_params=mr_params,
            custom_start_date=start_date,
            custom_end_date=end_date
        )
        print("Mean Reversion metrics:", mr_metrics)

        # MA Crossover metrics
        ma_metrics = reliability_service.get_signal_reliability(
            ticker=stock,
            strategy_type='ma_crossover',
            strategy_params=ma_params,
            custom_start_date=start_date,
            custom_end_date=end_date
        )
        print("MA Crossover metrics:", ma_metrics)

        # Volatility Breakout
        vb_params = {'atr_window': 14, 'atr_multiplier': 1.5, 'breakout_window': 20}
        vb_breakout = VolatilityBreakoutSignal(data, **vb_params)
        vb_signals = vb_breakout.detect_signals()
        vb_metrics = reliability_service.get_signal_reliability(
            ticker=stock,
            strategy_type='volatility_breakout',
            strategy_params=vb_params,
            custom_start_date=start_date,
            custom_end_date=end_date
        )
        print("Volatility Breakout metrics:", vb_metrics)

        vb_signals.to_csv(f"{stock}_volatility_breakout_signals.csv", index=False)

if __name__ == "__main__":
    unittest.main()