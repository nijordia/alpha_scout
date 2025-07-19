# For running this test use: python -m unittest tests.test_market_signals
import unittest
import pandas as pd
import yaml
import os
from src.data.fetcher import fetch_stock_data
from src.market_signals.mean_reversion import MeanReversionSignal
from src.market_signals.momentum import MACrossoverSignal, VolatilityBreakoutSignal
from src.market_signals.signal_processor import SignalProcessor
from datetime import datetime

class TestMarketSignals(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'config.yml'
        )
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            self.config = {}

        today = datetime.today()
        # Only test one stock here
        self.stock_data = fetch_stock_data('KO', start_date='2023-10-01', end_date=today.strftime('%Y-%m-%d'))
        if self.stock_data is None:
            print("Warning: Could not fetch real stock data. Using fallback test data.")
            self.stock_data = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=100),
                'close': [100 + i * 0.1 for i in range(100)],
                'open': [99 + i * 0.1 for i in range(100)],
                'high': [101 + i * 0.1 for i in range(100)],
                'low': [98 + i * 0.1 for i in range(100)],
                'volume': [1000000 for _ in range(100)]
            })

        window = self.config.get('mean_reversion_window', 50)
        threshold = self.config.get('mean_reversion_threshold', 1)
        self.mean_reversion = MeanReversionSignal(self.stock_data, config=self.config)
        short_window = self.config.get('momentum_short_window', 20)
        long_window = self.config.get('momentum_long_window', 50)
        self.ma_crossover = MACrossoverSignal(self.stock_data, config=self.config)
        atr_window = self.config.get('volatility_atr_window', 14)
        atr_multiplier = self.config.get('volatility_atr_multiplier', 1.5)
        breakout_window = self.config.get('volatility_breakout_window', 20)
        self.volatility_breakout = VolatilityBreakoutSignal(self.stock_data, config=self.config)

    def _run_backtest(self, signals_df, initial_capital=10000):
        df = signals_df.copy()
        df['position_pct'] = 0.0
        df['cash'] = initial_capital
        df['shares'] = 0
        df['portfolio_value'] = initial_capital
        df['pending_signal'] = None
        # Remove 'position' if it exists
        if 'position' in df.columns:
            df.drop(columns=['position'], inplace=True)
        for i in range(1, len(df)):
            df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
            df.loc[df.index[i], 'shares'] = df.loc[df.index[i-1], 'shares']
            df.loc[df.index[i], 'position_pct'] = df.loc[df.index[i-1], 'position_pct']
            pending_signal = df.loc[df.index[i-1], 'pending_signal']
            # Use position_pct to determine if invested (invested if > 0.01)
            invested = df.loc[df.index[i], 'position_pct'] > 0.01
            if pending_signal == 'buy' and not invested:
                open_price = df.loc[df.index[i], 'open']
                cash_to_use = df.loc[df.index[i], 'cash'] * 0.95
                shares_to_buy = cash_to_use // open_price
                cost = shares_to_buy * open_price
                df.loc[df.index[i], 'shares'] = shares_to_buy
                df.loc[df.index[i], 'cash'] -= cost
            elif pending_signal == 'sell' and invested:
                open_price = df.loc[df.index[i], 'open']
                sale_value = df.loc[df.index[i], 'shares'] * open_price
                df.loc[df.index[i], 'cash'] += sale_value
                df.loc[df.index[i], 'shares'] = 0
            df.loc[df.index[i], 'pending_signal'] = df.loc[df.index[i], 'signal']
            close_price = df.loc[df.index[i], 'close']
            shares_value = df.loc[df.index[i], 'shares'] * close_price
            df.loc[df.index[i], 'portfolio_value'] = df.loc[df.index[i], 'cash'] + shares_value
            # Update position_pct (fraction of portfolio in shares)
            if df.loc[df.index[i], 'portfolio_value'] > 0:
                df.loc[df.index[i], 'position_pct'] = shares_value / df.loc[df.index[i], 'portfolio_value']
            else:
                df.loc[df.index[i], 'position_pct'] = 0.0
        return df



    def test_backtest_mean_reversion(self):
        signals_data = self.mean_reversion.detect_signals()
        backtest_results = self._run_backtest(signals_data)
        initial_capital = 10000
        first_price = signals_data['close'].iloc[0]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        buy_hold_equity = signals_data['close'].apply(
            lambda price: (shares_bought * price) + remaining_cash
        )
        backtest_results['buy_hold_value'] = buy_hold_equity.values
        backtest_results['buy_hold_return_pct'] = ((buy_hold_equity / initial_capital) - 1) * 100
        backtest_results['strategy_return_pct'] = ((backtest_results['portfolio_value'] / initial_capital) - 1) * 100
        backtest_results['outperformance'] = backtest_results['strategy_return_pct'] - backtest_results['buy_hold_return_pct']
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'src', 'docs', 'mean_reversion_backtest_results.csv'
        )
        backtest_results.to_csv(output_path, index=False)
        print(f"Mean Reversion backtest results saved to {output_path}")

    def test_backtest_ma_crossover(self):
        ma_signals = self.ma_crossover.detect_signals()
        ma_results = self._run_backtest(ma_signals)
        initial_capital = 10000
        first_price = ma_signals['close'].iloc[0]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        ma_buy_hold = ma_signals['close'].apply(
            lambda price: (shares_bought * price) + remaining_cash
        )
        ma_results['buy_hold_value'] = ma_buy_hold.values
        ma_results['buy_hold_return_pct'] = ((ma_buy_hold / initial_capital) - 1) * 100
        ma_results['strategy_return_pct'] = ((ma_results['portfolio_value'] / initial_capital) - 1) * 100
        ma_results['outperformance'] = ma_results['strategy_return_pct'] - ma_results['buy_hold_return_pct']
        ma_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'src', 'docs', 'ma_crossover_backtest_results.csv'
        )
        ma_results.to_csv(ma_output_path, index=False)
        print(f"MA Crossover backtest results saved to {ma_output_path}")

    def test_backtest_volatility_breakout(self):
        try:
            vb_signals = self.volatility_breakout.detect_signals()
            vb_results = self._run_backtest(vb_signals)
            initial_capital = 10000
            first_price = vb_signals['close'].iloc[0]
            shares_bought = initial_capital // first_price
            remaining_cash = initial_capital - (shares_bought * first_price)
            vb_buy_hold = vb_signals['close'].apply(
                lambda price: (shares_bought * price) + remaining_cash
            )
            vb_results['buy_hold_value'] = vb_buy_hold.values
            vb_results['buy_hold_return_pct'] = ((vb_buy_hold / initial_capital) - 1) * 100
            vb_results['strategy_return_pct'] = ((vb_results['portfolio_value'] / initial_capital) - 1) * 100
            vb_results['outperformance'] = vb_results['strategy_return_pct'] - vb_results['buy_hold_return_pct']
            vb_output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'src', 'docs', 'volatility_breakout_backtest_results.csv'
            )
            vb_results.to_csv(vb_output_path, index=False)
            print(f"Volatility Breakout backtest results saved to {vb_output_path}")
        except ValueError as e:
            print(f"Could not backtest volatility breakout strategy: {str(e)}")

if __name__ == '__main__':
    unittest.main()
