import unittest
import pandas as pd
import yaml
import os
from src.data.fetcher import fetch_stock_data
from src.market_signals.mean_reversion import MeanReversionSignal
from src.market_signals.signal_processor import SignalProcessor
from datetime import datetime

class TestMarketSignals(unittest.TestCase):
    def setUp(self):
        # Load configuration from YAML file
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 
            'config.yml'
        )
        
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            self.config = {}
        
        # Get today's date
        today = datetime.today()
        
        # Fetch real stock data for testing
        self.stock_data = fetch_stock_data('SBLK', 
                                          start_date='2024-01-01', 
                                          end_date=today.strftime('%Y-%m-%d'))
        
        if self.stock_data is None:
            print("Warning: Could not fetch real stock data. Using fallback test data.")
            self.stock_data = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=100),
                'close': [100 + i * 0.1 for i in range(100)],
                'open': [99 + i * 0.1 for i in range(100)],
                'high': [101 + i * 0.1 for i in range(100)],
                'low': [98 + i * 0.1 for i in range(100)],
                'volume': [1000000 for _ in range(100)]
            })
        
        # Initialize signal classes with real data and configuration
        window = self.config.get('mean_reversion_window', 50)
        threshold = self.config.get('mean_reversion_threshold', 1.5)
        
        print(f"Using mean reversion parameters from config: window={window}, threshold={threshold}")
        self.mean_reversion = MeanReversionSignal(self.stock_data, config=self.config)
        
        # Initialize signal processor
        self.signal_processor = SignalProcessor()
        self.signal_processor.register_signal_generator('mean_reversion', self.mean_reversion)

    def test_mean_reversion_signal(self):
        """Test mean reversion signal detection with real stock data"""
        signals_data = self.mean_reversion.detect_signals()
        
        # Verify signals are generated
        self.assertIsNotNone(signals_data)
        self.assertIn('signal', signals_data.columns)
        
        # Check that signals are valid
        for signal in signals_data['signal'].unique():
            self.assertIn(signal, ['buy', 'sell', 'hold'])
        
        # Print some basic stats about the signals
        buy_count = (signals_data['signal'] == 'buy').sum()
        sell_count = (signals_data['signal'] == 'sell').sum()
        hold_count = (signals_data['signal'] == 'hold').sum()
        
        print(f"\nMean Reversion Signal Stats (window={self.mean_reversion.window}, threshold={self.mean_reversion.threshold}):")
        print(f"Data points: {len(signals_data)}")
        print(f"Buy signals: {buy_count}")
        print(f"Sell signals: {sell_count}")
        print(f"Hold signals: {hold_count}")
        

    def test_signal_processing_mean_reversion(self):
        """Test signal processing with mean reversion signals only"""
        if self.stock_data is not None and len(self.stock_data) > 50:
            # Generate mean reversion signals
            mr_signals = self.mean_reversion.detect_signals()
            
            # Take a sample point
            idx = min(30, len(mr_signals) - 1)  # Ensure we have a valid index
            mean_reversion_signal = mr_signals['signal'].iloc[idx]
            
            # Process the signal using the parameter approach
            processed_signals = self.signal_processor.process_signals(mean_reversion_signal=mean_reversion_signal)
            
            # Check the result
            self.assertIsInstance(processed_signals, dict)
            self.assertIn('mean_reversion', processed_signals)
            self.assertIn('combined', processed_signals)
            
            # When only mean reversion signal is provided, it should be in the results
            self.assertEqual(processed_signals['mean_reversion'], mean_reversion_signal)
            
            # Test with market data approach
            processed_data_signals = self.signal_processor.process_signals(market_data=self.stock_data)
            
            # Check that we got signals
            self.assertIsInstance(processed_data_signals, dict)
            self.assertIn('mean_reversion', processed_data_signals)
            self.assertIn('combined', processed_data_signals)

    def test_backtest_mean_reversion(self):
        """Test backtesting of mean reversion strategy with real data"""
        if self.stock_data is None or len(self.stock_data) < 100:
            self.skipTest("Not enough stock data for proper backtest")
        
        # Generate signals
        signals_data = self.mean_reversion.detect_signals()
        
        # Run a simple backtest
        initial_capital = 10000
        backtest_results = self._run_backtest(signals_data, initial_capital)
        
        # Calculate performance metrics
        final_capital = backtest_results['portfolio_value'].iloc[-1]
        total_return = ((final_capital / initial_capital) - 1) * 100
        
        # Calculate buy & hold baseline performance
        first_price = signals_data['close'].iloc[0]
        last_price = signals_data['close'].iloc[-1]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        final_buy_hold_value = (shares_bought * last_price) + remaining_cash
        buy_hold_return = ((final_buy_hold_value / initial_capital) - 1) * 100
        
        # Print backtest results with baseline comparison
        print(f"\nMean Reversion Backtest Results:")
        print(f"Initial capital: ${initial_capital}")
        print(f"Final capital (strategy): ${final_capital:.2f}")
        print(f"Total return (strategy): {total_return:.2f}%")
        print(f"\nBuy & Hold Baseline:")
        print(f"Final capital (buy & hold): ${final_buy_hold_value:.2f}")
        print(f"Total return (buy & hold): {buy_hold_return:.2f}%")
        print(f"\nComparison:")
        outperformance = total_return - buy_hold_return
        print(f"Strategy outperformance: {outperformance:.2f}%")
        
        # We don't assert profitability, just that the backtest ran
        self.assertGreater(len(backtest_results), 0)
        
        # Optional: Add a simple visualization of the performance comparison
        print("Visualizing performance comparison...")
        self._visualize_backtest_comparison(backtest_results, signals_data, initial_capital)
    def _visualize_backtest_comparison(self, backtest_results, signals_data, initial_capital):
        """
        Create a simple visualization of strategy vs buy & hold performance
        
        Args:
            backtest_results: DataFrame with backtest results
            signals_data: DataFrame with signal and price data
            initial_capital: Starting capital amount
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Calculate buy & hold equity curve
            first_price = signals_data['close'].iloc[0]
            shares_bought = initial_capital // first_price
            remaining_cash = initial_capital - (shares_bought * first_price)
            
            buy_hold_equity = signals_data['close'].apply(
                lambda price: (shares_bought * price) + remaining_cash
            )
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot the two equity curves
            plt.subplot(2, 1, 1)
            plt.plot(signals_data['date'], backtest_results['portfolio_value'], 
                    label='Mean Reversion Strategy', color='blue')
            plt.plot(signals_data['date'], buy_hold_equity, 
                    label='Buy & Hold Baseline', color='green', linestyle='--')
            plt.title('Strategy Performance Comparison')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            
            # Format x-axis to show dates nicely
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            # Plot the relative performance (strategy / buy & hold)
            plt.subplot(2, 1, 2)
            relative_performance = backtest_results['portfolio_value'] / buy_hold_equity
            plt.plot(signals_data['date'], relative_performance, color='purple')
            plt.axhline(y=1.0, color='black', linestyle='--')
            plt.title('Relative Performance (Strategy / Buy & Hold)')
            plt.xlabel('Date')
            plt.ylabel('Ratio')
            plt.grid(True)
            
            # Format x-axis for the second subplot
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('mean_reversion_vs_baseline.png')
            plt.close()
            print("Performance comparison chart saved as 'mean_reversion_vs_baseline.png'")
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def _run_backtest(self, signals_df, initial_capital=10000):
        """
        Simple backtest helper method
        
        Args:
            signals_df: DataFrame with signal column
            initial_capital: Starting capital amount
            
        Returns:
            DataFrame with backtest results
        """
        df = signals_df.copy()
        df['position'] = 0  # 0: no position, 1: long position
        df['cash'] = initial_capital
        df['shares'] = 0
        df['portfolio_value'] = initial_capital
        
        for i in range(1, len(df)):
            # Copy previous values
            df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
            df.loc[df.index[i], 'shares'] = df.loc[df.index[i-1], 'shares']
            df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
            
            # Current price and signal
            price = df.loc[df.index[i], 'close']
            signal = df.loc[df.index[i], 'signal']
            position = df.loc[df.index[i], 'position']
            
            # Execute trading logic
            if signal == 'buy' and position == 0:
                # Buy with 95% of available cash
                cash_to_use = df.loc[df.index[i-1], 'cash'] * 0.95
                shares_to_buy = cash_to_use // price
                cost = shares_to_buy * price
                
                df.loc[df.index[i], 'shares'] = shares_to_buy
                df.loc[df.index[i], 'cash'] -= cost
                df.loc[df.index[i], 'position'] = 1
                
            elif signal == 'sell' and position == 1:
                # Sell all shares
                sale_value = df.loc[df.index[i-1], 'shares'] * price
                
                df.loc[df.index[i], 'cash'] += sale_value
                df.loc[df.index[i], 'shares'] = 0
                df.loc[df.index[i], 'position'] = 0
            
            # Calculate portfolio value
            shares_value = df.loc[df.index[i], 'shares'] * price
            df.loc[df.index[i], 'portfolio_value'] = df.loc[df.index[i], 'cash'] + shares_value
        
        return df

# The following code provides examples of how to add tests for additional signals in the future
"""
# Example for adding a new signal test (commented out for now)
class TestMomentumSignals(unittest.TestCase):
    def setUp(self):
        # Fetch real stock data for testing
        self.stock_data = fetch_stock_data('AAPL', 
                                         start_date='2023-01-01', 
                                         end_date='2023-12-31')
        
        # Initialize with fallback data if needed
        if self.stock_data is None:
            # Create fallback test data
            pass
            
    def test_momentum_signal(self):
        # Test implementation for momentum signals
        pass
"""

if __name__ == '__main__':
    unittest.main()