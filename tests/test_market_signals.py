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
        self.stock_data = fetch_stock_data('SQM', 
                                          start_date='2024-01-01', 
                                          end_date=today.strftime('%Y-%m-%d'))
        
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
        
        # Initialize signal classes with real data and configuration
        window = self.config.get('mean_reversion_window', 50)
        threshold = self.config.get('mean_reversion_threshold', 1)
        
        print(f"Using mean reversion parameters from config: window={window}, threshold={threshold}")
        self.mean_reversion = MeanReversionSignal(self.stock_data, config=self.config)
        
        # Initialize momentum signals
        short_window = self.config.get('momentum_short_window', 20)
        long_window = self.config.get('momentum_long_window', 50)
        print(f"Using MA crossover parameters from config: short_window={short_window}, long_window={long_window}")
        self.ma_crossover = MACrossoverSignal(self.stock_data, config=self.config)
        
        atr_window = self.config.get('volatility_atr_window', 14)
        atr_multiplier = self.config.get('volatility_atr_multiplier', 1.5)
        breakout_window = self.config.get('volatility_breakout_window', 20)
        print(f"Using volatility breakout parameters from config: atr_window={atr_window}, atr_multiplier={atr_multiplier}, breakout_window={breakout_window}")
        self.volatility_breakout = VolatilityBreakoutSignal(self.stock_data, config=self.config)
        
        # Initialize signal processor
        self.signal_processor = SignalProcessor()
        self.signal_processor.register_signal_generator('mean_reversion', self.mean_reversion)
        self.signal_processor.register_signal_generator('ma_crossover', self.ma_crossover)
        self.signal_processor.register_signal_generator('volatility_breakout', self.volatility_breakout)

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
    
    def test_ma_crossover_signal(self):
        """Test moving average crossover signal detection"""
        signals_data = self.ma_crossover.detect_signals()
        
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
        
        print(f"\nMA Crossover Signal Stats (short={self.ma_crossover.short_window}, long={self.ma_crossover.long_window}):")
        print(f"Data points: {len(signals_data)}")
        print(f"Buy signals: {buy_count}")
        print(f"Sell signals: {sell_count}")
        print(f"Hold signals: {hold_count}")
    
    def test_volatility_breakout_signal(self):
        """Test volatility breakout signal detection"""
        try:
            signals_data = self.volatility_breakout.detect_signals()
            
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
            
            print(f"\nVolatility Breakout Signal Stats (ATR window={self.volatility_breakout.atr_window}, multiplier={self.volatility_breakout.atr_multiplier}, breakout window={self.volatility_breakout.breakout_window}):")
            print(f"Data points: {len(signals_data)}")
            print(f"Buy signals: {buy_count}")
            print(f"Sell signals: {sell_count}")
            print(f"Hold signals: {hold_count}")
        except ValueError as e:
            # This might occur if high/low data is not available
            print(f"Could not test volatility breakout: {str(e)}")

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
        """Test backtesting of mean reversion strategy"""
        # Generate signals
        signals_data = self.mean_reversion.detect_signals()
        
        # Run backtest
        backtest_results = self._run_backtest(signals_data)
        
        # Calculate buy & hold performance
        initial_capital = 10000
        first_price = signals_data['close'].iloc[0]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        
        # Create buy & hold equity curve
        buy_hold_equity = signals_data['close'].apply(
            lambda price: (shares_bought * price) + remaining_cash
        )
        
        # Calculate final values and returns
        strategy_final = backtest_results['portfolio_value'].iloc[-1]
        buy_hold_final = buy_hold_equity.iloc[-1]
        
        strategy_return_pct = ((strategy_final / initial_capital) - 1) * 100
        buy_hold_return_pct = ((buy_hold_final / initial_capital) - 1) * 100
        outperformance = strategy_return_pct - buy_hold_return_pct
        
        # Print comparison results
        print("\nMean Reversion Backtest Results:")
        print(f"Initial capital: ${initial_capital:.2f}")
        print(f"Final capital (strategy): ${strategy_final:.2f}")
        print(f"Total return (strategy): {strategy_return_pct:.2f}%")
        print("\nBuy & Hold Baseline:")
        print(f"Final capital (buy & hold): ${buy_hold_final:.2f}")
        print(f"Total return (buy & hold): {buy_hold_return_pct:.2f}%")
        print("\nComparison:")
        print(f"Strategy outperformance: {outperformance:.2f}%")
        
        # Add buy & hold data to results for CSV export
        backtest_results['buy_hold_value'] = buy_hold_equity.values
        backtest_results['buy_hold_return_pct'] = ((buy_hold_equity / initial_capital) - 1) * 100
        backtest_results['strategy_return_pct'] = ((backtest_results['portfolio_value'] / initial_capital) - 1) * 100
        backtest_results['outperformance'] = backtest_results['strategy_return_pct'] - backtest_results['buy_hold_return_pct']
        
        # Save results to CSV for inspection
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'backtest_results.csv'
        )
        backtest_results.to_csv(output_path, index=False)
        print(f"Detailed results saved to {output_path}")
        
        # Visualize the results
        print("Visualizing performance comparison...")
        self._visualize_backtest_comparison(backtest_results, signals_data, initial_capital)
    
    def test_backtest_momentum(self):
        """Test backtesting of momentum strategies"""
        initial_capital = 10000
        
        # Test MA Crossover
        ma_signals = self.ma_crossover.detect_signals()
        ma_results = self._run_backtest(ma_signals)
        
        # Calculate buy & hold for MA Crossover
        first_price = ma_signals['close'].iloc[0]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        
        ma_buy_hold = ma_signals['close'].apply(
            lambda price: (shares_bought * price) + remaining_cash
        )
        
        ma_strategy_final = ma_results['portfolio_value'].iloc[-1]
        ma_buy_hold_final = ma_buy_hold.iloc[-1]
        
        ma_strategy_return_pct = ((ma_strategy_final / initial_capital) - 1) * 100
        ma_buy_hold_return_pct = ((ma_buy_hold_final / initial_capital) - 1) * 100
        ma_outperformance = ma_strategy_return_pct - ma_buy_hold_return_pct
        
        # Print MA Crossover comparison
        print("\nMA Crossover Backtest Results:")
        print(f"Initial capital: ${initial_capital:.2f}")
        print(f"Final capital (strategy): ${ma_strategy_final:.2f}")
        print(f"Total return (strategy): {ma_strategy_return_pct:.2f}%")
        print("\nBuy & Hold Baseline:")
        print(f"Final capital (buy & hold): ${ma_buy_hold_final:.2f}")
        print(f"Total return (buy & hold): {ma_buy_hold_return_pct:.2f}%")
        print("\nComparison:")
        print(f"Strategy outperformance: {ma_outperformance:.2f}%")
        
        # Add buy & hold to results for CSV
        ma_results['buy_hold_value'] = ma_buy_hold.values
        ma_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ma_crossover_backtest_results.csv'
        )
        ma_results.to_csv(ma_output_path, index=False)
        
        # Visualize MA Crossover
        print("Visualizing performance comparison...")
        self._visualize_backtest_comparison(ma_results, ma_signals, initial_capital, 'ma_crossover_vs_baseline.png', 'MA Crossover')
        
        # Test Volatility Breakout if data is available
        try:
            vb_signals = self.volatility_breakout.detect_signals()
            vb_results = self._run_backtest(vb_signals)
            
            # Calculate buy & hold for Volatility Breakout
            first_price = vb_signals['close'].iloc[0]
            shares_bought = initial_capital // first_price
            remaining_cash = initial_capital - (shares_bought * first_price)
            
            vb_buy_hold = vb_signals['close'].apply(
                lambda price: (shares_bought * price) + remaining_cash
            )
            
            vb_strategy_final = vb_results['portfolio_value'].iloc[-1]
            vb_buy_hold_final = vb_buy_hold.iloc[-1]
            
            vb_strategy_return_pct = ((vb_strategy_final / initial_capital) - 1) * 100
            vb_buy_hold_return_pct = ((vb_buy_hold_final / initial_capital) - 1) * 100
            vb_outperformance = vb_strategy_return_pct - vb_buy_hold_return_pct
            
            # Print Volatility Breakout comparison
            print("\nVolatility Breakout Backtest Results:")
            print(f"Initial capital: ${initial_capital:.2f}")
            print(f"Final capital (strategy): ${vb_strategy_final:.2f}")
            print(f"Total return (strategy): {vb_strategy_return_pct:.2f}%")
            print("\nBuy & Hold Baseline:")
            print(f"Final capital (buy & hold): ${vb_buy_hold_final:.2f}")
            print(f"Total return (buy & hold): {vb_buy_hold_return_pct:.2f}%")
            print("\nComparison:")
            print(f"Strategy outperformance: {vb_outperformance:.2f}%")
            
            # Add buy & hold to results for CSV
            vb_results['buy_hold_value'] = vb_buy_hold.values
            vb_output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'volatility_breakout_backtest_results.csv'
            )
            vb_results.to_csv(vb_output_path, index=False)
            
            # Visualize Volatility Breakout
            print("Visualizing performance comparison...")
            self._visualize_backtest_comparison(vb_results, vb_signals, initial_capital, 
                                            'volatility_breakout_vs_baseline.png', 'Volatility Breakout')
        except ValueError as e:
            print(f"Could not backtest volatility breakout strategy: {str(e)}")


    def test_backtest_momentum(self):
        """Test backtesting of momentum strategies"""
        # Test MA Crossover
        ma_signals = self.ma_crossover.detect_signals()
        ma_results = self._run_backtest(ma_signals)
        ma_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ma_crossover_backtest_results.csv'
        )
        ma_results.to_csv(ma_output_path, index=False)
        self._visualize_backtest_comparison(ma_results, ma_signals, 10000, 'ma_crossover_vs_baseline.png', 'MA Crossover')
        
        # Test Volatility Breakout if data is available
        try:
            vb_signals = self.volatility_breakout.detect_signals()
            vb_results = self._run_backtest(vb_signals)
            vb_output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'volatility_breakout_backtest_results.csv'
            )
            vb_results.to_csv(vb_output_path, index=False)
            self._visualize_backtest_comparison(vb_results, vb_signals, 10000, 'volatility_breakout_vs_baseline.png', 'Volatility Breakout')
        except ValueError as e:
            print(f"Could not backtest volatility breakout strategy: {str(e)}")
        
    def _visualize_backtest_comparison(self, backtest_results, signals_data, initial_capital, 
                                       save_name='mean_reversion_vs_baseline.png', strategy_name='Mean Reversion'):
        """
        Create a simple visualization of strategy vs buy & hold performance
        
        Args:
            backtest_results: DataFrame with backtest results
            signals_data: DataFrame with signal and price data
            initial_capital: Starting capital amount
            save_name: Filename to save the chart
            strategy_name: Name of the strategy for chart labels
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
                    label=f'{strategy_name} Strategy', color='blue')
            plt.plot(signals_data['date'], buy_hold_equity, 
                    label='Buy & Hold Baseline', color='green', linestyle='--')
            plt.title(f'{strategy_name} Strategy Performance Comparison')
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
            plt.savefig(save_name)
            plt.close()
            print(f"Performance comparison chart saved as '{save_name}'")
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def _run_backtest(self, signals_df, initial_capital=10000):
        """
        Enhanced backtest helper method with position sizing and risk management
        
        Args:
            signals_df: DataFrame with signal column
            initial_capital: Starting capital amount
            
        Returns:
            DataFrame with backtest results
        """
        df = signals_df.copy()
        
        # Initialize portfolio metrics
        df['position_pct'] = 0.0  # Position as percentage of portfolio (0-1)
        df['cash'] = initial_capital
        df['shares'] = 0
        df['portfolio_value'] = initial_capital
        df['equity_pct'] = 0.0  # Equity percentage of total portfolio
        df['risk_budget'] = 0.02  # Risk 2% per trade
        df['max_position_pct'] = 1  # Maximum position size (80% of portfolio)
        
        # Set up signal strength (for position sizing)
        df['signal_strength'] = 0.0
        
        # Consecutive signal counters
        consecutive_buys = 0
        consecutive_sells = 0
        
        for i in range(1, len(df)):
            # Copy previous values
            df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
            df.loc[df.index[i], 'shares'] = df.loc[df.index[i-1], 'shares']
            df.loc[df.index[i], 'position_pct'] = df.loc[df.index[i-1], 'position_pct']
            
            # Current price and signal
            price = df.loc[df.index[i], 'close']
            signal = df.loc[df.index[i], 'signal']
            position_pct = df.loc[df.index[i], 'position_pct']
            
            # Calculate current portfolio value
            shares_value = df.loc[df.index[i], 'shares'] * price
            portfolio_value = df.loc[df.index[i], 'cash'] + shares_value
            df.loc[df.index[i], 'portfolio_value'] = portfolio_value
            
            # Calculate equity percentage for this bar
            if portfolio_value > 0:
                df.loc[df.index[i], 'equity_pct'] = shares_value / portfolio_value
            
            # Update signal counters and strength
            if signal == 'buy':
                consecutive_buys += 1
                consecutive_sells = 0
                # Signal strength increases with consecutive buys (capped at 1.0)
                df.loc[df.index[i], 'signal_strength'] = min(1.0, consecutive_buys * 0.25)
            elif signal == 'sell':
                consecutive_buys = 0
                consecutive_sells += 1
                # Signal strength increases with consecutive sells (capped at 1.0)
                df.loc[df.index[i], 'signal_strength'] = min(1.0, consecutive_sells * 0.25)
            else:
                # Hold signals decrease strength
                df.loc[df.index[i], 'signal_strength'] = max(0.0, df.loc[df.index[i-1], 'signal_strength'] - 0.1)
                
            # Execute trading logic with position sizing
            if signal == 'buy':
                # Scale the target position based on signal strength and available cash
                max_position_pct = df.loc[df.index[i], 'max_position_pct']
                # Lucky number seven
                target_position_pct = position_pct + (0.7 * df.loc[df.index[i], 'signal_strength'])
                target_position_pct = min(target_position_pct, max_position_pct)
                
                # Calculate position delta (how much to add)
                position_delta_pct = target_position_pct - position_pct
                
                # Only buy if we want to increase position
                if position_delta_pct > 0.01:  # Minimum 1% increase to avoid small trades
                    # Calculate cash to use for this increment
                    cash_to_use = portfolio_value * position_delta_pct
                    cash_available = df.loc[df.index[i], 'cash']
                    
                    # Make sure we have enough cash
                    cash_to_use = min(cash_to_use, cash_available * 0.95)
                    
                    # Calculate shares to buy
                    if price > 0:
                        shares_to_buy = cash_to_use // price
                        
                        if shares_to_buy > 0:
                            cost = shares_to_buy * price
                            
                            # Update portfolio
                            df.loc[df.index[i], 'shares'] += shares_to_buy
                            df.loc[df.index[i], 'cash'] -= cost
                            
                            # Recalculate position percentage
                            new_shares_value = df.loc[df.index[i], 'shares'] * price
                            new_portfolio_value = df.loc[df.index[i], 'cash'] + new_shares_value
                            df.loc[df.index[i], 'position_pct'] = new_shares_value / new_portfolio_value
                            df.loc[df.index[i], 'portfolio_value'] = new_portfolio_value
                
            elif signal == 'sell':
                # Scale the position reduction based on signal strength
                target_position_pct = position_pct * (1 - (0.5 * df.loc[df.index[i], 'signal_strength']))
                
                # Calculate position delta (how much to reduce)
                position_delta_pct = position_pct - target_position_pct
                
                # Only sell if reduction is significant
                if position_delta_pct > 0.01 and df.loc[df.index[i], 'shares'] > 0:
                    # Calculate shares to sell
                    shares_to_sell = df.loc[df.index[i], 'shares'] * position_delta_pct / position_pct
                    shares_to_sell = min(shares_to_sell, df.loc[df.index[i], 'shares'])  # Don't sell more than we have
                    
                    # Round to whole shares
                    shares_to_sell = int(shares_to_sell)
                    
                    if shares_to_sell > 0:
                        sale_value = shares_to_sell * price
                        
                        # Update portfolio
                        df.loc[df.index[i], 'shares'] -= shares_to_sell
                        df.loc[df.index[i], 'cash'] += sale_value
                        
                        # Recalculate position percentage
                        new_shares_value = df.loc[df.index[i], 'shares'] * price
                        new_portfolio_value = df.loc[df.index[i], 'cash'] + new_shares_value
                        
                        if new_portfolio_value > 0:
                            df.loc[df.index[i], 'position_pct'] = new_shares_value / new_portfolio_value
                        else:
                            df.loc[df.index[i], 'position_pct'] = 0.0
                            
                        df.loc[df.index[i], 'portfolio_value'] = new_portfolio_value
            
            # For hold signals, update portfolio value but don't change position
            
            # Safety check - recalculate portfolio value at the end
            shares_value = df.loc[df.index[i], 'shares'] * price
            portfolio_value = df.loc[df.index[i], 'cash'] + shares_value
            df.loc[df.index[i], 'portfolio_value'] = portfolio_value
            
            if portfolio_value > 0:
                df.loc[df.index[i], 'position_pct'] = shares_value / portfolio_value
            else:
                df.loc[df.index[i], 'position_pct'] = 0.0
        
        return df

if __name__ == '__main__':
    unittest.main()