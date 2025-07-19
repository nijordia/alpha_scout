# For running this test use: python -m unittest tests.test_market_signals
# For signal_reliability use: python -m unittest tests.test_market_signals.TestMarketSignals.test_strategy_validation
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
<<<<<<< HEAD
        self.stock_data = fetch_stock_data('SQM', 
=======
        self.stock_data = fetch_stock_data('OPFI', 
>>>>>>> origin/main
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

    def _print_strategy_performance(self, strategy_name, strategy_final, buy_hold_final, 
                                initial_capital, strategy_data=None):
        """
        Print a simplified performance comparison between strategy and buy & hold
        
        Args:
            strategy_name: Name of the strategy
            strategy_final: Final portfolio value using strategy
            buy_hold_final: Final portfolio value using buy & hold
            initial_capital: Starting capital amount
            strategy_data: Additional performance metrics dictionary (optional)
        """
        strategy_return_pct = ((strategy_final / initial_capital) - 1) * 100
        buy_hold_return_pct = ((buy_hold_final / initial_capital) - 1) * 100
        outperformance = strategy_return_pct - buy_hold_return_pct
        
        print(f"\n{'=' * 50}")
        print(f"{strategy_name} Strategy Performance")
        print(f"{'=' * 50}")
        print(f"Initial capital: ${initial_capital:.2f}")
        print(f"Strategy: ${strategy_final:.2f} ({strategy_return_pct:.2f}%)")
        print(f"Buy & Hold: ${buy_hold_final:.2f} ({buy_hold_return_pct:.2f}%)")
        print(f"Outperformance: {outperformance:.2f}% ({'BEAT MARKET' if outperformance > 0 else 'UNDERPERFORMED'})")
        
        if strategy_data:
            print(f"\nAdditional Metrics:")
            for key, value in strategy_data.items():
                print(f"  {key}: {value}")
        print(f"{'=' * 50}")

    def test_backtest_mean_reversion(self):
        """Test backtesting of mean reversion strategy"""
        # Generate signals
        signals_data = self.mean_reversion.detect_signals()
        
        # Run backtest with strategy type
        backtest_results = self._run_backtest(signals_data, strategy_type='mean_reversion')
        
        # Calculate buy & hold performance
        initial_capital = 10000
        first_price = signals_data['close'].iloc[0]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        
        # Create buy & hold equity curve
        buy_hold_equity = signals_data['close'].apply(
            lambda price: (shares_bought * price) + remaining_cash
        )
        
        # Calculate final values
        strategy_final = backtest_results['portfolio_value'].iloc[-1]
        buy_hold_final = buy_hold_equity.iloc[-1]
        
        # Print comparison results
        self._print_strategy_performance("Mean Reversion", strategy_final, buy_hold_final, initial_capital)
        
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
        self._visualize_backtest_comparison(backtest_results, signals_data, initial_capital)
<<<<<<< HEAD

    def test_backtest_momentum(self):
        """Test backtesting of momentum strategies"""
        initial_capital = 10000
        
        # Test MA Crossover
        ma_signals = self.ma_crossover.detect_signals()
        ma_results = self._run_backtest(ma_signals, strategy_type='ma_crossover')
        
        # Calculate buy & hold for MA Crossover
        first_price = ma_signals['close'].iloc[0]
        shares_bought = initial_capital // first_price
        remaining_cash = initial_capital - (shares_bought * first_price)
        
        ma_buy_hold = ma_signals['close'].apply(
            lambda price: (shares_bought * price) + remaining_cash
        )
        
        ma_strategy_final = ma_results['portfolio_value'].iloc[-1]
        ma_buy_hold_final = ma_buy_hold.iloc[-1]
        
        # Print MA Crossover comparison
        self._print_strategy_performance("MA Crossover", ma_strategy_final, ma_buy_hold_final, initial_capital)
        
        # Add buy & hold to results for CSV
        ma_results['buy_hold_value'] = ma_buy_hold.values
        ma_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ma_crossover_backtest_results.csv'
        )
        ma_results.to_csv(ma_output_path, index=False)
        
        # Visualize MA Crossover
        self._visualize_backtest_comparison(ma_results, ma_signals, initial_capital, 'ma_crossover_vs_baseline.png', 'MA Crossover')
        
        # Test Volatility Breakout if data is available
        try:
            vb_signals = self.volatility_breakout.detect_signals()
            vb_results = self._run_backtest(vb_signals, strategy_type='volatility_breakout')
            
            # Calculate buy & hold for Volatility Breakout
            first_price = vb_signals['close'].iloc[0]
            shares_bought = initial_capital // first_price
            remaining_cash = initial_capital - (shares_bought * first_price)
            
            vb_buy_hold = vb_signals['close'].apply(
                lambda price: (shares_bought * price) + remaining_cash
            )
            
            vb_strategy_final = vb_results['portfolio_value'].iloc[-1]
            vb_buy_hold_final = vb_buy_hold.iloc[-1]
            
            # Print Volatility Breakout comparison
            self._print_strategy_performance("Volatility Breakout", vb_strategy_final, vb_buy_hold_final, initial_capital)
            
            # Add buy & hold to results for CSV
            vb_results['buy_hold_value'] = vb_buy_hold.values
            vb_output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'volatility_breakout_backtest_results.csv'
            )
            vb_results.to_csv(vb_output_path, index=False)
            
            # Visualize Volatility Breakout
            self._visualize_backtest_comparison(vb_results, vb_signals, initial_capital, 
                                            'volatility_breakout_vs_baseline.png', 'Volatility Breakout')
        except ValueError as e:
            print(f"\n{'=' * 50}")
            print("Volatility Breakout Strategy: FAILED")
            print(f"Error: {str(e)}")
            print(f"{'=' * 50}")



 
    def _visualize_backtest_comparison(self, backtest_results, signals_data, initial_capital, 
                                       save_name='mean_reversion_vs_baseline.png', strategy_name='Mean Reversion'):
=======
    
    
    
    def _visualize_backtest_comparison(self, backtest_results, signals_data, initial_capital):
>>>>>>> origin/main
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

<<<<<<< HEAD
    def _run_backtest(self, signals_df, initial_capital=10000, strategy_type='mean_reversion'):
        """
        Enhanced backtest helper method with position sizing and risk management
=======

    def create_diagnostic_plot(self, stock_symbol, period_start=None, period_end=None, save_path=None):
        """
        Create a detailed diagnostic plot for signal analysis.
        
        Parameters:
        -----------
        stock_symbol : str
            Stock symbol for plot title
        period_start : str, optional
            Start date to focus analysis ('YYYY-MM-DD')
        period_end : str, optional
            End date to focus analysis ('YYYY-MM-DD')
        save_path : str, optional
            Path to save the plot (default: f"{stock_symbol}_diagnostic.png")
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if not hasattr(self, 'processed_data'):
            self.detect_signals()
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        
        # Make a copy to avoid modifying the original
        df = self.processed_data.copy()
        
        # Filter by date range if provided
        if period_start is not None and 'date' in df.columns:
            df = df[df['date'] >= pd.to_datetime(period_start)]
        if period_end is not None and 'date' in df.columns:
            df = df[df['date'] <= pd.to_datetime(period_end)]
        
        if len(df) == 0:
            print(f"No data available for the specified period")
            return None
        
        # Create a new figure with 2 subplots (price and metrics)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Price and bands
        ax1.plot(df['date'], df[self.price_col], label='Close Price', color='blue', linewidth=1.5)
        ax1.plot(df['date'], df['open'], label='Open Price', color='cyan', linewidth=1, alpha=0.7)
        ax1.plot(df['date'], df['sma'], label=f'SMA ({self.window})', color='black', linewidth=1.5, alpha=0.7)
        ax1.plot(df['date'], df['upper_band'], label=f'Upper Band ({self.threshold}σ)', color='red', linewidth=1.5, linestyle='--')
        ax1.plot(df['date'], df['lower_band'], label=f'Lower Band ({self.threshold}σ)', color='green', linewidth=1.5, linestyle='--')
        
        # Add buy/sell signals
        buy_signals = df[df['signal'] == 'buy']
        ax1.scatter(buy_signals['date'], buy_signals[self.price_col], marker='^', color='green', s=100, label='Buy Signal')
        
        sell_signals = df[df['signal'] == 'sell']
        ax1.scatter(sell_signals['date'], sell_signals[self.price_col], marker='v', color='red', s=100, label='Sell Signal')
        
        # Add next-day execution points with open price
        if len(buy_signals) > 0:
            for idx in buy_signals.index:
                if idx + 1 < len(df):
                    next_day = df.iloc[idx + 1]
                    ax1.scatter(next_day['date'], next_day['open'], marker='o', edgecolor='green', facecolor='none', s=80)
                    ax1.plot([df.iloc[idx]['date'], next_day['date']], 
                            [df.iloc[idx][self.price_col], next_day['open']], 
                            'g--', alpha=0.6)
        
        if len(sell_signals) > 0:
            for idx in sell_signals.index:
                if idx + 1 < len(df):
                    next_day = df.iloc[idx + 1]
                    ax1.scatter(next_day['date'], next_day['open'], marker='o', edgecolor='red', facecolor='none', s=80)
                    ax1.plot([df.iloc[idx]['date'], next_day['date']], 
                            [df.iloc[idx][self.price_col], next_day['open']], 
                            'r--', alpha=0.6)
        
        # Highlight areas where price is near bands but not crossing
        near_lower = df[df['near_lower_band'] == True]
        for idx in near_lower.index:
            ax1.axvspan(df.iloc[idx]['date'] - pd.Timedelta(days=0.5), 
                        df.iloc[idx]['date'] + pd.Timedelta(days=0.5), 
                        alpha=0.2, color='green')
        
        near_upper = df[df['near_upper_band'] == True]
        for idx in near_upper.index:
            ax1.axvspan(df.iloc[idx]['date'] - pd.Timedelta(days=0.5), 
                        df.iloc[idx]['date'] + pd.Timedelta(days=0.5), 
                        alpha=0.2, color='red')
        
        # Set up the plot
        ax1.set_title(f'Mean Reversion Analysis for {stock_symbol} (Window: {self.window}, Threshold: {self.threshold})', fontsize=16)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Metrics
        # Calculate distance from price to bands (as percentage)
        df['price_to_lower_pct'] = ((df[self.price_col] - df['lower_band']) / df['lower_band'] * 100)
        df['price_to_upper_pct'] = ((df[self.price_col] - df['upper_band']) / df['upper_band'] * 100)
        
        # Plot percentage distance to bands
        ax2.plot(df['date'], df['price_to_lower_pct'], label='Distance to Lower Band (%)', color='green', alpha=0.7)
        ax2.plot(df['date'], df['price_to_upper_pct'], label='Distance to Upper Band (%)', color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add colored regions for signal zones
        ax2.axhspan(0, -100, alpha=0.1, color='green', label='Buy Zone')
        ax2.axhspan(0, 100, alpha=0.1, color='red', label='Sell Zone')
        
        ax2.set_ylabel('Distance to Bands (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Format x-axis dates to match the first plot
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = f"{stock_symbol}_diagnostic.png"
        
        plt.savefig(save_path)
        print(f"Diagnostic plot saved as {save_path}")
        
        return fig


    def _run_backtest(self, signals_df, initial_capital=10000):
        """
        Simple backtest helper method with realistic next-day execution
>>>>>>> origin/main
        
        Args:
            signals_df: DataFrame with signal column
            initial_capital: Starting capital amount
            strategy_type: Type of strategy being tested ('mean_reversion', 'ma_crossover', 'volatility_breakout')
                
        Returns:
            DataFrame with backtest results
        """
        df = signals_df.copy()
        
        # Initialize portfolio metrics
        df['position_pct'] = 0.0  # Position as percentage of portfolio (0-1)
        df['cash'] = initial_capital
        df['shares'] = 0
        df['portfolio_value'] = initial_capital
<<<<<<< HEAD
        df['equity_pct'] = 0.0  # Equity percentage of total portfolio
        df['risk_budget'] = 0.02  # Risk 2% per trade
        df['max_position_pct'] = 1  # Maximum position size (80% of portfolio)
        
        # Set up signal strength (for position sizing)
        df['signal_strength'] = 0.0
        
        # Consecutive signal counters
        consecutive_buys = 0
        consecutive_sells = 0
=======
        df['pending_signal'] = None  # Store signals for next-day execution
>>>>>>> origin/main
        
        for i in range(1, len(df)):
            # Copy previous values
            df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
            df.loc[df.index[i], 'shares'] = df.loc[df.index[i-1], 'shares']
            df.loc[df.index[i], 'position_pct'] = df.loc[df.index[i-1], 'position_pct']
            
<<<<<<< HEAD
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
                
                # For mean reversion, use gradual signal strength
                if strategy_type == 'mean_reversion':
                    # Signal strength increases with consecutive buys (capped at 1.0)
                    df.loc[df.index[i], 'signal_strength'] = min(1.0, consecutive_buys * 0.25)
                else:
                    # For momentum strategies, use binary signal strength
                    df.loc[df.index[i], 'signal_strength'] = 1.0
                    
            elif signal == 'sell':
                consecutive_buys = 0
                consecutive_sells += 1
                
                # For mean reversion, use gradual signal strength
                if strategy_type == 'mean_reversion':
                    # Signal strength increases with consecutive sells (capped at 1.0)
                    df.loc[df.index[i], 'signal_strength'] = min(1.0, consecutive_sells * 0.25)
                else:
                    # For momentum strategies, use binary signal strength
                    df.loc[df.index[i], 'signal_strength'] = 1.0
            else:
                # Hold signals decrease strength only for mean reversion
                if strategy_type == 'mean_reversion':
                    df.loc[df.index[i], 'signal_strength'] = max(0.0, df.loc[df.index[i-1], 'signal_strength'] - 0.1)
                else:
                    df.loc[df.index[i], 'signal_strength'] = 0.0
                    
            # Execute trading logic with position sizing
            if signal == 'buy':
                # Scale the target position based on signal strength and available cash
                max_position_pct = df.loc[df.index[i], 'max_position_pct']
                
                if strategy_type == 'mean_reversion':
                    # For mean reversion, scale gradually with signal strength
                    target_position_pct = position_pct + (0.7 * df.loc[df.index[i], 'signal_strength'])
                else:
                    # For momentum strategies, use full position sizing immediately
                    target_position_pct = max_position_pct
                    
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
                            
                            if new_portfolio_value > 0:
                                df.loc[df.index[i], 'position_pct'] = new_shares_value / new_portfolio_value
                            else:
                                df.loc[df.index[i], 'position_pct'] = 0.0
                                
                            df.loc[df.index[i], 'portfolio_value'] = new_portfolio_value
                
            elif signal == 'sell':
                # Scale the position reduction based on signal strength
                if strategy_type == 'mean_reversion':
                    # For mean reversion, scale gradually with signal strength
                    target_position_pct = position_pct * (1 - (0.5 * df.loc[df.index[i], 'signal_strength']))
                else:
                    # For momentum strategies, exit completely
                    target_position_pct = 0.0
                    
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

    def test_strategy_validation(self):
        """Test and validate all strategies across multiple tickers, saving results to CSV"""
        from src.data.fetcher import fetch_stock_data
        from datetime import datetime, timedelta
        import pandas as pd
        import os
        
        print("\n" + "=" * 50)
        print("STRATEGY VALIDATION TEST")
        print("=" * 50)
        
        # List of tickers to test (from your bot)
        tickers = ['SBLK', 'BTC-USD', 'NKE', 'SPY', 'KO', 'CRV-USD']
        
        # Strategy types and parameters to validate
        strategies = [
            {
                'name': 'Mean Reversion',
                'type': 'mean_reversion',
                'class': MeanReversionSignal,
                'params': self.config.get('mean_reversion', {})  # Use config values
            },
            {
                'name': 'MA Crossover',
                'type': 'ma_crossover', 
                'class': MACrossoverSignal,
                'params': self.config.get('momentum', {})  # Use config values
            },
            {
                'name': 'Volatility Breakout',
                'type': 'volatility_breakout',
                'class': VolatilityBreakoutSignal,
                'params': self.config.get('volatility', {})  # Use config values
            }
        ]
        
        # Prepare results dataframe
        results = []
        
        # End date for backtest (today)
        end_date = datetime.today()
        # Start date (1 year ago for better sample size)
        start_date = end_date - timedelta(days=365)
        
        # Test each ticker with each strategy
        for ticker in tickers:
            print(f"\nTesting {ticker}...")
            
            try:
                # Fetch historical data for this ticker
                data = fetch_stock_data(
                    ticker, 
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if data is None or len(data) < 50:
                    print(f"  ⚠️ Insufficient data for {ticker}")
                    for strategy in strategies:
                        results.append({
                            'ticker': ticker,
                            'strategy': strategy['name'],
                            'signal': 'INSUFFICIENT DATA',
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d'),
                            'data_points': 0,
                            'buy_signals': 0,
                            'sell_signals': 0,
                            'hold_signals': 0,
                            'backtest_return': None,
                            'buy_hold_return': None,
                            'outperformance': None
                        })
                    continue
                    
                # Calculate buy & hold baseline performance
                initial_capital = 10000
                first_price = data['close'].iloc[0]
                last_price = data['close'].iloc[-1]
                buy_hold_return_pct = ((last_price / first_price) - 1) * 100
                
                # Test each strategy on this ticker
                for strategy in strategies:
                    strategy_name = strategy['name']
                    strategy_type = strategy['type']
                    strategy_class = strategy['class']
                    
                    print(f"  Testing {strategy_name}...")
                    
                    try:
                        # Initialize strategy with data
                        strategy_instance = strategy_class(data, config=self.config)
                        
                        # Get signals
                        signals_data = strategy_instance.detect_signals()
                        
                        # Count signal types
                        buy_count = (signals_data['signal'] == 'buy').sum()
                        sell_count = (signals_data['signal'] == 'sell').sum()
                        hold_count = (signals_data['signal'] == 'hold').sum()
                        
                        # Get latest signal
                        latest_signal = signals_data['signal'].iloc[-1] if not signals_data.empty else 'UNKNOWN'
                        
                        # Run backtest
                        backtest_results = self._run_backtest(signals_data, strategy_type=strategy['type'])
                        
                        # Calculate backtest performance
                        if not backtest_results.empty:
                            # Calculate final portfolio values
                            strategy_final = backtest_results['portfolio_value'].iloc[-1]
                            strategy_return_pct = ((strategy_final / initial_capital) - 1) * 100
                            outperformance = strategy_return_pct - buy_hold_return_pct
                        else:
                            strategy_return_pct = 0
                            outperformance = -buy_hold_return_pct
                        
                        # Calculate win rate based on signals that resulted in profitable moves
                        signal_performance = []
                        for i in range(1, len(signals_data)):
                            if signals_data['signal'].iloc[i-1] == 'buy':
                                # For buy signals, measure forward returns
                                entry_price = signals_data['close'].iloc[i-1]
                                exit_price = signals_data['close'].iloc[i]
                                pct_change = (exit_price / entry_price - 1) * 100
                                signal_performance.append({
                                    'signal': 'buy',
                                    'return': pct_change,
                                    'profitable': pct_change > 0
                                })
                            elif signals_data['signal'].iloc[i-1] == 'sell':
                                # For sell signals, measure inverse of forward returns
                                entry_price = signals_data['close'].iloc[i-1]
                                exit_price = signals_data['close'].iloc[i]
                                pct_change = (entry_price / exit_price - 1) * 100
                                signal_performance.append({
                                    'signal': 'sell',
                                    'return': pct_change,
                                    'profitable': pct_change > 0
                                })
                        
                        # Calculate signal performance metrics
                        signal_df = pd.DataFrame(signal_performance) if signal_performance else pd.DataFrame()
                        if not signal_df.empty and len(signal_df) > 0:
                            win_rate = (signal_df['profitable'].sum() / len(signal_df)) * 100
                            avg_return = signal_df['return'].mean()
                        else:
                            win_rate = None
                            avg_return = None
                        
                        # Add results to list
                        results.append({
                            'ticker': ticker,
                            'strategy': strategy_name,
                            'latest_signal': latest_signal.upper(),
                            'data_points': len(signals_data),
                            'buy_signals': buy_count,
                            'sell_signals': sell_count,
                            'hold_signals': hold_count,
                            'win_rate': win_rate,
                            'avg_signal_return': avg_return,
                            'backtest_return': strategy_return_pct,
                            'buy_hold_return': buy_hold_return_pct,
                            'outperformance': outperformance,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d')
                        })
                        
                        # Save individual backtest results
                        backtest_output_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            f'backtest_{ticker}_{strategy_type}.csv'
                        )
                        if not backtest_results.empty:
                            backtest_results.to_csv(backtest_output_path, index=False)
                        
                    except ValueError as e:
                        print(f"    ⚠️ Strategy error: {e}")
                        results.append({
                            'ticker': ticker,
                            'strategy': strategy_name,
                            'latest_signal': 'ERROR',
                            'data_points': len(data) if data is not None else 0,
                            'buy_signals': 0,
                            'sell_signals': 0,
                            'hold_signals': 0,
                            'win_rate': None,
                            'avg_signal_return': None,
                            'backtest_return': None,
                            'buy_hold_return': buy_hold_return_pct,
                            'outperformance': None,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d')
                        })
                    except Exception as e:
                        print(f"    ❌ Unexpected error: {e}")
                        import traceback
                        traceback.print_exc()
            
            except Exception as e:
                print(f"  ❌ Error testing {ticker}: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert results to DataFrame 
        results_df = pd.DataFrame(results)
        
        # Format the DataFrame for better readability
        for col in ['win_rate', 'avg_signal_return', 'backtest_return', 'buy_hold_return', 'outperformance']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                )
        
        # Save results
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'strategy_validation_results.csv'
        )
        results_df.to_csv(csv_path, index=False)
        print(f"\nStrategy validation results saved to: {csv_path}")
        
        # Print summary by strategy
        print("\nSTRATEGY VALIDATION SUMMARY:")
        print("-" * 30)
        
        for strategy_name in [s['name'] for s in strategies]:
            strategy_results = results_df[results_df['strategy'] == strategy_name]
            outperformance_values = []
            
            for idx, row in strategy_results.iterrows():
                try:
                    if row['outperformance'] != "N/A":
                        value = float(row['outperformance'].strip('%'))
                        outperformance_values.append(value)
                except:
                    pass
            
            if outperformance_values:
                avg_outperformance = sum(outperformance_values) / len(outperformance_values)
                win_count = sum(1 for val in outperformance_values if val > 0)
                win_pct = (win_count / len(outperformance_values)) * 100 if outperformance_values else 0
            else:
                avg_outperformance = 0
                win_count = 0
                win_pct = 0
            
            print(f"{strategy_name}:")
            print(f"  Avg outperformance: {avg_outperformance:.2f}%")
            print(f"  Beat buy & hold: {win_count}/{len(outperformance_values)} tickers ({win_pct:.1f}%)")
        
        print("=" * 50)






=======
            # Get pending signal from previous day
            pending_signal = df.loc[df.index[i-1], 'pending_signal']
            position = df.loc[df.index[i], 'position']
            
            # Execute trades based on PREVIOUS day's signal at TODAY's open price
            if pending_signal == 'buy' and position == 0:
                # Buy with 95% of available cash at today's OPEN price
                open_price = df.loc[df.index[i], 'open']
                cash_to_use = df.loc[df.index[i], 'cash'] * 0.95
                shares_to_buy = cash_to_use // open_price
                cost = shares_to_buy * open_price
                
                df.loc[df.index[i], 'shares'] = shares_to_buy
                df.loc[df.index[i], 'cash'] -= cost
                df.loc[df.index[i], 'position'] = 1
                
            elif pending_signal == 'sell' and position == 1:
                # Sell all shares at today's OPEN price
                open_price = df.loc[df.index[i], 'open']
                sale_value = df.loc[df.index[i], 'shares'] * open_price
                
                df.loc[df.index[i], 'cash'] += sale_value
                df.loc[df.index[i], 'shares'] = 0
                df.loc[df.index[i], 'position'] = 0
            
            # Store today's signal for execution tomorrow
            df.loc[df.index[i], 'pending_signal'] = df.loc[df.index[i], 'signal']
            
            # Calculate portfolio value at today's close (for performance tracking)
            close_price = df.loc[df.index[i], 'close']
            shares_value = df.loc[df.index[i], 'shares'] * close_price
            df.loc[df.index[i], 'portfolio_value'] = df.loc[df.index[i], 'cash'] + shares_value
        
        return df

# Move this function outside the class to make it importable
def analyze_specific_stock(stock_symbol, start_date='2024-01-01', end_date=None, window=50, threshold=1.5):
    """
    Analyze a specific stock that might have signal generation issues.
    
    Parameters:
    -----------
    stock_symbol : str
        Stock symbol to analyze
    start_date : str
        Start date for analysis ('YYYY-MM-DD')
    end_date : str, optional
        End date for analysis ('YYYY-MM-DD')
    window : int
        Moving average window size
    threshold : float
        Standard deviation threshold for bands
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Analyzing {stock_symbol} from {start_date} to {end_date}...")
    
    # Fetch stock data
    data = fetch_stock_data(stock_symbol, start_date=start_date, end_date=end_date)
    
    if data is None or len(data) < window+10:
        print(f"Not enough data for {stock_symbol} to perform analysis")
        return
    
    # Create mean reversion signal object with specified parameters
    mean_rev = MeanReversionSignal(data, window=window, threshold=threshold)
    
    # Generate signals
    signals_data = mean_rev.detect_signals()
    
    # Print summary stats
    buy_count = (signals_data['signal'] == 'buy').sum()
    sell_count = (signals_data['signal'] == 'sell').sum()
    hold_count = (signals_data['signal'] == 'hold').sum()
    
    print(f"\nMean Reversion Signal Stats for {stock_symbol}:")
    print(f"Window: {window}, Threshold: {threshold}")
    print(f"Data points: {len(signals_data)}")
    print(f"Buy signals: {buy_count} ({buy_count/len(signals_data)*100:.1f}%)")
    print(f"Sell signals: {sell_count} ({sell_count/len(signals_data)*100:.1f}%)")
    print(f"Hold signals: {hold_count} ({hold_count/len(signals_data)*100:.1f}%)")
    
    # Export data to CSV for detailed analysis
    csv_path = f"{stock_symbol}_signal_data.csv"
    signals_data.to_csv(csv_path, index=False)
    print(f"Signal data exported to {csv_path}")
    
    # Analyze periods without signals
    last_signal_date = None
    longest_gap = 0
    current_gap = 0
    gap_start = None
    long_gaps = []
    
    for i, row in signals_data.iterrows():
        if row['signal'] != 'hold':
            if current_gap > 20:  # More than a month of trading days
                long_gaps.append((gap_start, row['date'], current_gap))
            current_gap = 0
            last_signal_date = row['date']
        else:
            if last_signal_date is None:
                gap_start = row['date']
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
            if current_gap == 20:  # Just hit a month without signals
                gap_start = signals_data.iloc[i-19]['date']
    
    if long_gaps:
        print(f"\nPeriods with no signals for over 20 trading days:")
        for start, end, days in long_gaps:
            print(f"  {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days} days)")
    
    print(f"Longest period without signals: {longest_gap} trading days")
    
    return signals_data
>>>>>>> origin/main

# This should be outside the class as well
if __name__ == '__main__':
    # Example usage - add problematic stocks here
    analyze_specific_stock('NVDA', window=50, threshold=1.5)
    analyze_specific_stock('SPY', window=50, threshold=1.5)


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
