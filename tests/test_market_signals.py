# For running this test use: python -m unittest tests.test_market_signals
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
        self.stock_data = fetch_stock_data('OPFI', 
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
        df['pending_signal'] = None  # Store signals for next-day execution
        
        for i in range(1, len(df)):
            # Copy previous values
            df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
            df.loc[df.index[i], 'shares'] = df.loc[df.index[i-1], 'shares']
            df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
            
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
