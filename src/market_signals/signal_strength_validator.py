"""
Utility to validate and optimize signal strength application across different strategy types.
"""
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from src.market_signals.momentum import MACrossoverSignal, VolatilityBreakoutSignal
from src.market_signals.mean_reversion import MeanReversionSignal

class SignalStrengthValidator:
    """
    Validates signal strength behavior across different trading strategies.
    """
    
    def __init__(self):
        self.backtest_files = []
        self.results = {}
        
    def load_backtest_data(self):
        """Load all backtest files in the project root directory."""
        self.backtest_files = glob.glob(os.path.join(os.getcwd(), 'backtest_*.csv'))
        print(f"Found {len(self.backtest_files)} backtest files.")
        return self.backtest_files
    
    def analyze_signal_strength_effectiveness(self):
        """Analyze how signal strength affects returns for each strategy type."""
        if not self.backtest_files:
            self.load_backtest_data()
        
        # Categorize by strategy type
        strategy_results = {
            "mean_reversion": [],
            "ma_crossover": [],
            "volatility_breakout": []
        }
        
        for file_path in self.backtest_files:
            file_name = os.path.basename(file_path)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            try:
                # Extract strategy type from filename
                for strategy in strategy_results.keys():
                    if strategy in file_name:
                        df = pd.read_csv(file_path)
                        
                        # Analyze correlation between signal strength and returns
                        if 'signal_strength' in df.columns and len(df) > 5:
                            # Calculate price change for next day
                            df['next_day_return'] = df['close'].pct_change(1).shift(-1) * 100
                            
                            # Group by signal strength rounded to nearest 0.1
                            df['signal_strength_bin'] = np.round(df['signal_strength'], 1)
                            
                            # Calculate metrics by signal strength
                            strength_metrics = []
                            for strength, group in df.groupby('signal_strength_bin'):
                                if not group.empty and not group['next_day_return'].isna().all():
                                    # Average return for each signal strength level
                                    avg_return = group['next_day_return'].mean()
                                    # Win rate for each signal strength level
                                    win_rate = (group['next_day_return'] > 0).mean() * 100
                                    
                                    strength_metrics.append({
                                        'signal_strength': strength,
                                        'avg_return': avg_return,
                                        'win_rate': win_rate,
                                        'count': len(group)
                                    })
                            
                            # Calculate correlation between signal strength and next day return
                            correlation = df['signal_strength'].corr(df['next_day_return'])
                            
                            # Check if signal strength is being used (non-zero values)
                            strength_used = (df['signal_strength'] > 0).sum() > 0
                            
                            ticker = file_name.split('_')[1]
                            
                            strategy_results[strategy].append({
                                'ticker': ticker,
                                'correlation': correlation,
                                'strength_metrics': strength_metrics,
                                'strength_used': strength_used
                            })
                        break
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        self.results = strategy_results
        return strategy_results
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis."""
        if not self.results:
            self.analyze_signal_strength_effectiveness()
        
        recommendations = {}
        
        for strategy, results in self.results.items():
            if not results:
                recommendations[strategy] = "No data available for analysis."
                continue
                
            # Calculate average correlation across all tickers
            correlations = [r['correlation'] for r in results if not pd.isna(r['correlation'])]
            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                
                if strategy == "mean_reversion":
                    if avg_correlation > 0.1:
                        recommendations[strategy] = "KEEP: Signal strength has positive correlation with returns."
                    else:
                        recommendations[strategy] = "ADJUST: Consider tweaking signal strength calculation for better correlation."
                else:  # Momentum strategies
                    if abs(avg_correlation) < 0.05:
                        recommendations[strategy] = "REMOVE: Signal strength shows minimal correlation with returns."
                    else:
                        recommendations[strategy] = "SIMPLIFY: Use binary signal strength (0 or 1) instead of gradual scale."
            else:
                recommendations[strategy] = "INCONCLUSIVE: Insufficient correlation data."
                
        return recommendations
    
    def visualize_results(self, save_path=None):
        """Visualize effectiveness of signal strength for each strategy."""
        if not self.results:
            self.analyze_signal_strength_effectiveness()
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        for i, (strategy, results) in enumerate(self.results.items()):
            ax = axes[i]
            ax.set_title(f"Signal Strength Effectiveness: {strategy.replace('_', ' ').title()}")
            ax.set_xlabel("Signal Strength")
            ax.set_ylabel("Average Return (%)")
            
            # Plot each ticker's signal strength vs return
            for result in results:
                ticker = result['ticker']
                strength_metrics = result.get('strength_metrics', [])
                
                if strength_metrics:
                    strengths = [m['signal_strength'] for m in strength_metrics]
                    returns = [m['avg_return'] for m in strength_metrics]
                    
                    ax.plot(strengths, returns, 'o-', label=f"{ticker} (corr={result['correlation']:.2f})")
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Results visualization saved to {save_path}")
        else:
            plt.show()
            
    def print_summary(self):
        """Print a summary of the analysis."""
        if not self.results:
            self.analyze_signal_strength_effectiveness()
            
        recommendations = self.generate_recommendations()
        
        print("\n" + "=" * 80)
        print("SIGNAL STRENGTH EFFECTIVENESS SUMMARY")
        print("=" * 80)
        
        for strategy, results in self.results.items():
            print(f"\n{strategy.replace('_', ' ').title()} Strategy:")
            print("-" * 50)
            
            if not results:
                print("  No data available for analysis.")
                continue
                
            correlations = [r['correlation'] for r in results if not pd.isna(r['correlation'])]
            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                print(f"  Average correlation with returns: {avg_correlation:.4f}")
                
                for result in results:
                    ticker = result['ticker']
                    corr = result['correlation']
                    strength_used = result['strength_used']
                    print(f"  {ticker}: correlation={corr:.4f}, signal strength used: {strength_used}")
            
            print(f"\n  Recommendation: {recommendations[strategy]}")
        
        print("\n" + "=" * 80)
        print("Implementation Plan:")
        
        for strategy, recommendation in recommendations.items():
            if "REMOVE" in recommendation:
                print(f"  - Modify {strategy} to use constant signal strength (1.0) instead of variable")
            elif "SIMPLIFY" in recommendation:
                print(f"  - Simplify {strategy} to use binary signal strength (0.0 or 1.0)")
            elif "ADJUST" in recommendation:
                print(f"  - Adjust {strategy} signal strength calculation to improve correlation")
            elif "KEEP" in recommendation:
                print(f"  - Keep current {strategy} signal strength implementation")
                
        print("=" * 80)

if __name__ == "__main__":
    validator = SignalStrengthValidator()
    validator.print_summary()
    validator.visualize_results(save_path="signal_strength_analysis.png")