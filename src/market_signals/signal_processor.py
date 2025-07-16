from typing import Dict, Any, Optional, List

class SignalProcessor:
    """
    Process trading signals from different strategies.
    Follows clean architecture principles with dependency inversion.
    This processor maintains separate signals from each strategy without combining them.
    """
    def __init__(self, signal_generators=None):
        """
        Initialize the signal processor with optional signal generators.
        
        Parameters:
        -----------
        signal_generators : dict, optional
            Dictionary of signal generator objects, keyed by signal type
        """
        self.signal_generators = signal_generators or {}
        self.signals = {}
        self.processed_results = {}
    
    def register_signal_generator(self, signal_type: str, generator: Any) -> None:
        """
        Register a signal generator for a specific signal type.
        
        Parameters:
        -----------
        signal_type : str
            The type of signal (e.g., 'mean_reversion', 'momentum')
        generator : object
            The signal generator object with a detect_signals method
        """
        self.signal_generators[signal_type] = generator
    
    def process_signals(self, market_data=None, **kwargs) -> Dict[str, Any]:
        """
        Process signals from either market data or direct signal input.
        Keeps signals separate from each strategy without combining.
        
        Parameters:
        -----------
        market_data : pandas.DataFrame, optional
            Market data to generate signals from
        **kwargs : dict
            Direct signals of different types ('mean_reversion_signal', 'momentum_signal', etc.)
            
        Returns:
        --------
        dict
            Dictionary containing processed signals from each strategy
        """
        # Clear previous processed results
        self.processed_results = {}
        self.signals = {}
        
        # Process market data if provided
        if market_data is not None:
            self._process_from_market_data(market_data)
        
        # Process direct signals if provided
        self._process_from_direct_signals(kwargs)
        
        # Return all signals without combining
        return self.signals.copy()
    
    def _process_from_market_data(self, market_data) -> None:
        """Process signals from market data using registered generators."""
        for signal_type, generator in self.signal_generators.items():
            if hasattr(generator, 'detect_signals'):
                signals_data = generator.detect_signals()
                if not signals_data.empty:
                    self.signals[signal_type] = signals_data.iloc[-1]['signal']
                    self.processed_results[f"{signal_type}_data"] = signals_data
            elif hasattr(generator, 'generate_signals'):
                # Handle generators that might have a different method name
                signals = generator.generate_signals()
                if signals:
                    self.signals[signal_type] = signals[-1]  # Get the most recent signal
                    self.processed_results[f"{signal_type}_data"] = signals
    
    def _process_from_direct_signals(self, signals_dict: Dict[str, str]) -> None:
        """Process directly provided signals."""
        for key, value in signals_dict.items():
            if key.endswith('_signal'):
                signal_type = key.replace('_signal', '')
                self.signals[signal_type] = value
    
    def get_signal_data(self, signal_type: str) -> Any:
        """
        Get the full signal data for a specific signal type.
        
        Parameters:
        -----------
        signal_type : str
            The type of signal to retrieve data for
        
        Returns:
        --------
        object
            The signal data, or None if not available
        """
        key = f"{signal_type}_data"
        return self.processed_results.get(key)
    
    def get_latest_signals(self) -> Dict[str, str]:
        """
        Get the latest signals for all registered signal types.
        
        Returns:
        --------
        dict
            Dictionary with latest signals
        """
        return self.signals.copy()
    
    def add_signal_type(self, signal_type: str) -> None:
        """
        Prepare the processor for a new signal type.
        
        Parameters:
        -----------
        signal_type : str
            The type of signal to add
        """
        if signal_type not in self.signals:
            self.signals[signal_type] = 'hold'  # Initialize with neutral signal
    
    def reset(self) -> None:
        """Reset the processor's state."""
        self.signals = {}
        self.processed_results = {}

# Usage example
if __name__ == "__main__":
    # This is just an example for documentation
    processor = SignalProcessor()
    
    # Register signal generators
    from src.market_signals.mean_reversion import MeanReversionSignal
    
    # Example data (would be real data in actual usage)
    import pandas as pd
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'close': [100 + i * 0.1 for i in range(100)]
    })
    
    # Create signal generators
    mean_rev = MeanReversionSignal(sample_data)
    
    # Register with processor
    processor.register_signal_generator('mean_reversion', mean_rev)
    
    # Process signals
    signals = processor.process_signals(market_data=sample_data)
    print(signals)  # Will show only individual signals without combining
    
    # Or process direct signals
    direct_signals = processor.process_signals(mean_reversion_signal='buy')
    print(direct_signals)  # Shows the direct signal without combining