class MomentumSignal:
    def __init__(self, data, window=14):
        self.data = data
        self.window = window

    def calculate_momentum(self):
        if len(self.data) < self.window:
            raise ValueError("Not enough data to calculate momentum.")
        
        momentum = []
        for i in range(self.window, len(self.data)):
            momentum_value = self.data[i] - self.data[i - self.window]
            momentum.append(momentum_value)
        
        return momentum

    def generate_signals(self):
        momentum_values = self.calculate_momentum()
        signals = []

        for value in momentum_values:
            if value > 0:
                signals.append("buy")
            elif value < 0:
                signals.append("sell")
            else:
                signals.append("hold")

        return signals