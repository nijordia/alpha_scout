class BacktestEngine:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        self.results = []

    def run_backtest(self):
        for index in range(len(self.data)):
            if self.strategy.should_enter(self.data[index]):
                self.enter_position(index)
            elif self.strategy.should_exit(self.data[index]):
                self.exit_position(index)

    def enter_position(self, index):
        # Logic for entering a position
        pass

    def exit_position(self, index):
        # Logic for exiting a position
        pass

    def calculate_results(self):
        # Logic for calculating backtest results
        pass

    def get_results(self):
        return self.results