import unittest
from backtesting.strategy_evaluator import BacktestEngine
from src.backtesting.performance_metrics import PerformanceMetrics

class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.backtest_engine = BacktestEngine()

    def test_run_backtest(self):
        result = self.backtest_engine.run_backtest(strategy='mean_reversion', data='test_data')
        self.assertIsNotNone(result)
        self.assertIn('returns', result)
        self.assertIn('drawdown', result)

class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self.performance_metrics = PerformanceMetrics()

    def test_calculate_returns(self):
        metrics = self.performance_metrics.calculate_returns(initial_investment=1000, final_value=1200)
        self.assertEqual(metrics, 0.2)

    def test_calculate_drawdown(self):
        drawdown = self.performance_metrics.calculate_drawdown(peak=1200, trough=800)
        self.assertEqual(drawdown, 0.3333333333333333)

if __name__ == '__main__':
    unittest.main()