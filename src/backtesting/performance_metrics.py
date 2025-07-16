def calculate_returns(prices):
    return (prices[-1] - prices[0]) / prices[0] * 100

def calculate_drawdown(prices):
    peak = prices[0]
    max_drawdown = 0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = [r - risk_free_rate for r in returns]
    return sum(excess_returns) / (len(excess_returns) ** 0.5)

def performance_summary(prices, returns):
    return {
        "total_return": calculate_returns(prices),
        "max_drawdown": calculate_drawdown(prices),
        "sharpe_ratio": calculate_sharpe_ratio(returns)
    }