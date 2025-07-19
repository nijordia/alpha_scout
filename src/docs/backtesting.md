# Backtesting Methodology

This document describes the backtesting methodology used to evaluate the reliability of trading signals in the Market Signals Bot.

## Overview

The backtesting system evaluates how each trading strategy would have performed historically for a specific stock. This provides users with context about how reliable a particular signal might be.

## Key Metrics

### Win Rate

The percentage of signals that resulted in a profitable trade within a specified holding period.

```
Win Rate = (Number of Profitable Signals / Total Signals) × 100%
```

A higher win rate indicates a more reliable signal, but should be considered alongside the average return per signal.

### Average Return Per Signal

The average percentage return when following signals within a specified holding period.

```
Average Return = Sum of Returns for All Signals / Number of Signals
```

This metric shows the typical magnitude of profit or loss when following this signal.

### Market Outperformance Rate

The percentage of signals that outperformed a simple buy-and-hold strategy over the same period.

```
Outperformance Rate = (Number of Signals Beating Buy & Hold / Total Signals) × 100%
```

A rate above 50% indicates the strategy adds timing value compared to simply holding the stock.

## Timeframes

Each metric is calculated over multiple timeframes to show how the strategy performs in different scenarios:

- Short-term (14 days)
- Medium-term (30 days)
- Long-term (90 days)

## Implementation Details

### Backtesting Process

1. Historical data is fetched for the specified ticker
2. Strategy signals are generated using the same logic as the live system
3. Each signal is evaluated by calculating the return over various holding periods
4. Results are compared against a buy-and-hold baseline
5. Metrics are calculated and cached to improve performance

### Limitations

- **Past Performance**: Historical performance doesn't guarantee future results
- **Transaction Costs**: The backtest doesn't account for commissions, slippage, or spread
- **Liquidity**: The backtest assumes perfect execution at close prices
- **Market Conditions**: A strategy might perform differently in changing market regimes

## Interpreting Results

When a signal is presented, the reliability metrics provide context about how similar signals have performed historically:

```
AAPL: 🟢 BUY (Mean Reversion)
Win Rate: 65.3% (30d) | Avg: +2.4% | Beats market: 58.7% of trades
```

This shows that historically:
- 65.3% of Mean Reversion buy signals for AAPL were profitable over a 30-day period
- The average return per signal was +2.4%
- The strategy outperformed simply buying and holding AAPL in 58.7% of trades

## Configurations

The backtesting system can be configured in `config.yml`:

```yaml
# Backtesting configuration
backtest_days: 365         # Days of historical data to use
reliability_periods:       # Holding periods to evaluate
  - 14                     # Short-term (days)
  - 30                     # Medium-term (days)
  - 90                     # Long-term (days)
```