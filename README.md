# Market Signals Telegram Bot

This project is a Telegram bot that notifies users about market signals, including mean reverting and momentum signals. It is designed to fetch market data, process signals, and send notifications to users through Telegram.

## Project Structure

```
market-signals-telegram-bot
├── src
│   ├── main.py                  # Entry point of the application
│   ├── bot                      # Contains bot-related functionality
│   │   ├── __init__.py
│   │   ├── handlers.py          # Message handlers for the Telegram bot
│   │   ├── telegram_bot.py      # Manages the Telegram bot instance
│   │   └── notification_manager.py # Logic for sending notifications
│   ├── market_signals           # Contains market signal detection logic
│   │   ├── __init__.py
│   │   ├── mean_reversion.py     # Detects mean reverting signals
│   │   ├── momentum.py           # Detects momentum signals
│   │   └── signal_processor.py    # Processes generated signals
│   ├── data                     # Handles data fetching and processing
│   │   ├── __init__.py
│   │   ├── fetcher.py           # Fetches market data
│   │   └── data_processor.py     # Processes fetched data
│   └── backtesting              # Implements backtesting functionality
│       ├── __init__.py
│       ├── backtest_engine.py    # Backtesting engine for trading strategies
│       └── performance_metrics.py  # Calculates performance metrics
├── tests                        # Contains unit tests for the project
│   ├── __init__.py
│   ├── test_bot.py              # Unit tests for bot functionality
│   ├── test_market_signals.py    # Unit tests for market signals
│   ├── test_data.py             # Unit tests for data handling
│   └── test_backtesting.py       # Unit tests for backtesting
├── config                       # Configuration files
│   └── config.yml               # Configuration settings for the bot
├── requirements.txt             # Project dependencies
├── .gitignore                   # Files to ignore in version control
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd market-signals-telegram-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the bot by editing the `config/config.yml` file with your API keys and other settings.

## Usage

To run the bot, execute the following command:
```
python src/main.py
```

The bot will start and listen for market signals, sending notifications to users as configured.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.