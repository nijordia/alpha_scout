import logging
from bot.telegram_bot import TelegramBot
from data.fetcher import fetch_market_data
from market_signals.signal_processor import process_signals

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Market Signals Telegram Bot...")

    # Initialize the Telegram bot
    bot = TelegramBot()
    
    # Fetch market data
    market_data = fetch_market_data()
    
    # Process signals
    signals = process_signals(market_data)
    
    # Start the bot
    bot.start()

if __name__ == "__main__":
    main()