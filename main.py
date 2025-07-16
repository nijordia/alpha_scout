import logging
import argparse
import os
import sys
import yaml

# Make sure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.bot.telegram_bot import TelegramBot
from src.daily_signal_generator import send_daily_notifications

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'config', 
        'config.yml'
    )
    
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}

def main():
    """
    Main entry point for the application
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Market Signals Telegram Bot")
    parser.add_argument(
        "--daily-run", action="store_true",
        help="Run daily signal generation and exit"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force sending notifications regardless of notification time"
    )
    parser.add_argument(
        "--token", type=str,
        help="Telegram bot token (overrides config file)"
    )
    parser.add_argument(
        "--ignore-time", action="store_true",
        help="Ignore notification time constraints for all users"
    )
    parser.add_argument(
        "--test-user", type=str,
        help="Test notification for a specific user ID"
    )
    args = parser.parse_args()
    
    # Configure logging
    log_file = os.path.join(os.path.dirname(__file__), "market_signals.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log startup
    logger.info("Starting Market Signals Telegram Bot...")
    
    # Load configuration
    config = load_config()
    
    # Get token from args or config
    token = args.token or config.get('api_key')
    
    if not token:
        logger.error("No Telegram bot token provided. Please add it to config.yml or use --token")
        exit(1)
    
    if args.daily_run:
        # Just run the daily signal generation and exit
        logger.info("Running daily signal generation")
        if args.force:
            logger.info("Forcing notifications regardless of time")
        if args.test_user:
            logger.info(f"Testing for specific user: {args.test_user}")
        send_daily_notifications(force=args.force, test_user=args.test_user, ignore_time=args.ignore_time)
        logger.info("Daily signal generation complete")
    else:
        # Start the bot for interactive mode
        try:
            bot = TelegramBot(token=token)
            bot.run()
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            exit(1)

if __name__ == "__main__":
    main()