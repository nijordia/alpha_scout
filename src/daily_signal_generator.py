import os
import logging
import yaml
from datetime import datetime
import argparse
import asyncio

from src.bot.user_preferences import UserPreferencesManager
from src.market_signals.signal_service import SignalService
from src.bot.notification_manager import NotificationManager
from src.market_signals.signal_service import SignalService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_signals.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
user_prefs = UserPreferencesManager() 
signal_service = SignalService(user_prefs)
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 
        'config.yml'
    )
    
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}



async def send_daily_notifications(force=False, test_user=None, ignore_time=False, token=None):
    """
    Generate signals and send notifications to users
    """
    # Load configuration
    config = load_config()
    api_key = token or os.environ.get('TELEGRAM_API_KEY') or config.get('api_key')
    if not api_key:
        logger.error("No API key found in configuration or environment")
        return

    # Initialize components
    user_prefs = UserPreferencesManager()
    signal_service = SignalService(user_prefs)
    notification_manager = NotificationManager(api_key)

    users = user_prefs.get_all_users()
    logger.info(f"Found {len(users)} users")
    current_time = datetime.now().strftime('%H:%M')

    total_notifications_sent = 0
    users_notified = 0

    for user_id in users:
        try:
            preferences = user_prefs.get_user_preferences(user_id)
            notification_time = preferences.get('notification_time', '20:30')
            if not force and not ignore_time and notification_time != current_time:
                logger.debug(f"Skipping user {user_id} (notification time: {notification_time}, current time: {current_time})")
                continue

            logger.info(f"Generating signals for user {user_id}")
            # Use the centralized SignalService to get all signals and metrics
            user_results = signal_service.get_signal_metrics_for_user(user_id)

            # Filter to only include active signals (buy/sell)
            active_signals = {}
            for stock_result in user_results:
                stock = stock_result["stock"]
                for signal in stock_result["signals"]:
                    if signal.get("raw_signal") in ["buy", "sell"]:
                        if stock not in active_signals:
                            active_signals[stock] = {}
                        active_signals[stock][signal["type"]] = signal["raw_signal"]

            if active_signals:
                users_notified += 1
                logger.info(f"Found {len(active_signals)} stocks with active signals for user {user_id}")

                # Send daily summary
                await notification_manager.send_daily_summary(user_id, active_signals)
                total_notifications_sent += 1

                # Send individual notifications for each active signal, using the formatted metrics
                for stock_result in user_results:
                    stock = stock_result["stock"]
                    for signal in stock_result["signals"]:
                        if signal.get("raw_signal") in ["buy", "sell"]:
                            # Send notification with formatted text (includes metrics)
                            await notification_manager.notify_signal(
                                user_id,
                                stock,
                                signal["type"],
                                signal["raw_signal"],
                                {"formatted_text": signal["text"], "metrics": signal["metrics"]}
                            )
                            total_notifications_sent += 1
            else:
                logger.info(f"No active signals for user {user_id}")
                tracked_stocks = preferences.get("tracked_stocks", [])
                if tracked_stocks:
                    logger.info(f"User {user_id} is tracking {len(tracked_stocks)} stocks but has no active signals")

        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}", exc_info=True)

    logger.info(f"Daily signal processing complete. Notified {users_notified} users with {total_notifications_sent} notifications.")
    return users_notified > 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Signals Daily Generator")
    parser.add_argument(
        "--force", action="store_true",
        help="Force sending notifications regardless of notification time"
    )
    
    args = parser.parse_args()
    
    if args.force:
        # Override the time check for testing
        logger.info("Forcing notification generation regardless of time")
        
    logger.info("Starting daily signal generation")
    
    # Use asyncio.run to run the async function
    success = asyncio.run(send_daily_notifications(force=args.force))
    
    if success:
        logger.info("Daily signal generation complete - signals were sent")
    else:
        logger.info("Daily signal generation complete - no signals were sent")