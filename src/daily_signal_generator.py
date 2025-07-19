import os
import logging
import yaml
from datetime import datetime
import argparse
import asyncio

from src.bot.user_preferences import UserPreferencesManager
from src.market_signals.signal_service import SignalService
from src.bot.notification_manager import NotificationManager

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

async def send_daily_notifications(force=False, test_user=None, ignore_time=False):
    """
    Generate signals and send notifications to users
    
    Parameters:
    -----------
    force : bool
        If True, send notifications regardless of notification time
    """
    # Load configuration
    config = load_config()
    api_key = config.get('api_key')
    
    if not api_key:
        logger.error("No API key found in configuration")
        return
    
    # Initialize components
    user_prefs = UserPreferencesManager()
    signal_service = SignalService(user_prefs)
    notification_manager = NotificationManager(api_key)
    
    # Get all users
    users = user_prefs.get_all_users()
    logger.info(f"Found {len(users)} users")
    
    # Current time for filtering notifications
    current_time = datetime.now().strftime('%H:%M')


    # When running with --force, send to all users regardless of time
    # When running with --ignore-time, also ignore time constraints
    if force or ignore_time:
        logger.info(f"Running with force={force}, ignore_time={ignore_time}. Will send notifications regardless of time.")
        # Process will continue for all users
    else:
        logger.info(f"Current time: {current_time}. Will only send to users with matching notification time.")
        # Process will only send to users with matching notification_time
    
    # Stats for logging
    total_notifications_sent = 0
    users_notified = 0
    
    for user_id in users:
        try:
            # Get user preferences
            preferences = user_prefs.get_user_preferences(user_id)
            notification_time = preferences.get('notification_time', '20:30')
            
            # Skip if it's not time for this user's notification and not forcing
            if not force and not ignore_time and notification_time != current_time:
                logger.debug(f"Skipping user {user_id} (notification time: {notification_time}, current time: {current_time})")
                continue

            # If ignore_time is True, continue processing for this user
            if ignore_time:
                logger.info(f"Ignoring time constraint for user {user_id}")
            
            # Generate signals for this user
            logger.info(f"Generating signals for user {user_id}")
            user_signals = signal_service.generate_signals_for_user(user_id)
            
            # Filter to only include active signals (buy/sell)
            active_signals = signal_service.filter_active_signals(user_signals)
            
            if active_signals:
                # User has active signals
                users_notified += 1
                logger.info(f"Found {len(active_signals)} stocks with active signals for user {user_id}")
                
                # Send daily summary - ADD AWAIT HERE
                await notification_manager.send_daily_summary(user_id, active_signals)
                total_notifications_sent += 1
                
                # Send individual notifications for each active signal
                for stock, signals in active_signals.items():
                    for signal_type, value in signals.items():
                        # Only send individual notifications for buy/sell signals
                        if value in ["buy", "sell"]:
                            # Get additional details for this stock/signal
                            signal_details = {}
                            
                            # For mean reversion, include price and bands information
                            # Update the signal detail collection in send_daily_notifications

                            # Inside the for loop for active signals, when collecting signal details:
                            if signal_type == "mean_reversion":
                                try:
                                    # Fetch latest stock data
                                    from src.data.fetcher import fetch_stock_data
                                    data = fetch_stock_data(stock)
                                    if data is not None and not data.empty:
                                        # Get user parameters
                                        mr_params = preferences.get("signal_params", {}).get("mean_reversion", {})
                                        window = mr_params.get("window", 50)
                                        threshold = mr_params.get("threshold", 1.5)
                                        
                                        # Create mean reversion object with user parameters
                                        from src.market_signals.mean_reversion import MeanReversionSignal
                                        mean_rev = MeanReversionSignal(data, window=window, threshold=threshold)
                                        signal_info = mean_rev.get_latest_signal_formatted()
                                        
                                        signal_details = {
                                            'signal': signal_info['signal'],
                                            'emoji': signal_info['emoji'],
                                            'formatted_text': signal_info['formatted_text']
                                        }                                   
                                        # Add reliability metric
                                        try:
                                            from src.backtesting.signal_reliability import SignalReliabilityService
                                            reliability_service = SignalReliabilityService()
                                            signal_details['reliability'] = reliability_service.get_signal_reliability(
                                                ticker=stock,
                                                strategy_type='mean_reversion',
                                                strategy_params={'window': window, 'threshold': threshold}
                                            )
                                        except Exception as e:
                                            logger.error(f"Error getting reliability metrics for {stock}: {e}")
                                except Exception as e:
                                    logger.error(f"Error getting additional details for {stock}: {e}")
                            
                            # For MA Crossover strategy
                            elif signal_type == "ma_crossover":
                                try:
                                    # Fetch latest stock data
                                    from src.data.fetcher import fetch_stock_data
                                    data = fetch_stock_data(stock)
                                    if data is not None and not data.empty:
                                        # Get user parameters
                                        ma_params = preferences.get("signal_params", {}).get("ma_crossover", {})
                                        short_window = ma_params.get("short_window", 20)
                                        long_window = ma_params.get("long_window", 50)
                                        
                                        # Create MA Crossover object with user parameters
                                        from src.market_signals.momentum import MACrossoverSignal
                                        ma_signal = MACrossoverSignal(data, short_window=short_window, long_window=long_window)
                                        signal_info = ma_signal.get_latest_signal_formatted()
                                        
                                        # Extract details
                                        signal_details = signal_info.get('details', {})
                                        
                                        # Add reliability metrics
                                        try:
                                            from src.backtesting.signal_reliability import SignalReliabilityService
                                            reliability_service = SignalReliabilityService()
                                            signal_details['reliability'] = reliability_service.get_signal_reliability(
                                                ticker=stock,
                                                strategy_type='ma_crossover',
                                                strategy_params={'short_window': short_window, 'long_window': long_window}
                                            )
                                        except Exception as e:
                                            logger.error(f"Error getting reliability metrics for {stock}: {e}")
                                except Exception as e:
                                    logger.error(f"Error getting additional details for {stock}: {e}")
                            
                            # For Volatility Breakout strategy
                            elif signal_type == "volatility_breakout":
                                try:
                                    # Fetch latest stock data
                                    from src.data.fetcher import fetch_stock_data
                                    data = fetch_stock_data(stock)
                                    if data is not None and not data.empty:
                                        # Get user parameters
                                        vb_params = preferences.get("signal_params", {}).get("volatility_breakout", {})
                                        atr_window = vb_params.get("atr_window", 14)
                                        atr_multiplier = vb_params.get("atr_multiplier", 1.5)
                                        breakout_window = vb_params.get("breakout_window", 20)
                                        
                                        # Create Volatility Breakout object with user parameters
                                        from src.market_signals.momentum import VolatilityBreakoutSignal
                                        vb_signal = VolatilityBreakoutSignal(
                                            data, 
                                            atr_window=atr_window,
                                            atr_multiplier=atr_multiplier,
                                            breakout_window=breakout_window
                                        )
                                        signal_info = vb_signal.get_latest_signal_formatted()
                                        
                                        # Extract details
                                        signal_details = signal_info.get('details', {})
                                        
                                        # Add reliability metrics
                                        try:
                                            from src.backtesting.signal_reliability import SignalReliabilityService
                                            reliability_service = SignalReliabilityService()
                                            signal_details['reliability'] = reliability_service.get_signal_reliability(
                                                ticker=stock,
                                                strategy_type='volatility_breakout',
                                                strategy_params={
                                                    'atr_window': atr_window,
                                                    'atr_multiplier': atr_multiplier,
                                                    'breakout_window': breakout_window
                                                }
                                            )
                                        except Exception as e:
                                            logger.error(f"Error getting reliability metrics for {stock}: {e}")
                                except Exception as e:
                                    logger.error(f"Error getting additional details for {stock}: {e}")    
                                
                                
                                
                            
                            
                            # Send individual notification - ADD AWAIT HERE
                            await notification_manager.notify_signal(
                                user_id, stock, signal_type, value, signal_details
                            )
                            total_notifications_sent += 1

                            
            else:
                # No active signals for this user
                logger.info(f"No active signals for user {user_id}")
                
                # Send a "no signals" notification only if the user has tracked stocks
                tracked_stocks = preferences.get("tracked_stocks", [])
                if tracked_stocks:
                    # We'll skip this notification to align with the requirement
                    # to only notify on buy/sell signals
                    logger.info(f"User {user_id} is tracking {len(tracked_stocks)} stocks but has no active signals")
                
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}", exc_info=True)
    
    # Log summary
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