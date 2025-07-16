import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import CommandHandler, CallbackContext, ConversationHandler, MessageHandler, Filters, CallbackQueryHandler
import logging
import os

from .user_preferences import UserPreferencesManager
from src.data.fetcher import fetch_stock_data
from src.market_signals.mean_reversion import MeanReversionSignal

# Add these new states for the conversation handler
SET_PARAM_TYPE, SET_PARAM_VALUE = range(2)
# Make sure user_preferences directory is created
os.makedirs("user_preferences", exist_ok=True)

# Initialize user preferences manager
user_prefs = UserPreferencesManager(storage_path="user_preferences")

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext) -> None:
    """Send a welcome message when the command /start is issued"""
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.first_name
    
    # Get or initialize user preferences
    preferences = user_prefs.get_user_preferences(user_id)
    
    welcome_message = (
        f"Welcome to the Market Signals Bot, {user_name}!\n\n"
        f"This bot helps you track stock signals like mean reversion.\n\n"
        f"Use /help to see all available commands.\n"
        f"Use /add to start tracking stocks."
    )
    
    update.message.reply_text(welcome_message)

def add_stock(update: Update, context: CallbackContext) -> None:
    """Add a stock to the user's tracked list"""
    user_id = str(update.effective_user.id)
    
    # Check if a stock symbol was provided
    if not context.args:
        update.message.reply_text("Please provide a stock symbol. Example: /add AAPL")
        return
    
    stock_symbol = context.args[0].upper()
    
    # Verify the stock symbol exists by trying to fetch data
    try:
        data = fetch_stock_data(stock_symbol, start_date=None, end_date=None)
        if data is None or len(data) == 0:
            update.message.reply_text(f"Could not find data for stock symbol: {stock_symbol}. Please check the symbol and try again.")
            return
        
        logger.info(f"Successfully fetched data for {stock_symbol}")
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        update.message.reply_text(f"Error verifying stock symbol: {stock_symbol}. Please try again later.")
        return
    
    # Add the stock to the user's preferences
    if user_prefs.add_stock_for_user(user_id, stock_symbol):
        preferences = user_prefs.get_user_preferences(user_id)
        tracked_stocks = preferences.get("tracked_stocks", [])
        
        update.message.reply_text(
            f"Added {stock_symbol} to your tracked stocks.\n\n"
            f"You are now tracking {len(tracked_stocks)} stocks."
        )
    else:
        update.message.reply_text(f"Failed to add {stock_symbol} to your tracked stocks. Please try again.")

def remove_stock(update: Update, context: CallbackContext) -> None:
    """Remove a stock from the user's tracked list"""
    user_id = str(update.effective_user.id)
    
    # Check if a stock symbol was provided
    if not context.args:
        update.message.reply_text("Please provide a stock symbol. Example: /remove AAPL")
        return
    
    stock_symbol = context.args[0].upper()
    
    # Check if the user is tracking this stock
    preferences = user_prefs.get_user_preferences(user_id)
    if stock_symbol not in preferences.get("tracked_stocks", []):
        update.message.reply_text(f"You are not tracking {stock_symbol}.")
        return
    
    # Remove the stock from the user's preferences
    if user_prefs.remove_stock_for_user(user_id, stock_symbol):
        preferences = user_prefs.get_user_preferences(user_id)
        tracked_stocks = preferences.get("tracked_stocks", [])
        
        update.message.reply_text(
            f"Removed {stock_symbol} from your tracked stocks.\n\n"
            f"You are now tracking {len(tracked_stocks)} stocks."
        )
    else:
        update.message.reply_text(f"Failed to remove {stock_symbol} from your tracked stocks. Please try again.")

def list_stocks(update: Update, context: CallbackContext) -> None:
    """List all stocks the user is tracking"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    tracked_stocks = preferences.get("tracked_stocks", [])
    
    if not tracked_stocks:
        update.message.reply_text(
            "You are not tracking any stocks yet.\n\n"
            "Use /add SYMBOL to start tracking stocks."
        )
        return
    
    stocks_list = "\n".join(tracked_stocks)
    message = f"You are tracking {len(tracked_stocks)} stocks:\n\n{stocks_list}"
    
    update.message.reply_text(message)

def set_notification_time(update: Update, context: CallbackContext) -> None:
    """Set the time for daily notifications"""
    user_id = str(update.effective_user.id)
    
    # Check if a time was provided
    if not context.args:
        update.message.reply_text("Please provide a time in 24-hour format. Example: /time 08:30")
        return
    
    time_str = context.args[0]
    
    # Validate time format
    import re
    if not re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", time_str):
        update.message.reply_text("Invalid time format. Please use HH:MM in 24-hour format. Example: 08:30")
        return
    
    # Update notification time in user preferences
    if user_prefs.update_notification_time(user_id, time_str):
        update.message.reply_text(f"Your daily notification time has been set to {time_str} UTC.")
    else:
        update.message.reply_text("Failed to update notification time. Please try again.")

def settings(update: Update, context: CallbackContext) -> None:
    """Show settings menu"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    signal_types = preferences.get("signal_types", [])
    
    keyboard = [
        [
            InlineKeyboardButton("Mean Reversion", callback_data="toggle_mean_reversion"),
            InlineKeyboardButton("✅" if "mean_reversion" in signal_types else "❌", callback_data="toggle_mean_reversion")
        ],
        # Add more signal types as they become available
        [
            InlineKeyboardButton("Done", callback_data="settings_done")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text("Select the signal types you want to track:", reply_markup=reply_markup)

def button_callback(update: Update, context: CallbackContext) -> None:
    """Handle button clicks from inline keyboards"""
    query = update.callback_query
    query.answer()
    
    user_id = str(query.from_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    signal_types = preferences.get("signal_types", [])
    
    if query.data == "settings_done":
        query.edit_message_text(text="Settings updated!")
        return
    
    elif query.data.startswith("toggle_"):
        signal_type = query.data.replace("toggle_", "")
        
        # Toggle signal type
        if signal_type in signal_types:
            signal_types.remove(signal_type)
        else:
            signal_types.append(signal_type)
        
        # Update preferences
        user_prefs.update_signal_types(user_id, signal_types)
        
        # Update UI
        keyboard = [
            [
                InlineKeyboardButton("Mean Reversion", callback_data="toggle_mean_reversion"),
                InlineKeyboardButton("✅" if "mean_reversion" in signal_types else "❌", callback_data="toggle_mean_reversion")
            ],
            # Add more signal types as they become available
            [
                InlineKeyboardButton("Done", callback_data="settings_done")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        query.edit_message_text(text="Select the signal types you want to track:", reply_markup=reply_markup)

def get_signals(update: Update, context: CallbackContext) -> None:
    """Get current signals for the user's tracked stocks"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    tracked_stocks = preferences.get("tracked_stocks", [])
    signal_types = preferences.get("signal_types", ["mean_reversion"])
    
    if not tracked_stocks:
        update.message.reply_text(
            "You are not tracking any stocks yet.\n\n"
            "Use /add SYMBOL to start tracking stocks."
        )
        return
    
    # Get signal parameters for display
    signal_params = preferences.get("signal_params", {})
    mr_params = signal_params.get("mean_reversion", {"window": 50, "threshold": 1.5})
    mr_window = mr_params.get("window", 50)
    mr_threshold = mr_params.get("threshold", 1.5)
    
    update.message.reply_text(
        f"Fetching signals for {len(tracked_stocks)} stocks using your parameters:\n"
        f"- Mean Reversion: Window={mr_window}, Threshold={mr_threshold}\n\n"
        "This may take a moment..."
    )
    
    # Rest of the function remains the same
    signals_results = []
    
    for stock in tracked_stocks:
        try:
            # Fetch the latest stock data
            data = fetch_stock_data(stock)
            if data is None or len(data) == 0:
                signals_results.append(f"❓ {stock}: Could not fetch data")
                continue
            
            signals = []
            
            # Process mean reversion signals if enabled
            if "mean_reversion" in signal_types:
                try:
                    # Get user's custom parameters for mean reversion
                    mr_params = preferences.get("signal_params", {}).get("mean_reversion", {})
                    window = mr_params.get("window", 50)
                    threshold = mr_params.get("threshold", 1.5)
                    
                    # Initialize mean reversion signal with user's parameters
                    mean_rev = MeanReversionSignal(data, window=window, threshold=threshold)
                    signal_info = mean_rev.get_latest_signal_formatted()
                    
                    # Add the formatted signal information
                    signals.append(f"Mean Reversion: {signal_info['formatted_text']}")
                    
                    # Log additional details for debugging
                    logger.debug(f"{stock} mean reversion: {signal_info['signal']} " +
                            f"(Price: {signal_info['details']['price']:.2f}, " +
                            f"SMA: {signal_info['details']['sma']:.2f}, " +
                            f"Bands: {signal_info['details']['lower_band']:.2f}-{signal_info['details']['upper_band']:.2f})")
                except Exception as e:
                    logger.error(f"Error processing mean reversion signal for {stock}: {e}")
                    signals.append("Mean Reversion: ❌ ERROR")
            
            # Format and add the signals to the results
            if signals:
                signal_text = "\n  - ".join(signals)
                signals_results.append(f"{stock}:\n  - {signal_text}")
            else:
                signals_results.append(f"{stock}: No signals generated")
                
        except Exception as e:
            logger.error(f"Error getting signals for {stock}: {e}")
            signals_results.append(f"❌ {stock}: Error getting signals")
                
    # Send the results
    if signals_results:
        message = "Current signals for your tracked stocks:\n\n" + "\n\n".join(signals_results)
    else:
        message = "No signals were generated for your tracked stocks."
    
    update.message.reply_text(message)

def error_handler(update: Update, context: CallbackContext) -> None:
    """Log errors and send a message to the user"""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    try:
        # Send message to the user
        if update and update.effective_message:
            update.effective_message.reply_text("Sorry, something went wrong. Please try again later.")
    except:
        pass

def param_settings(update: Update, context: CallbackContext) -> None:
    """Show signal parameter settings"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    signal_types = preferences.get("signal_types", [])
    
    # Create message with current parameter settings
    message = "*Signal Parameter Settings*\n\n"
    
    if "mean_reversion" in signal_types:
        mr_params = preferences.get("signal_params", {}).get("mean_reversion", {})
        window = mr_params.get("window", 50)
        threshold = mr_params.get("threshold", 1.5)
        message += f"*Mean Reversion Parameters:*\n"
        message += f"- Window: `{window}`\n"
        message += f"- Threshold: `{threshold}`\n\n"
    
    # Add other signal types as they become available
    
    message += "Use the following commands to update parameters:\n"
    message += "`/setparam mean_reversion window VALUE`\n"
    message += "`/setparam mean_reversion threshold VALUE`\n"
    
    update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

def set_param(update: Update, context: CallbackContext) -> None:
    """Set a parameter value for a signal type"""
    user_id = str(update.effective_user.id)
    
    # Check if enough arguments are provided
    if len(context.args) < 3:
        update.message.reply_text(
            "Please provide a signal type, parameter name, and value.\n"
            "Example: `/setparam mean_reversion window 30`", 
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    signal_type = context.args[0].lower()
    param_name = context.args[1].lower()
    param_value_str = context.args[2]
    
    # Validate signal type
    valid_signal_types = ["mean_reversion"]  # Add more as they become available
    if signal_type not in valid_signal_types:
        update.message.reply_text(
            f"Invalid signal type: {signal_type}. Valid types are: {', '.join(valid_signal_types)}"
        )
        return
    
    # Validate parameter name for mean_reversion
    if signal_type == "mean_reversion":
        valid_params = ["window", "threshold"]
        if param_name not in valid_params:
            update.message.reply_text(
                f"Invalid parameter for mean_reversion: {param_name}. Valid parameters are: {', '.join(valid_params)}"
            )
            return
    
    # Convert and validate parameter value
    try:
        if param_name == "window":
            # Window should be an integer > 0
            param_value = int(param_value_str)
            if param_value <= 0:
                raise ValueError("Window must be greater than 0")
        elif param_name == "threshold":
            # Threshold should be a float > 0
            param_value = float(param_value_str)
            if param_value <= 0:
                raise ValueError("Threshold must be greater than 0")
        else:
            param_value = param_value_str
    except ValueError as e:
        update.message.reply_text(f"Invalid parameter value: {str(e)}")
        return
    
    # Update the parameter
    updated = user_prefs.update_signal_params(user_id, signal_type, {param_name: param_value})
    
    if updated:
        update.message.reply_text(
            f"Parameter updated: {signal_type} {param_name} = {param_value}"
        )
        
        # Show the updated settings
        param_settings(update, context)
    else:
        update.message.reply_text("Failed to update parameter. Please try again.")

# Add the following to the register_handlers function
def register_handlers(dispatcher) -> None:
    """Register all command handlers"""
    # Existing handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("add", add_stock))
    dispatcher.add_handler(CommandHandler("remove", remove_stock))
    dispatcher.add_handler(CommandHandler("list", list_stocks))
    dispatcher.add_handler(CommandHandler("signals", get_signals))
    dispatcher.add_handler(CommandHandler("time", set_notification_time))
    dispatcher.add_handler(CommandHandler("settings", settings))
    
    # New handlers for parameter settings
    dispatcher.add_handler(CommandHandler("params", param_settings))
    dispatcher.add_handler(CommandHandler("setparam", set_param))
    
    dispatcher.add_handler(CallbackQueryHandler(button_callback))
    
    # Add error handler
    dispatcher.add_error_handler(error_handler)

# Also update the help_command function to include the new commands
def help_command(update: Update, context: CallbackContext) -> None:
    """Send a help message when the command /help is issued"""
    help_text = (
        "Available commands:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/add SYMBOL - Add a stock to track (e.g., /add AAPL)\n"
        "/remove SYMBOL - Remove a tracked stock\n"
        "/list - List all stocks you're tracking\n"
        "/signals - Show current signals for your tracked stocks\n"
        "/time HH:MM - Set your daily notification time (UTC)\n"
        "/settings - Manage your signal preferences\n"
        "/params - View your signal parameter settings\n"
        "/setparam TYPE NAME VALUE - Set a signal parameter (e.g., /setparam mean_reversion window 30)"
    )
    
    update.message.reply_text(help_text)