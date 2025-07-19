import re
import logging
import os
import asyncio

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CommandHandler, ContextTypes, ConversationHandler, 
    MessageHandler, filters, CallbackQueryHandler
)
from telegram.constants import ParseMode

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    
    await update.message.reply_text(welcome_message)

async def add_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a stock to the user's tracked list"""
    user_id = str(update.effective_user.id)
    
    # Check if a stock symbol was provided
    if not context.args:
        await update.message.reply_text("Please provide a stock symbol. Example: /add AAPL")
        return
    
    stock_symbol = context.args[0].upper()
    
    # Verify the stock symbol exists by trying to fetch data
    try:
        # Show that we're working on it
        await update.message.reply_text(f"Verifying stock symbol {stock_symbol}...")
        
        # Run fetch_stock_data in a thread to avoid blocking
        data = await asyncio.to_thread(
            fetch_stock_data, stock_symbol, None, None
        )
        
        if data is None or len(data) == 0:
            await update.message.reply_text(f"Could not find data for stock symbol: {stock_symbol}. Please check the symbol and try again.")
            return
        
        logger.info(f"Successfully fetched data for {stock_symbol}")
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        await update.message.reply_text(f"Error verifying stock symbol: {stock_symbol}. Please try again later.")
        return
    
    # Add the stock to the user's preferences
    if user_prefs.add_stock_for_user(user_id, stock_symbol):
        preferences = user_prefs.get_user_preferences(user_id)
        tracked_stocks = preferences.get("tracked_stocks", [])
        
        await update.message.reply_text(
            f"Added {stock_symbol} to your tracked stocks.\n\n"
            f"You are now tracking {len(tracked_stocks)} stocks."
        )
    else:
        await update.message.reply_text(f"Failed to add {stock_symbol} to your tracked stocks. Please try again.")

async def remove_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a stock from the user's tracked list"""
    user_id = str(update.effective_user.id)
    
    # Check if a stock symbol was provided
    if not context.args:
        await update.message.reply_text("Please provide a stock symbol. Example: /remove AAPL")
        return
    
    stock_symbol = context.args[0].upper()
    
    # Check if the user is tracking this stock
    preferences = user_prefs.get_user_preferences(user_id)
    if stock_symbol not in preferences.get("tracked_stocks", []):
        await update.message.reply_text(f"You are not tracking {stock_symbol}.")
        return
    
    # Remove the stock from the user's preferences
    if user_prefs.remove_stock_for_user(user_id, stock_symbol):
        preferences = user_prefs.get_user_preferences(user_id)
        tracked_stocks = preferences.get("tracked_stocks", [])
        
        await update.message.reply_text(
            f"Removed {stock_symbol} from your tracked stocks.\n\n"
            f"You are now tracking {len(tracked_stocks)} stocks."
        )
    else:
        await update.message.reply_text(f"Failed to remove {stock_symbol} from your tracked stocks. Please try again.")

async def list_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all stocks the user is tracking"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    tracked_stocks = preferences.get("tracked_stocks", [])
    
    if not tracked_stocks:
        await update.message.reply_text(
            "You are not tracking any stocks yet.\n\n"
            "Use /add SYMBOL to start tracking stocks."
        )
        return
    
    stocks_list = "\n".join(tracked_stocks)
    message = f"You are tracking {len(tracked_stocks)} stocks:\n\n{stocks_list}"
    
    await update.message.reply_text(message)

async def set_notification_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set the time for daily notifications"""
    user_id = str(update.effective_user.id)
    
    # Check if a time was provided
    if not context.args:
        await update.message.reply_text("Please provide a time in 24-hour format. Example: /time 08:30")
        return
    
    time_str = context.args[0]
    
    # Validate time format
    if not re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", time_str):
        await update.message.reply_text("Invalid time format. Please use HH:MM in 24-hour format. Example: 08:30")
        return
    
    # Update notification time in user preferences
    if user_prefs.update_notification_time(user_id, time_str):
        await update.message.reply_text(f"Your daily notification time has been set to {time_str} UTC.")
    else:
        await update.message.reply_text("Failed to update notification time. Please try again.")

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show settings menu"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    signal_types = preferences.get("signal_types", [])
    
    keyboard = [
        [
            InlineKeyboardButton("Mean Reversion", callback_data="toggle_mean_reversion"),
            InlineKeyboardButton("✅" if "mean_reversion" in signal_types else "❌", callback_data="toggle_mean_reversion")
        ],
        [
            InlineKeyboardButton("Moving Average Crossover", callback_data="toggle_ma_crossover"),
            InlineKeyboardButton("✅" if "ma_crossover" in signal_types else "❌", callback_data="toggle_ma_crossover")
        ],
        [
            InlineKeyboardButton("Volatility Breakout", callback_data="toggle_volatility_breakout"),
            InlineKeyboardButton("✅" if "volatility_breakout" in signal_types else "❌", callback_data="toggle_volatility_breakout")
        ],
        [
            InlineKeyboardButton("Done", callback_data="settings_done")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select the signal types you want to track:", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks from inline keyboards"""
    query = update.callback_query
    await query.answer()
    
    user_id = str(query.from_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    signal_types = preferences.get("signal_types", [])
    
    if query.data == "settings_done":
        await query.edit_message_text(text="Settings updated!")
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
            [
                InlineKeyboardButton("Moving Average Crossover", callback_data="toggle_ma_crossover"),
                InlineKeyboardButton("✅" if "ma_crossover" in signal_types else "❌", callback_data="toggle_ma_crossover")
            ],
            [
                InlineKeyboardButton("Volatility Breakout", callback_data="toggle_volatility_breakout"),
                InlineKeyboardButton("✅" if "volatility_breakout" in signal_types else "❌", callback_data="toggle_volatility_breakout")
            ],
            [
                InlineKeyboardButton("Done", callback_data="settings_done")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="Select the signal types you want to track:", reply_markup=reply_markup)

async def get_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get current signals for the user's tracked stocks"""
    user_id = str(update.effective_user.id)
    preferences = user_prefs.get_user_preferences(user_id)
    tracked_stocks = preferences.get("tracked_stocks", [])
    signal_types = preferences.get("signal_types", ["mean_reversion"])

    if not tracked_stocks:
        await update.message.reply_text(
            "You are not tracking any stocks yet.\n\n"
            "Use /add SYMBOL to start tracking stocks."
        )
        return

    # Get signal parameters for display
    signal_params = preferences.get("signal_params", {})

    # Prepare parameter display message
    params_text = ""
    if "mean_reversion" in signal_types:
        mr_params = signal_params.get("mean_reversion", {"window": 50, "threshold": 1.5})
        mr_window = mr_params.get("window", 50)
        mr_threshold = mr_params.get("threshold", 1.5)
        params_text += f"- Mean Reversion: Window={mr_window}, Threshold={mr_threshold}\n"

    if "ma_crossover" in signal_types:
        ma_params = signal_params.get("ma_crossover", {"short_window": 20, "long_window": 50})
        short_window = ma_params.get("short_window", 20)
        long_window = ma_params.get("long_window", 50)
        params_text += f"- Moving Average: Short={short_window}, Long={long_window}\n"

    if "volatility_breakout" in signal_types:
        vb_params = signal_params.get("volatility_breakout", {
            "atr_window": 14, "atr_multiplier": 1.5, "breakout_window": 20
        })
        atr_window = vb_params.get("atr_window", 14)
        atr_multiplier = vb_params.get("atr_multiplier", 1.5)
        breakout_window = vb_params.get("breakout_window", 20)
        params_text += f"- Volatility: ATR={atr_window}, Mult={atr_multiplier}, Window={breakout_window}\n"

    await update.message.reply_text(
        f"Fetching signals for {len(tracked_stocks)} stocks using your parameters:\n"
        f"{params_text}\n"
        "This may take a moment..."
    )

    # Process signals for each tracked stock
    signals_results = []

    # Import needed signal classes
    from src.market_signals.mean_reversion import MeanReversionSignal
    from src.market_signals.momentum import MACrossoverSignal, VolatilityBreakoutSignal

    # Import reliability service once
    from src.backtesting.signal_reliability import SignalReliabilityService
    reliability_service = SignalReliabilityService()

    for stock in tracked_stocks:
        try:
            # Fetch the latest stock data only once
            data = await asyncio.to_thread(fetch_stock_data, stock)
            if data is None or len(data) == 0:
                signals_results.append(f"❓ {stock}: Could not fetch data")
                continue

            signals = []

            # Helper to calculate buy & hold return for the same period
            def calc_bh_return(data, period):
                if len(data) < period:
                    return None
                start_price = data['close'].iloc[-period]
                end_price = data['close'].iloc[-1]
                return (end_price - start_price) / start_price * 100

            if "mean_reversion" in signal_types:
                try:
                    mr_params = preferences.get("signal_params", {}).get("mean_reversion", {})
                    window = mr_params.get("window", 50)
                    threshold = mr_params.get("threshold", 1.5)

                    mean_rev = MeanReversionSignal(data, window=window, threshold=threshold)
                    signal_info = mean_rev.get_latest_signal_formatted()

                    signal_text = f"Mean Reversion: {signal_info['formatted_text']}"

                    try:
                        reliability = reliability_service.get_signal_reliability(
                            ticker=stock,
                            strategy_type='mean_reversion',
                            strategy_params={'window': window, 'threshold': threshold}
                        )
                        # Use the same period as reliability metrics
                        period = reliability.get('period', 30)
                        bh_return = calc_bh_return(data, period)
                        avg_return = reliability.get('avg_return', 0)
                        if bh_return is not None:
                            vs_bh = avg_return - bh_return
                            vs_bh_text = f" | vs BH: {'+' if vs_bh > 0 else ''}{vs_bh:.2f}%"
                        else:
                            vs_bh_text = ""
                        reliability_text = (
                            f" | Win: {reliability['win_rate']}% | Avg: {'+' if avg_return > 0 else ''}{avg_return}%{vs_bh_text}"
                        )
                        signal_text += reliability_text

                    except ValueError as e:
                        logger.warning(f"Insufficient data for {stock} mean reversion reliability: {e}")
                        signal_text += " | Metrics: Insufficient historical data"
                    except Exception as e:
                        logger.error(f"Error calculating reliability for {stock} mean reversion: {e}")
                        signal_text += " | Metrics: Calculation error"

                    signals.append(signal_text)
                except Exception as e:
                    logger.error(f"Error processing mean reversion signal for {stock}: {e}")

            if "ma_crossover" in signal_types:
                try:
                    ma_params = preferences.get("signal_params", {}).get("ma_crossover", {})
                    short_window = ma_params.get("short_window", 20)
                    long_window = ma_params.get("long_window", 50)

                    ma_signal = MACrossoverSignal(data, short_window=short_window, long_window=long_window)
                    signal_info = ma_signal.get_latest_signal_formatted()

                    signal_text = f"MA Crossover: {signal_info['formatted_text']}"

                    try:
                        reliability = reliability_service.get_signal_reliability(
                            ticker=stock,
                            strategy_type='ma_crossover',
                            strategy_params={'short_window': short_window, 'long_window': long_window}
                        )
                        period = reliability.get('period', 30)
                        bh_return = calc_bh_return(data, period)
                        avg_return = reliability.get('avg_return', 0)
                        if bh_return is not None:
                            vs_bh = avg_return - bh_return
                            vs_bh_text = f" | vs BH: {'+' if vs_bh > 0 else ''}{vs_bh:.2f}%"
                        else:
                            vs_bh_text = ""
                        reliability_text = (
                            f" | Win: {reliability['win_rate']}% | Avg: {'+' if avg_return > 0 else ''}{avg_return}%{vs_bh_text}"
                        )
                        signal_text += reliability_text

                    except ValueError as e:
                        logger.warning(f"Insufficient data for {stock} MA crossover reliability: {e}")
                        signal_text += " | Metrics: Insufficient historical data"
                    except Exception as e:
                        logger.error(f"Error calculating reliability for {stock} MA crossover: {e}")
                        signal_text += " | Metrics: Calculation error"

                    signals.append(signal_text)
                except Exception as e:
                    logger.error(f"Error processing MA crossover signal for {stock}: {e}")

            if "volatility_breakout" in signal_types:
                try:
                    vb_params = preferences.get("signal_params", {}).get("volatility_breakout", {})
                    atr_window = vb_params.get("atr_window", 14)
                    atr_multiplier = vb_params.get("atr_multiplier", 1.5)
                    breakout_window = vb_params.get("breakout_window", 20)

                    vb_signal = VolatilityBreakoutSignal(
                        data,
                        atr_window=atr_window,
                        atr_multiplier=atr_multiplier,
                        breakout_window=breakout_window
                    )
                    signal_info = vb_signal.get_latest_signal_formatted()

                    signal_text = f"Volatility Breakout: {signal_info['formatted_text']}"

                    try:
                        reliability = reliability_service.get_signal_reliability(
                            ticker=stock,
                            strategy_type='volatility_breakout',
                            strategy_params={
                                'atr_window': atr_window,
                                'atr_multiplier': atr_multiplier,
                                'breakout_window': breakout_window
                            }
                        )
                        period = reliability.get('period', 30)
                        bh_return = calc_bh_return(data, period)
                        avg_return = reliability.get('avg_return', 0)
                        if bh_return is not None:
                            vs_bh = avg_return - bh_return
                            vs_bh_text = f" | vs BH: {'+' if vs_bh > 0 else ''}{vs_bh:.2f}%"
                        else:
                            vs_bh_text = ""
                        reliability_text = (
                            f" | Win: {reliability['win_rate']}% | Avg: {'+' if avg_return > 0 else ''}{avg_return}%{vs_bh_text}"
                        )
                        signal_text += reliability_text

                    except ValueError as e:
                        logger.warning(f"Insufficient data for {stock} volatility breakout reliability: {e}")
                        signal_text += " | Metrics: Insufficient historical data"
                    except Exception as e:
                        logger.error(f"Error calculating reliability for {stock} volatility breakout: {e}")
                        signal_text += " | Metrics: Calculation error"

                    signals.append(signal_text)
                except Exception as e:
                    logger.error(f"Error processing volatility breakout signal for {stock}: {e}")

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

    await update.message.reply_text(message)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and send a message to the user"""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    try:
        # Send message to the user
        if update and update.effective_message:
            await update.effective_message.reply_text("Sorry, something went wrong. Please try again later.")
    except:
        pass

async def param_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    
    # Add MA Crossover parameters
    if "ma_crossover" in signal_types:
        ma_params = preferences.get("signal_params", {}).get("ma_crossover", {})
        short_window = ma_params.get("short_window", 20)
        long_window = ma_params.get("long_window", 50)
        message += f"*Moving Average Crossover Parameters:*\n"
        message += f"- Short Window: `{short_window}`\n"
        message += f"- Long Window: `{long_window}`\n\n"
    
    # Add Volatility Breakout parameters
    if "volatility_breakout" in signal_types:
        vb_params = preferences.get("signal_params", {}).get("volatility_breakout", {})
        atr_window = vb_params.get("atr_window", 14)
        atr_multiplier = vb_params.get("atr_multiplier", 1.5)
        breakout_window = vb_params.get("breakout_window", 20)
        message += f"*Volatility Breakout Parameters:*\n"
        message += f"- ATR Window: `{atr_window}`\n"
        message += f"- ATR Multiplier: `{atr_multiplier}`\n"
        message += f"- Breakout Window: `{breakout_window}`\n\n"
    
    message += "*Update Parameters Commands:*\n"
    message += "For Mean Reversion:\n"
    message += "`/setparam mean_reversion window VALUE`\n"
    message += "`/setparam mean_reversion threshold VALUE`\n\n"
    
    message += "For Moving Average Crossover:\n"
    message += "`/setparam ma_crossover short_window VALUE`\n"
    message += "`/setparam ma_crossover long_window VALUE`\n\n"
    
    message += "For Volatility Breakout:\n"
    message += "`/setparam volatility_breakout atr_window VALUE`\n"
    message += "`/setparam volatility_breakout atr_multiplier VALUE`\n"
    message += "`/setparam volatility_breakout breakout_window VALUE`\n"
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def set_param(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set a parameter value for a signal type"""
    user_id = str(update.effective_user.id)
    
    # Check if enough arguments are provided
    if len(context.args) < 3:
        await update.message.reply_text(
            "Please provide a signal type, parameter name, and value.\n"
            "Example: `/setparam mean_reversion window 30`", 
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    signal_type = context.args[0].lower()
    param_name = context.args[1].lower()
    param_value_str = context.args[2]
    
    # Validate signal type
    valid_signal_types = ["mean_reversion", "ma_crossover", "volatility_breakout"]
    if signal_type not in valid_signal_types:
        await update.message.reply_text(
            f"Invalid signal type: {signal_type}. Valid types are: {', '.join(valid_signal_types)}"
        )
        return
    
    # Validate parameter name for each signal type
    valid_params = {
        "mean_reversion": ["window", "threshold"],
        "ma_crossover": ["short_window", "long_window"],
        "volatility_breakout": ["atr_window", "atr_multiplier", "breakout_window"]
    }
    
    if param_name not in valid_params.get(signal_type, []):
        await update.message.reply_text(
            f"Invalid parameter for {signal_type}: {param_name}. "
            f"Valid parameters are: {', '.join(valid_params.get(signal_type, []))}"
        )
        return
    
    # Convert and validate parameter value
    try:
        if param_name in ["window", "short_window", "long_window", "atr_window", "breakout_window"]:
            # These parameters should be integers > 0
            param_value = int(param_value_str)
            if param_value <= 0:
                raise ValueError(f"{param_name} must be greater than 0")
            
            # Additional validation for short_window < long_window
            if param_name == "short_window" and signal_type == "ma_crossover":
                ma_params = user_prefs.get_signal_params(user_id, "ma_crossover")
                long_window = ma_params.get("long_window", 50)
                if param_value >= long_window:
                    raise ValueError(f"short_window must be less than long_window ({long_window})")
            
            if param_name == "long_window" and signal_type == "ma_crossover":
                ma_params = user_prefs.get_signal_params(user_id, "ma_crossover")
                short_window = ma_params.get("short_window", 20)
                if param_value <= short_window:
                    raise ValueError(f"long_window must be greater than short_window ({short_window})")
                
        elif param_name in ["threshold", "atr_multiplier"]:
            # These parameters should be floats > 0
            param_value = float(param_value_str)
            if param_value <= 0:
                raise ValueError(f"{param_name} must be greater than 0")
        else:
            param_value = param_value_str
    except ValueError as e:
        await update.message.reply_text(f"Invalid parameter value: {str(e)}")
        return
    
    # Update the parameter
    updated = user_prefs.update_signal_params(user_id, signal_type, {param_name: param_value})
    
    if updated:
        await update.message.reply_text(
            f"Parameter updated: {signal_type} {param_name} = {param_value}"
        )
        
        # Show the updated settings
        await param_settings(update, context)
    else:
        await update.message.reply_text("Failed to update parameter. Please try again.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        "/settings - Manage your signal preferences (Mean Reversion, MA Crossover, Volatility Breakout)\n"
        "/params - View your signal parameter settings\n"
        "/setparam TYPE NAME VALUE - Set a signal parameter\n\n"
        "Signal types:\n"
        "- mean_reversion: Bollinger Band based mean reversion signals\n"
        "- ma_crossover: Moving Average crossover momentum signals\n"
        "- volatility_breakout: Volatility-based breakout signals"
    )
    
    await update.message.reply_text(help_text)


async def metrics_explanation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send an explanation of all test metrics and abbreviations."""
    explanation = (
        "*Metrics Explained* 📊\n\n"
        "- *Win Rate*: Percentage of signals that resulted in a profit.\n"
        "- *Avg Return*: Average percentage return per signal.\n"
        "- *Buy & Hold (BH) Return*: The return from simply holding the asset over the same period.\n"
        "- *Strategy Return*: Total return from following the strategy signals.\n"
        "- *Outperformance*: How much better the strategy performed compared to buy & hold.\n"
        "- *Max Drawdown*: Largest peak-to-trough decline during the period.\n"
        "- *Signal Count*: Number of signals generated (buy/sell).\n"
        "- *Period*: Time span covered by the backtest.\n\n"
        "_BH = Buy & Hold_\n"
    )
    await update.message.reply_text(explanation, parse_mode=ParseMode.MARKDOWN)




async def nyse_close_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set notification time to 30 minutes after NYSE close"""
    user_id = str(update.effective_user.id)
    
    # Get current date in Eastern Time
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # NYSE closes at 16:00 ET, add 30 minutes
    nyse_close_plus_30 = "16:30"
    
    # Update notification time
    if user_prefs.update_notification_time(user_id, nyse_close_plus_30):
        await update.message.reply_text(
            f"Your daily notification time has been set to 30 minutes after NYSE close ({nyse_close_plus_30} ET).\n\n"
            f"Note: This will be converted to your local timezone when notifications are sent."
        )
    else:
        await update.message.reply_text("Failed to update notification time. Please try again.")