import logging
import os
from datetime import datetime, time, timedelta
import asyncio
import pytz

# New imports for python-telegram-bot v20+
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, 
    MessageHandler, ContextTypes, filters
)

# Import handlers from handlers.py
from src.bot.handlers import (
    start, help_command, add_stock, remove_stock, list_stocks, 
    get_signals, set_notification_time, settings, button_callback,
    param_settings, set_param, metrics_explanation, nyse_close_command  # Add the new handler here
)

class TelegramBot:
    """Telegram bot for market signals"""
    
    def __init__(self, token):
        """Initialize the bot with the given token"""
        self.logger = logging.getLogger(__name__)
        self.token = token
        
        # Use ApplicationBuilder directly
        self.application = ApplicationBuilder().token(self.token).build()
        
        # Register command handlers from handlers.py
        self.application.add_handler(CommandHandler("start", start))
        self.application.add_handler(CommandHandler("help", help_command))
        self.application.add_handler(CommandHandler("add", add_stock))
        self.application.add_handler(CommandHandler("remove", remove_stock))
        self.application.add_handler(CommandHandler("list", list_stocks))
        self.application.add_handler(CommandHandler("signals", get_signals))
        self.application.add_handler(CommandHandler("signal", get_signals))  # Alias for /signals
        self.application.add_handler(CommandHandler("time", set_notification_time))
        self.application.add_handler(CommandHandler("settings", settings))
        self.application.add_handler(CommandHandler("params", param_settings))
        self.application.add_handler(CommandHandler("setparam", set_param))
        self.application.add_handler(CommandHandler("metrics", metrics_explanation))
        self.application.add_handler(CommandHandler("nyse_close", nyse_close_command))
        
        # Register message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Register callback query handler
        self.application.add_handler(CallbackQueryHandler(button_callback))
        
        self.logger.info("Bot initialized")
    
    def run(self):
        """Start the bot"""
        self.logger.info("Starting bot")
        # Run the bot until the user presses Ctrl-C
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        text = update.message.text
        await update.message.reply_text(f'You said: {text}')