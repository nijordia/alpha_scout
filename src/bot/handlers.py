from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, Filters, CallbackContext

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Welcome to the Market Signals Bot!')

def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Available commands:\n/start - Start the bot\n/help - Get help')

def notify_signal(update: Update, context: CallbackContext, signal: str) -> None:
    update.message.reply_text(f'New market signal detected: {signal}')

def handle_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text
    update.message.reply_text(f'You said: {text}')

def register_handlers(dispatcher) -> None:
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))