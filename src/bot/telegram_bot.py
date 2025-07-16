from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class TelegramBot:
    def __init__(self, token: str):
        self.updater = Updater(token, use_context=True)
        self.dispatcher = self.updater.dispatcher

    def start(self, update: Update, context: CallbackContext):
        update.message.reply_text('Hello! I am your market signals bot. Use /help to see available commands.')

    def help_command(self, update: Update, context: CallbackContext):
        update.message.reply_text('Available commands:\n/start - Start the bot\n/help - Show this help message')

    def run(self):
        self.dispatcher.add_handler(CommandHandler("start", self.start))
        self.dispatcher.add_handler(CommandHandler("help", self.help_command))
        self.updater.start_polling()
        self.updater.idle()