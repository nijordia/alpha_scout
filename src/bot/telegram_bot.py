import logging
import yaml
import os
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters

from .handlers import register_handlers

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token=None):
        """
        Initialize the Telegram bot
        
        Parameters:
        -----------
        token : str, optional
            Telegram bot token (if not provided, will try to load from config)
        """
        # Load token from config if not provided
        if token is None:
            config = self._load_config()
            token = config.get('api_key')
            
        if not token:
            raise ValueError("No bot token provided")
            
        self.updater = Updater(token, use_context=True)
        self.dispatcher = self.updater.dispatcher

    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'config', 
            'config.yml'
        )
        
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    def run(self):
        """Register handlers and start the bot"""
        # Register command handlers
        register_handlers(self.dispatcher)
        
        # Start the Bot
        logger.info("Starting bot")
        self.updater.start_polling()
        
        # Run the bot until the user presses Ctrl-C
        self.updater.idle()