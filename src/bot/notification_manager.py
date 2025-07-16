from telegram import Bot, ParseMode
import logging

class NotificationManager:
    def __init__(self, bot_token):
        self.bot = Bot(token=bot_token)
        self.logger = logging.getLogger(__name__)

    def send_notification(self, chat_id, message):
        """
        Send a notification to a specific chat ID
        
        Parameters:
        -----------
        chat_id : str or int
            Chat ID to send the notification to
        message : str
            Message to send
        
        Returns:
        --------
        bool
            True if the notification was sent successfully, False otherwise
        """
        try:
            self.bot.send_message(
                chat_id=chat_id, 
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification to {chat_id}: {e}")
            return False

    def notify_signal(self, chat_id, stock, signal_type, signal_value, additional_details=None):
        """
        Send a signal notification
        
        Parameters:
        -----------
        chat_id : str or int
            Chat ID to send the notification to
        stock : str
            Stock symbol
        signal_type : str
            Type of signal (e.g., 'mean_reversion', 'momentum')
        signal_value : str
            Signal value (e.g., 'buy', 'sell', 'hold')
        additional_details : dict, optional
            Additional details to include in the notification
        
        Returns:
        --------
        bool
            True if the notification was sent successfully, False otherwise
        """
        # Format signal type for display
        signal_type_display = signal_type.replace("_", " ").title()
        
        # Choose emoji based on signal value
        emoji = "🟢" if signal_value == "buy" else "🔴" if signal_value == "sell" else "⚪"
        
        # Build the message
        message = f"*Market Signal Alert!*\n\n"
        message += f"*Stock:* {stock}\n"
        message += f"*Signal Type:* {signal_type_display}\n"
        message += f"*Signal:* {emoji} {signal_value.upper()}\n"
        
        # Add additional details if provided
        if additional_details:
            message += "\n*Details:*\n"
            for key, value in additional_details.items():
                key_display = key.replace("_", " ").title()
                message += f"- {key_display}: {value}\n"
        
        return self.send_notification(chat_id, message)

    def send_daily_summary(self, chat_id, signals_summary):
        """
        Send a daily summary of signals
        
        Parameters:
        -----------
        chat_id : str or int
            Chat ID to send the notification to
        signals_summary : dict
            Dictionary of stocks and their signals
        
        Returns:
        --------
        bool
            True if the notification was sent successfully, False otherwise
        """
        message = "*Daily Market Signals Summary*\n\n"
        
        if not signals_summary:
            message += "No signals to report today."
            return self.send_notification(chat_id, message)
        
        # Count signals by type
        buy_signals = []
        sell_signals = []
        
        for stock, signals in signals_summary.items():
            for signal_type, value in signals.items():
                if value == "buy":
                    buy_signals.append(f"{stock} ({signal_type})")
                elif value == "sell":
                    sell_signals.append(f"{stock} ({signal_type})")
        
        # Add summary counts
        message += f"*Buy Signals:* {len(buy_signals)}\n"
        message += f"*Sell Signals:* {len(sell_signals)}\n\n"
        
        # Add detailed signals
        if buy_signals:
            message += "*Buy:*\n"
            for signal in buy_signals:
                message += f"- {signal}\n"
            message += "\n"
        
        if sell_signals:
            message += "*Sell:*\n"
            for signal in sell_signals:
                message += f"- {signal}\n"
        
        return self.send_notification(chat_id, message)