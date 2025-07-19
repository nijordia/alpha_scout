import logging
from telegram import Bot
from telegram.constants import ParseMode

class NotificationManager:
    def __init__(self, bot_token):
        self.bot = Bot(token=bot_token)
        self.logger = logging.getLogger(__name__)

    async def send_notification(self, chat_id, message):
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
            # Added await keyword for async API
            await self.bot.send_message(
                chat_id=chat_id, 
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification to {chat_id}: {e}")
            return False

# Update the notify_signal method in NotificationManager class

    async def notify_signal(self, chat_id, stock, signal_type, signal_value, additional_details=None):
        """Send a signal notification"""
        # Format signal type for display
        signal_type_display = signal_type.replace("_", " ").title()
        
        # Choose emoji based on signal value
        emoji = "🟢" if signal_value == "buy" else "🔴" if signal_value == "sell" else "⚪"
        
        # Build the message
        message = f"*Market Signal Alert!*\n\n"
        message += f"*Stock:* {stock}\n"
        message += f"*Signal Type:* {signal_type_display}\n"
        message += f"*Signal:* {emoji} {signal_value.upper()}\n"
        
        # Add reliability metrics if available
        if additional_details and 'reliability' in additional_details:
            reliability = additional_details['reliability']
            period = reliability.get('period', 30)
            win_rate = reliability.get('win_rate', 0)
            avg_return = reliability.get('avg_return', 0)
            outperformance = reliability.get('market_outperformance', 0)
            
            message += f"\n*Signal Reliability:*\n"
            message += f"Win Rate: {win_rate:.1f}% ({period}d)\n"
            message += f"Avg Return: {'+' if avg_return > 0 else ''}{avg_return:.1f}%\n"
            message += f"Beats Market: {outperformance:.1f}% of trades\n"
        
        # REMOVED: The additional details section that showed technical indicators
        
        return await self.send_notification(chat_id, message)
    
    async def send_daily_summary(self, chat_id, signals_summary):
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
            return await self.send_notification(chat_id, message)
        
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
        
        return await self.send_notification(chat_id, message)