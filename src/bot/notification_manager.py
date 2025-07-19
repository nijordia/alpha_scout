import logging
import os
from telegram import Bot
from telegram.constants import ParseMode

class NotificationManager:
    def __init__(self, bot_token=None, chat_id=None):
        # Load bot token and chat_id from environment if not provided
        self.bot_token = bot_token or os.environ.get("TELEGRAM_API_KEY")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.bot = Bot(token=self.bot_token)
        self.logger = logging.getLogger(__name__)

    async def send_notification(self, chat_id=None, message=None):
        """
        Send a notification to a specific chat ID

        Parameters:
        -----------
        chat_id : str or int
            Chat ID to send the notification to (defaults to self.chat_id)
        message : str
            Message to send

        Returns:
        --------
        bool
            True if the notification was sent successfully, False otherwise
        """
        chat_id = chat_id or self.chat_id
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification to {chat_id}: {e}")
            return False

    async def notify_signal(self, chat_id=None, stock=None, signal_type=None, signal_value=None, additional_details=None):
        """Send a signal notification"""
        chat_id = chat_id or self.chat_id
        signal_type_display = signal_type.replace("_", " ").title() if signal_type else ""
        emoji = "🟢" if signal_value == "buy" else "🔴" if signal_value == "sell" else "⚪"

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
            # Calculate vs BH if possible
            bh_return = reliability.get('buy_hold_return', None)
            if bh_return is not None:
                vs_bh = avg_return - bh_return
                vs_bh_text = f" | vs BH: {'+' if vs_bh > 0 else ''}{vs_bh:.1f}%"
            else:
                vs_bh_text = ""
            outperformance = reliability.get('market_outperformance', 0)

            message += f"\n*Signal Reliability:*\n"
            message += f"Win Rate: {win_rate:.1f}% ({period}d)\n"
            message += f"Avg Return: {'+' if avg_return > 0 else ''}{avg_return:.1f}%{vs_bh_text}\n"
            message += f"Beats Market: {outperformance:.1f}% of trades\n"

        return await self.send_notification(chat_id, message)

    async def send_daily_summary(self, chat_id=None, signals_summary=None):
        """
        Send a daily summary of signals

        Parameters:
        -----------
        chat_id : str or int
            Chat ID to send the notification to (defaults to self.chat_id)
        signals_summary : dict
            Dictionary of stocks and their signals

        Returns:
        --------
        bool
            True if the notification was sent successfully, False otherwise
        """
        chat_id = chat_id or self.chat_id
        message = "*Daily Market Signals Summary*\n\n"

        if not signals_summary:
            message += "No signals to report today."
            return await self.send_notification(chat_id, message)

        buy_signals = []
        sell_signals = []

        for stock, signals in signals_summary.items():
            for signal_type, value in signals.items():
                if value == "buy":
                    buy_signals.append(f"{stock} ({signal_type})")
                elif value == "sell":
                    sell_signals.append(f"{stock} ({signal_type})")

        message += f"*Buy Signals:* {len(buy_signals)}\n"
        message += f"*Sell Signals:* {len(sell_signals)}\n\n"

        if buy_signals:
            message += "*Buy:*\n"
            for signal in buy_signals:
                message += f"- {signal}\n"
            message += "\n"

        if sell_signals:
            message += "*Sell:*\n"
            for signal in sell_signals:
                message += f"- {signal}\n"