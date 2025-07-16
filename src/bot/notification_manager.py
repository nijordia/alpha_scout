from telegram import Bot

class NotificationManager:
    def __init__(self, bot_token, chat_id):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    def send_notification(self, message):
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"Error sending notification: {e}")

    def notify_signal(self, signal_type, signal_details):
        message = f"Market Signal Alert!\nType: {signal_type}\nDetails: {signal_details}"
        self.send_notification(message)